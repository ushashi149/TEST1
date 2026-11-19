"""
FixProFusion — Actual AFM Inference (v2_JD)
Runs the frozen pretrained ProFusion model using real (actual) AFM images as conditioning.

Adapted from: experiments/actual_afm_v1/run_actual_afm.py
Changes:
  - collect_imgs replaces collect_pngs: accepts .png, .jpg, .jpeg files
  - IMAGE_DIR points to input_imgs folders from JD NeRF experiment
"""

import os
import sys
import cv2
import torch
import imageio
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from pytorch3d.renderer import FoVOrthographicCameras
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer.cameras import look_at_view_transform
from transformers import logging as transformers_logging

from external.nerf.network_grid import NeRFNetwork
from external.sparsefusion_utils.common_utils import get_lpips_fn, normalize, unnormalize
from external.sparsefusion_utils.render_utils import init_ray_sampler
from external.sparsefusion_utils.external_utils import PerceptualLoss

from upsrt.model.model import UpSRT
from dino.model.model import DINOv2KeyExtractor
from diffusion.pipeline_control_net import DiffusionPipelineCN
from upsrt.renderer.rays import get_grid_raybundle_gpu

N_COND = 6  # number of conditioning images ProFusion expects

IMG_EXTS = (".png", ".jpg", ".jpeg")


def collect_imgs(root):
    """Collect all PNG/JPG paths recursively, sorted for reproducibility."""
    imgs = []
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            if f.lower().endswith(IMG_EXTS):
                imgs.append(os.path.join(dirpath, f))
    return sorted(imgs)


def load_as_gray3ch(path):
    """Load any PNG/JPG → grayscale → 3-channel uint8 numpy (H, W, 3)."""
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(f"cv2 could not read: {path}")
    if x.ndim == 2:
        gray = x
    elif x.shape[2] == 4:
        # RGBA → composite over black → gray
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        bgr   = x[:, :, :3].astype(np.float32) * alpha
        gray  = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] != 256 or gray.shape[1] != 256:
        interp = cv2.INTER_AREA if gray.shape[0] > 256 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (256, 256), interpolation=interp)
    return np.stack([gray, gray, gray], axis=-1)  # (256, 256, 3)


class ActualAfmDataset(Dataset):
    """
    Loads up to N_COND actual AFM images (PNG/JPG) from image_dir (recursive).
    If fewer than N_COND images exist, repeats cyclically to reach N_COND.
    """

    def __init__(self, image_dir, save_dir):
        super().__init__()
        self.image_dir = image_dir
        image_size = [256, 256]
        min_x, min_y = -1.1, -1.1
        max_x, max_y =  1.1,  1.1
        znear, zfar  = -1.1,  1.1

        all_imgs = collect_imgs(image_dir)
        assert len(all_imgs) > 0, f"No PNG/JPG files found under: {image_dir}"

        # Pick conditioning images
        if len(all_imgs) >= N_COND:
            cond_paths = all_imgs[:N_COND]
            extra      = all_imgs[N_COND:N_COND + 1]  # one extra for query reference
        else:
            # Repeat cyclically if fewer than N_COND images
            cond_paths = [all_imgs[i % len(all_imgs)] for i in range(N_COND)]
            extra      = []

        print(f"[ActualAfmDataset] {len(all_imgs)} images found, using {N_COND} for conditioning")
        for i, p in enumerate(cond_paths):
            print(f"  cond[{i}]: {os.path.relpath(p, image_dir)}")

        os.makedirs(os.path.join(save_dir, 'input_imgs'), exist_ok=True)

        self.images = []
        for idx, path in enumerate(cond_paths):
            arr = load_as_gray3ch(path).astype(np.float32) / 255.0
            img = torch.tensor(arr)
            # Save copy for inspection
            save_path = os.path.join(save_dir, 'input_imgs', f"input_img_{idx}.jpg")
            imageio.imwrite(save_path, (arr[:, :, 0] * 255).astype(np.uint8))
            img = torch.permute(img, (2, 0, 1)).contiguous()
            img = img * 2.0 - 1.0
            self.images.append(img)

        # Save query reference if available
        if extra:
            q_arr = load_as_gray3ch(extra[0])
            cv2.imwrite(os.path.join(save_dir, 'input_imgs', 'query_reference.jpg'), q_arr[:, :, 0])

        self.cameras = self.create_synth_cameras(
            num_cameras=250, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
            znear=znear, zfar=zfar, image_size=image_size, inference=False)
        self.inference_cameras = self.create_synth_cameras(
            num_cameras=32, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
            znear=znear, zfar=zfar, image_size=image_size, inference=True)
        self.images = torch.stack(self.images, dim=0)
        self.input_cameras = self.create_cameras_with_identity_extrinsics(
            num_cameras=len(self.images),
            min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
            znear=znear, zfar=zfar, image_size=image_size)

    @staticmethod
    def create_synth_cameras(num_cameras, min_x, min_y, max_x, max_y,
                              znear, zfar, image_size, inference=False):
        base_elevation = 22.5
        min_ele, max_ele = base_elevation - 30.0, base_elevation + 30.0
        synth_cameras = []
        if inference:
            azimuths_   = torch.linspace(0.0, 360.0, num_cameras + 1)[:-1]
            elevations_ = torch.ones(num_cameras) * base_elevation
        for i in range(num_cameras):
            if inference:
                distance_choice  = 2.0
                azimuth_choice   = azimuths_[i]
                elevation_choice = elevations_[i]
                roll_choice      = 180.0
            else:
                if i == 0:
                    azimuth_choice, elevation_choice = 0.0, base_elevation
                    distance_choice, roll_choice = 2.0, 0.0
                else:
                    azimuth_choice   = np.random.random() * 360.0
                    elevation_choice = np.random.random() * (max_ele - min_ele) + min_ele
                    roll_choice      = np.random.randint(0, 360)
                    distance_choice  = 2.0
            R, T = look_at_view_transform(dist=distance_choice, elev=elevation_choice,
                                          azim=azimuth_choice, degrees=True)
            roll_rad = torch.tensor([[0, 0, roll_choice]]) * (torch.pi / 180.0)
            R_roll   = euler_angles_to_matrix(roll_rad, "XYZ")
            R        = torch.bmm(R, R_roll)
            camera   = FoVOrthographicCameras(R=R, T=T,
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, znear=znear, zfar=zfar)
            synth_cameras.append(camera)
        return synth_cameras

    @staticmethod
    def create_cameras_with_identity_extrinsics(num_cameras, min_x, min_y, max_x, max_y,
                                                 znear, zfar, image_size):
        cameras = []
        for _ in range(num_cameras):
            camera = FoVOrthographicCameras(
                R=torch.eye(3, dtype=torch.float32).unsqueeze(0),
                T=torch.zeros((1, 3), dtype=torch.float32),
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                znear=znear, zfar=zfar)
            cameras.append(camera)
        return cameras

    def __getitem__(self, idx):
        return (self.images, self.cameras, self.input_cameras, self.inference_cameras)


class CachedQueryDataset(torch.utils.data.Dataset):
    def __init__(self, query_cameras, cache):
        self.query_cameras = query_cameras[0]
        self.cache = cache

    def __len__(self):
        return len(self.query_cameras)

    @staticmethod
    def collate_fn(batch):
        batched_cameras = concat_cameras([x["query_cameras"] for x in batch])
        return {
            "slt":           torch.cat([x["slt"]           for x in batch], dim=0),
            "cond_images":   torch.cat([x["cond_images"]   for x in batch], dim=0),
            "query_rgb_256": torch.cat([x["query_rgb_256"] for x in batch], dim=0),
            "query_cameras": batched_cameras,
        }

    def __getitem__(self, idx):
        return {
            "slt":           self.cache[idx]["slt"],
            "cond_images":   self.cache[idx]["cond_images"],
            "query_rgb_256": self.cache[idx]["query_rgb_256"],
            "query_cameras": self.query_cameras[idx],
        }


def get_cfg(cfg_path):
    return OmegaConf.load(cfg_path)

def save_image(path, tensor, unnorm=False):
    img = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    if unnorm: img = img * 0.5 + 0.5
    img = np.clip(img, 0.0, 1.0)
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)

def _collect_attr(cams, attr):
    return torch.cat([getattr(x, attr) for x in cams], dim=0)

def concat_cameras(list_of_cameras):
    return FoVOrthographicCameras(
        R  =_collect_attr(list_of_cameras, "R"),
        T  =_collect_attr(list_of_cameras, "T"),
        min_x=_collect_attr(list_of_cameras, "min_x"),
        max_x=_collect_attr(list_of_cameras, "max_x"),
        min_y=_collect_attr(list_of_cameras, "min_y"),
        max_y=_collect_attr(list_of_cameras, "max_y"),
    )

def batched_cameras_to_list(batched_cameras):
    cameras = []
    for i in range(len(batched_cameras)):
        cam = FoVOrthographicCameras(
            R=batched_cameras.R[i:i+1], T=batched_cameras.T[i:i+1],
            min_x=batched_cameras.min_x[i:i+1], max_x=batched_cameras.max_x[i:i+1],
            min_y=batched_cameras.min_y[i:i+1], max_y=batched_cameras.max_y[i:i+1])
        cameras.append(cam)
    return cameras

def _prepare_condition(srt_model, dino_model, input_views, input_cameras, query_cameras):
    dino_features = dino_model(input_views)
    query_features, plucker_encoding, slt = srt_model.get_query_features(
        dino_features=dino_features, input_cameras=input_cameras,
        query_cameras=query_cameras, image_size=(32, 32),
        decoder_return_type="pre_rgb_mlp_features",
        return_grid_rays=True, return_slt=True)
    cond_images = torch.cat((query_features, plucker_encoding), dim=3)
    return (cond_images.detach().cpu(), slt.detach().cpu()), dino_features

def get_default_ngp_opt():
    opt = argparse.Namespace()
    opt.cuda_ray = True
    opt.bg_radius = 0
    opt.density_thresh = 10
    opt.bound = 1
    opt.min_near = 0.05
    return opt


def distillation_loop(gpu, cfg, args, opt, model_tuple, save_dir, seq_name):
    print("[***] Preparing training setup")
    lambda_opacity = 1e-3
    lambda_entropy = 0.0
    lambda_percep  = 0.1
    plot_log_freq  = 20
    img_log_freq   = 100
    hw_scale       = 2.0
    gradient_accumulation_steps = 1
    sds_bootstrap_itrs = 300
    start_percep_step  = -1
    bg_color_choice    = 0.0
    max_itr            = 3000

    perceptual_loss = PerceptualLoss('vgg', device=f'cuda:{gpu}')
    fusion_loss_list = []

    os.makedirs(os.path.join(save_dir, "render_imgs", seq_name), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "render_gifs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "log"), exist_ok=True)
    srt_model, dino_model, diffusion_pipeline = model_tuple

    dataset = ActualAfmDataset(image_dir=args.image_dir, save_dir=save_dir)
    custom_imgs, custom_cameras, input_cameras, inference_cameras = dataset[0]

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    diffusion_pipeline.handle, srt_model, dino_model = accelerator.prepare(
        diffusion_pipeline.handle, srt_model, dino_model)

    input_views    = custom_imgs[None].to(gpu)
    input_cameras  = [[x.to(f"cuda:{gpu}") for x in input_cameras]]
    custom_cameras = [x.to(f"cuda:{gpu}") for x in custom_cameras]
    inference_cameras = [x.to(f"cuda:{gpu}") for x in inference_cameras]

    batched_custom_cameras = concat_cameras(custom_cameras)
    cam_dist_max    = torch.max(torch.linalg.norm(batched_custom_cameras.get_camera_center(), axis=1))
    min_depth       = cam_dist_max * 0.1
    volume_extent   = cam_dist_max * 1.8
    sampler_grid, _, sampler_feat = init_ray_sampler(gpu, 256, 256, min=min_depth, max=volume_extent, scale_factor=hw_scale)

    cache = dict()
    for q_idx in tqdm(range(len(custom_cameras)), desc="Caching condition inputs"):
        with torch.no_grad():
            condition_, dino_features_ = _prepare_condition(
                srt_model, dino_model,
                input_views=input_views[:1],
                input_cameras=input_cameras[:1],
                query_cameras=[custom_cameras[q_idx]])
            query_rgb_256, _ = srt_model.get_query_features(
                dino_features=dino_features_, input_cameras=input_cameras[:1],
                query_cameras=[custom_cameras[q_idx]], image_size=(256, 256),
                decoder_return_type="pred_rgb", return_grid_rays=False)
            cond_images_, slt_ = condition_
            cache[q_idx] = {
                "cond_images":   cond_images_, "slt": slt_,
                "query_rgb_256": torch.permute(query_rgb_256, (0, 3, 1, 2)).contiguous().cpu()}

    custom_cameras_cpu = [x.cpu() for x in custom_cameras]
    query_dataset = CachedQueryDataset(query_cameras=[custom_cameras_cpu], cache=cache)
    query_loader  = DataLoader(dataset=query_dataset, batch_size=1, shuffle=True,
                               num_workers=0, pin_memory=True,
                               collate_fn=CachedQueryDataset.collate_fn)
    query_loader  = accelerator.prepare(query_loader)
    query_itr     = iter(query_loader)

    ngp_network = NeRFNetwork(opt).cuda(gpu).train()
    optimizer   = torch.optim.Adam(ngp_network.get_params(lr=5e-4))
    optimizer   = accelerator.prepare(optimizer)
    print("[***] Training setup prepared!")

    for itr in tqdm(range(max_itr), desc="Training"):
        lambda_percep_ = lambda_percep if itr >= start_percep_step else 0.0
        with accelerator.accumulate(ngp_network):
            total_loss, sds_loss = 0.0, 0.0
            ngp_network.train()
            if opt.cuda_ray and itr % 16 == 0:
                ngp_network.update_extra_state()
            try:
                batch = next(query_itr)
            except StopIteration:
                query_itr = iter(query_loader)
                batch = next(query_itr)

            batch_slt         = batch["slt"]
            batch_cameras     = batch["query_cameras"]
            batch_cond_images = batch["cond_images"]
            ray_bundle = get_grid_raybundle_gpu(batch_cameras, image_size=(128, 128))
            H, W = 128, 128
            rays_o = ray_bundle.origins
            rays_d = ray_bundle.directions
            B, N   = rays_o.shape[:2]
            bg_color = bg_color_choice
            outputs  = ngp_network.render(rays_o, rays_d, staged=False, perturb=True,
                bg_color=bg_color, ambient_ratio=1.0, shading='albedo',
                force_all_rays=True, **vars(opt))
            rendered_images = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            rendered_sil    = outputs['weights_sum'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            rendered_images = F.interpolate(rendered_images, scale_factor=hw_scale, mode='bilinear')
            rendered_sil    = F.interpolate(rendered_sil,    scale_factor=hw_scale, mode='bilinear')

            ddim_steps = 30
            t_end   = ddim_steps - 1
            t_start = ddim_steps - 5 if itr < sds_bootstrap_itrs else 1
            with torch.no_grad():
                srt_cond = (batch_cond_images, batch_slt, [batch_cameras])
                normed_pred, alpha_cumprod = diffusion_pipeline.forward_multi_step_denoise(
                    clean_data=normalize(rendered_images), srt_cond=srt_cond, batch_size=1,
                    unconditional_guidance_scale=cfg.diffusion.model.unconditional_guidance_scale,
                    ddim_steps=ddim_steps, t_start=t_start, t_end=t_end)
                pred_img = unnormalize(normed_pred).clip(0.0, 1.0)

            if itr % 200 == 0:
                imageio.imwrite(os.path.join(save_dir, "log", f"rendered_{str(itr).zfill(5)}.jpg"),
                    (rendered_images.permute(0,2,3,1).detach().cpu().numpy()[0]*255).astype(np.uint8))
                imageio.imwrite(os.path.join(save_dir, "log", f"pred_{str(itr).zfill(5)}.jpg"),
                    (pred_img.permute(0,2,3,1).detach().cpu().numpy()[0]*255).astype(np.uint8))

            fusion_weight = (1 - alpha_cumprod).to(pred_img.device)
            if itr < sds_bootstrap_itrs:
                fusion_weight = 1.0
            sds_loss = fusion_weight * ((rendered_images - pred_img)**2).mean()
            if lambda_percep_ > 0.0:
                sds_loss += perceptual_loss(rendered_images, pred_img, normalize=True).mean() * lambda_percep_
            opacity_term  = torch.sqrt((rendered_sil ** 2) + .01).mean() if lambda_opacity > 0 else torch.zeros(1).cuda(gpu)
            entropy_term  = torch.zeros(1).cuda(gpu)
            total_loss    = sds_loss + lambda_opacity * opacity_term + lambda_entropy * entropy_term
            fusion_loss_list.append(sds_loss.item())
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

        if itr % plot_log_freq == 0:
            plt.plot(list(range(len(fusion_loss_list))), fusion_loss_list, linewidth=1)
            plt.savefig(os.path.join(save_dir, "log", f"{seq_name}_fusionloss.jpg"))
            plt.cla(); plt.close()

        if itr % img_log_freq == 0:
            with torch.no_grad():
                ngp_network.eval()
                ray_bundle = get_grid_raybundle_gpu(batch_cameras, image_size=(128, 128))
                rays_o, rays_d = ray_bundle.origins, ray_bundle.directions
                B, _ = rays_o.shape[:2]
                outputs = ngp_network.render_batched(rays_o, rays_d, batched=True, perturb=True,
                    bg_color=bg_color, ambient_ratio=1.0, shading='albedo',
                    force_all_rays=True, **vars(opt))
                rendered_v = outputs['image'].reshape(B, 128, 128, 3).contiguous()[0].detach().cpu().numpy()
                rendered_s = outputs['weights_sum'].reshape(B, 128, 128, 1).expand(-1,-1,-1,3).contiguous()[0].detach().cpu().numpy()
            render_vis = np.hstack((rendered_v, rendered_s))
            imageio.imwrite(os.path.join(save_dir, "log", f"{seq_name}_vis_{str(itr).zfill(5)}.jpg"),
                (render_vis * 255).astype(np.uint8))

    circle_rgb_list, circle_sil_list = [], []
    for ci in tqdm(range(len(inference_cameras)), desc="Rendering orbit"):
        render_camera = inference_cameras[ci].to(f"cuda:{gpu}")
        with torch.no_grad():
            ngp_network.eval()
            ray_bundle = get_grid_raybundle_gpu(render_camera, image_size=(128, 128))
            rays_o, rays_d = ray_bundle.origins, ray_bundle.directions
            B, _ = rays_o.shape[:2]
            outputs = ngp_network.render_batched(rays_o, rays_d, batched=True, perturb=True,
                bg_color=bg_color, ambient_ratio=1.0, shading='albedo',
                force_all_rays=True, **vars(opt))
            rendered_rgb = outputs['image'].reshape(B, 128, 128, 3)[0].detach().cpu().numpy()
            rendered_sil = outputs['weights_sum'].reshape(B, 128, 128, 1).expand(-1,-1,-1,3)[0].detach().cpu().numpy()
        circle_rgb_list.append((rendered_rgb * 255.0).astype(np.uint8))
        circle_sil_list.append((rendered_sil * 255.0).astype(np.uint8))

    fps = 10.0
    imageio.mimwrite(os.path.join(save_dir, "render_gifs", f"{seq_name}_rgb.gif"),
        circle_rgb_list, duration=1000*(1/fps), loop=0)
    imageio.mimwrite(os.path.join(save_dir, "render_gifs", f"{seq_name}_sil.gif"),
        circle_sil_list, duration=1000*(1/fps), loop=0)

    model_path = os.path.join(save_dir, f"{seq_name}.pt")
    torch.save({'model_state_dict': ngp_network.state_dict()}, model_path)
    print(f"[***] NeRF model saved to {model_path}")


def run(args):
    gpu      = 0
    seq_name = f"{args.protein_name}_actual"
    save_dir = os.path.join(args.out_dir, seq_name)
    for sub in ['', 'log', 'metrics', 'render_imgs', 'render_gifs', 'input_imgs']:
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    cfg = get_cfg(args.config_path)
    opt = get_default_ngp_opt()

    print("[***] Setting up models")
    srt_model          = UpSRT(cfg.srt.model)
    dino_model         = DINOv2KeyExtractor(cfg.dino.model)
    diffusion_pipeline = DiffusionPipelineCN(cfg.diffusion.model, srt_model=srt_model, dino_model=dino_model)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    srt_model.load_state_dict(
        {k.replace('UpSRT.', ''): v for k, v in ckpt['state_dict'].items() if 'UpSRT' in k})
    dino_model.load_state_dict(
        {k.replace('feature_extractor.', ''): v for k, v in ckpt['state_dict'].items() if 'feature_extractor' in k})
    diffusion_pipeline.handle.load_state_dict(
        {k.replace('diffusion_model.handle.', ''): v for k, v in ckpt['state_dict'].items() if 'diffusion_model.handle' in k})
    print(f"[***] Loaded checkpoint: {args.ckpt_path}")

    srt_model.eval()
    dino_model.eval()
    diffusion_pipeline.handle.eval()
    model_tuple = (srt_model, dino_model, diffusion_pipeline)
    print("[***] Model setup complete!")

    distillation_loop(gpu=gpu, cfg=cfg, args=args, opt=opt,
                      model_tuple=model_tuple, save_dir=save_dir, seq_name=seq_name)


def main():
    parser = argparse.ArgumentParser(description="FixProFusion — actual AFM inference v2_JD")
    parser.add_argument('--protein-name', required=True,
                        help='Protein name, e.g. Actual_WRAC')
    parser.add_argument('--image-dir', required=True,
                        help='Directory containing actual AFM images (PNG/JPG, searched recursively)')
    parser.add_argument('--out-dir', required=True,
                        help='Output root directory')
    parser.add_argument('--ckpt-path', required=True,
                        help='Path to ProFusion checkpoint (.ckpt)')
    parser.add_argument('--config-path', required=True,
                        help='Path to inference.yaml config')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    transformers_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    main()
