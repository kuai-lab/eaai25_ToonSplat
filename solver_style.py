from matplotlib import pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle
import imageio
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import os, os.path as osp, shutil, sys
from transforms3d.euler import euler2mat
from omegaconf import OmegaConf
from transformers import AutoImageProcessor, AutoModel

from lib_data.get_data import prepare_real_seq_style
from lib_data.data_provider import DatabasePoseProvider

from lib_toonsplat.templates import get_template
from lib_toonsplat.model import GaussianTemplateModelStyle, AdditionalBones

from lib_toonsplat.optim_utils import *
from lib_render.gauspl_renderer import render_cam_pcl

# from lib_marchingcubes.gaumesh_utils import MeshExtractor
from lib_toonsplat.model_utils import transform_mu_frame

from utils.misc import *
from utils.viz import viz_render

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
import time

from lib_render.camera_sampling import sample_camera, fov2K, opencv2blender
import logging

from viz_utils_style import viz_spinning, viz_human_all
from utils.ssim import ssim

import argparse
from datetime import datetime
from test_utils.test_func_style import test

from pytorch_wavelets import DWTForward
import torch_dct as dct
            
import pdb
import wandb



class TGFitter:
    def __init__(
        self,
        log_dir,
        profile_fn,
        mode,
        template_model_path="data/smpl_model/SMPL_NEUTRAL.pkl",
        device=torch.device("cuda:0"),
        **kwargs,
    ) -> None:
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.profile_fn = profile_fn
        try:
            shutil.copy(profile_fn, osp.join(self.log_dir, osp.basename(profile_fn)))
        except:
            pass

        self.mode = mode
        assert self.mode in ["human"]

        self.template_model_path = template_model_path
        self.device = device

        # * auto set attr
        cfg = OmegaConf.load(profile_fn)
        # assign the cfg to self attribute
        for k, v in cfg.items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            setattr(self, k, v)
        seed_everything(self.SEED)
        
        

        # * explicitly set flags
        self.FAST_TRAINING = getattr(self, "FAST_TRAINING", False)

        self.LAMBDA_SSIM = getattr(self, "LAMBDA_SSIM", 0.0)
        self.LAMBDA_LPIPS = getattr(self, "LAMBDA_LPIPS", 0.0)
        if self.LAMBDA_LPIPS > 0:
            from utils.lpips import LPIPS

            self.lpips = LPIPS(net="vgg").to(self.device)
            for param in self.lpips.parameters():
                param.requires_grad = False

        if isinstance(self.RESET_OPACITY_STEPS, int):
            self.RESET_OPACITY_STEPS = [
                i
                for i in range(1, self.TOTAL_steps)
                if i % self.RESET_OPACITY_STEPS == 0
            ]
        if isinstance(self.REGAUSSIAN_STEPS, int):
            self.REGAUSSIAN_STEPS = [
                i for i in range(1, self.TOTAL_steps) if i % self.REGAUSSIAN_STEPS == 0
            ]
        self.normal_loss = torch.nn.MSELoss()
        self.normal_sc_loss = torch.nn.MSELoss()
        self.style_loss = torch.nn.MSELoss()
        
        # self.processor_style = AutoImageProcessor.from_pretrained('facebook/dinov2-base').to_device('cuda')
        self.model_style = AutoModel.from_pretrained('facebook/dinov2-base')
        self.model_style.to("cuda")
        for param in self.model_style.parameters():
            param.requires_grad = False

        # prepare base rotation # self.viz_base_R: 3x3
        if self.mode == "human":
            viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
        viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
        self.viz_base_R = viz_base_R_opencv.to(self.device)


        if self.mode == "human":
            self.reg_base_R_global = (
                matrix_to_axis_angle(
                    torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[
                        None
                    ]
                )[0]
                .float()
                .to(self.device)
            )

        self.writer = create_log(
            self.log_dir, name=osp.basename(self.profile_fn).split(".")[0], debug=False
        )
        return


    def prepare_real_seq(
        self,
        seq_name,
        dataset_mode,
        split,
        ins_avt_wild_start_end_skip=None,
        image_zoom_ratio=0.5,
        data_stay_gpu_flag=True,
    ):
        provider, dataset = prepare_real_seq_style(
            seq_name=seq_name,
            dataset_mode=dataset_mode,
            split=split,
            ins_avt_wild_start_end_skip=ins_avt_wild_start_end_skip,
            image_zoom_ratio=getattr(
                self, "IMAGE_ZOOM_RATIO", image_zoom_ratio
            ),  # ! this overwrite the func arg
            balance=getattr(self, "VIEW_BALANCE_FLAG", False),
        )
        provider.to(self.device)
        if getattr(self, "DATA_STAY_GPU_FLAG", data_stay_gpu_flag):
            provider.move_images_to_device(self.device)
        provider.viz_selection_prob(
            osp.join(self.log_dir, f"split_{split}_view_prob.png")
        )
        return provider, dataset

    def load_saved_model(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = osp.join(self.log_dir, "model.pth")
        ret = self._get_model_optimizer(betas=None)
        model = ret[0]

        model.load(torch.load(ckpt_path))
        model.to(self.device)
        model.eval()
        logging.info("After loading:")
        model.summary()
        return model

    def _get_model_optimizer(self, betas, add_bones_total_t=0):
        seed_everything(self.SEED)

        template = get_template(
            mode=self.mode,
            template_model_path=self.template_model_path,
            init_beta=betas,
            cano_pose_type=getattr(self, "CANO_POSE_TYPE", "t_pose"),
            voxel_deformer_res=getattr(self, "VOXEL_DEFORMER_RES", 64),
        )

        add_bones = AdditionalBones(
            num_bones=getattr(self, "W_REST_DIM", 0),
            mode=getattr(self, "W_REST_MODE", "pose-mlp"),
            total_t=add_bones_total_t,
            # pose mlp
            mlp_hidden_dims=getattr(
                self, "ADD_BONES_MLP_HIDDEN_DIMS", [256, 256, 256, 256]
            ),
            pose_dim=23 * 3 if self.mode == "human" else 34 * 3 + 7,
        )

        model = GaussianTemplateModelStyle(
            template=template,
            add_bones=add_bones,
            w_correction_flag=getattr(self, "W_CORRECTION_FLAG", False),
            # w_rest_dim=getattr(self, "W_REST_DIM", 0),
            f_localcode_dim=getattr(self, "F_LOCALCODE_DIM", 0),
            max_sph_order=getattr(self, "MAX_SPH_ORDER", 0),
            w_memory_type=getattr(self, "W_MEMORY_TYPE", "point"),
            max_scale=getattr(self, "MAX_SCALE", 0.1),
            min_scale=getattr(self, "MIN_SCALE", 0.0),
            # * init
            init_mode=getattr(self, "INIT_MODE", "on_mesh"),
            opacity_init_value=getattr(self, "OPACITY_INIT_VALUE", 0.9),
            # on mesh init
            onmesh_init_subdivide_num=getattr(self, "ONMESH_INIT_SUBDIVIDE_NUM", 0),
            onmesh_init_scale_factor=getattr(self, "ONMESH_INIT_SCALE_FACTOR", 1.0),
            onmesh_init_thickness_factor=getattr(
                self, "ONMESH_INIT_THICKNESS_FACTOR", 0.5
            ),
            # near mesh init
            nearmesh_init_num=getattr(self, "NEARMESH_INIT_NUM", 1000),
            nearmesh_init_std=getattr(self, "NEARMESH_INIT_STD", 0.5),
            scale_init_value=getattr(self, "SCALE_INIT_VALUE", 1.0),
        ).to(self.device)

        logging.info(f"Init with {model.N} Gaussians")

        # * set optimizer
        LR_SPH_REST = getattr(self, "LR_SPH_REST", self.LR_SPH / 20.0)
        optimizer = torch.optim.Adam(
            model.get_optimizable_list(
                lr_p=self.LR_P,
                lr_o=self.LR_O,
                lr_s=self.LR_S,
                lr_q=self.LR_Q,
                lr_sph=self.LR_SPH,
                lr_sph_rest=LR_SPH_REST,
                lr_w=self.LR_W,
                lr_w_rest=self.LR_W_REST,
                lr_f=getattr(self, "LR_F_LOCAL", 0.0),
            ),
            weight_decay=self.WEIGHT_DECAY
        )

        xyz_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_P,
            lr_final=self.LR_P_FINAL,
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )
        w_dc_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W,
            lr_final=getattr(self, "LR_W_FINAL", self.LR_W),
            lr_delay_mult=0.01,  # 0.02
        )
        w_rest_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W_REST,
            lr_final=getattr(self, "LR_W_REST_FINAL", self.LR_W_REST),
            lr_delay_mult=0.01,  # 0.02
        )
        sph_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_SPH,
            lr_final=getattr(self, "LR_SPH_FINAL", self.LR_SPH),
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )
        sph_rest_scheduler_func = get_expon_lr_func(
            lr_init=LR_SPH_REST,
            lr_final=getattr(self, "LR_SPH_REST_FINAL", LR_SPH_REST),
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )

        return (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        )

    def _get_pose_optimizer(self, data_provider, add_bones):
        if data_provider is None and add_bones.num_bones == 0:
            # dummy one
            return torch.optim.Adam(params=[torch.zeros(1, requires_grad=True)]), {}

        # * prepare pose optimizer list and the schedulers
        scheduler_dict, pose_optim_l = {}, []
        if data_provider is not None:
            start_step = getattr(self, "POSE_OPTIMIZE_START_STEP", 0)
            end_step = getattr(self, "POSE_OPTIMIZE_END_STEP", self.TOTAL_steps)
            pose_optim_l.extend(
                [
                    {
                        "params": data_provider.pose_base_list,
                        "lr": self.POSE_R_BASE_LR,
                        "name": "pose_base",
                    },
                    {
                        "params": data_provider.pose_rest_list,
                        "lr": self.POSE_R_REST_LR,
                        "name": "pose_rest",
                    },
                    {
                        "params": data_provider.global_trans_list,
                        "lr": self.POSE_T_LR,
                        "name": "pose_trans",
                    },
                ]
            )
            scheduler_dict["pose_base"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_BASE_LR,
                lr_final=getattr(self, "POSE_R_BASE_LR_FINAL", self.POSE_R_BASE_LR),
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_rest"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_REST_LR,
                lr_final=getattr(self, "POSE_R_REST_LR_FINAL", self.POSE_R_REST_LR),
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_trans"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_T_LR,
                lr_final=getattr(self, "POSE_T_LR_FINAL", self.POSE_T_LR),
                lr_delay_mult=0.01,  # 0.02
            )
        if add_bones.num_bones > 0:
            # have additional bones
            pose_optim_l.append(
                {
                    "params": add_bones.parameters(),
                    "lr": getattr(self, "LR_W_REST_BONES", 0.0),
                    "name": "add_bones",
                }
            )

        pose_optim_mode = getattr(self, "POSE_OPTIM_MODE", "adam")
        if pose_optim_mode == "adam":
            optimizer_smpl = torch.optim.Adam(pose_optim_l)
        elif pose_optim_mode == "sgd":
            optimizer_smpl = torch.optim.SGD(pose_optim_l)
        else:
            raise NotImplementedError(f"Unknown pose optimizer mode {pose_optim_mode}")
        return optimizer_smpl, scheduler_dict

    def is_pose_step(self, step):
        flag = (
            step >= self.POSE_START_STEP
            and step <= self.POSE_END_STEP
            and step % self.POSE_STEP_INTERVAL == 0
        )
        return flag

    def _fit_step(
        self,
        model,
        data_pack,
        act_sph_ord,
        random_bg=True,
        scale_multiplier=1.0,
        opacity_multiplier=1.0,
        opa_th=-1,
        use_box_crop_pad=-1,
        default_bg=[0.0, 0.0, 0.0],
        add_bones_As=None,
    ):
        gt_rgb, gt_mask, gt_normal, gt_depth, K, pose_base, pose_rest, global_trans, time_index, style_feature = data_pack
        gt_rgb = gt_rgb.clone()

        pose = torch.cat([pose_base, pose_rest], dim=1)
        H, W = gt_rgb.shape[1:3]
        additional_dict = {"t": time_index}
        if add_bones_As is not None:
            additional_dict["As"] = add_bones_As
        mu, fr, sc, op, normal, normal_sc, sph, additional_ret = model(
            pose, # ([1, 24, 3])
            global_trans, # ([1, 3])
            style_feature,
            additional_dict=additional_dict, # {"t": time_index}
            active_sph_order=act_sph_ord,
        )

        sc = sc * scale_multiplier
        op = op * opacity_multiplier

        loss_recon = 0.0
        loss_normal = 0.0
        loss_normal_sc = 0.0
        loss_normal_both = 0.0
        loss_mask = 0.0
        loss_depth = 0.0
        loss_style = 0.0
        loss_dwt = 0.0
        loss_dct = 0.0
        loss_fft = 0.0
        loss_lpips, loss_ssim = (
            torch.zeros(1).to(gt_rgb.device).squeeze(),
            torch.zeros(1).to(gt_rgb.device).squeeze(),
        )
        
        render_pkg_list, rgb_target_list, mask_target_list, normal_target_list, depth_target_list = [], [], [], [], []
        for i in range(len(gt_rgb)):
            if random_bg:
                bg = np.random.uniform(0.0, 1.0, size=3)
            else:
                bg = np.array(default_bg)
            bg_tensor = torch.from_numpy(bg).float().to(gt_rgb.device)
            gt_rgb[i][gt_mask[i] == 0] = bg_tensor[None, None, :]
            render_pkg = render_cam_pcl(
                mu[i], fr[i], sc[i], op[i],normal[i], normal_sc[i], sph[i], style_feature, H, W, K[i], False, act_sph_ord, bg
            )
            if opa_th > 0.0:
                bg_mask = render_pkg["alpha"][0] < opa_th
                render_pkg["rgb"][:, bg_mask] = (
                    render_pkg["rgb"][:, bg_mask] * 0.0 + default_bg[0]
                )

            # * lpips before the pad
            if self.LAMBDA_LPIPS > 0:
                _loss_lpips = self.lpips(
                    render_pkg["rgb"][None],
                    gt_rgb.permute(0, 3, 1, 2),
                )
                loss_lpips = loss_lpips + _loss_lpips
                
            

            if use_box_crop_pad > 0:
                # pad the gt mask and crop the image, for a large image with a small fg object
                _yl, _yr, _xl, _xr = get_bbox(gt_mask[i], use_box_crop_pad)
                for key in ["rgb", "alpha", "dep", "normal", "normal_sc"]:
                    render_pkg[key] = render_pkg[key][:, _yl:_yr, _xl:_xr]
                rgb_target = gt_rgb[i][_yl:_yr, _xl:_xr]
                mask_target = gt_mask[i][_yl:_yr, _xl:_xr]
                normal_target = gt_normal[i][_yl:_yr, _xl:_xr]
                depth_target = gt_depth[i][_yl:_yr, _xl:_xr]
            else:
                rgb_target = gt_rgb[i]
                mask_target = gt_mask[i]
                normal_target = gt_normal[i]
                depth_target = gt_depth[i]
            depth_target = depth_target.unsqueeze(-1)
           
            
            
            
            
            
            dwt = DWTForward(wave='bior4.4', J=2, mode='periodization').to(rgb_target.device)
            yl, yh = dwt(render_pkg["rgb"].unsqueeze(0))
            yl_gt, yh_gt = dwt(rgb_target.permute(2,0,1).unsqueeze(0))

            dct_ = dct.dct_2d(render_pkg["rgb"].permute(1, 2, 0))
            dct_gt = dct.dct_2d(rgb_target)

            fft = torch.fft.fft2(render_pkg["rgb"].permute(1, 2, 0), norm="ortho")
            fft_gt = torch.fft.fft2(rgb_target, norm="ortho")

            
            _loss_dwt = abs(yl - yl_gt).mean() + abs(yh[1][:,:,2,:,:] - yh_gt[1][:,:,2,:,:]).mean()
            _loss_dct = abs(dct_ - dct_gt).mean()
            _loss_fft = abs(fft - fft_gt).mean()


            # * standard recon loss
            _loss_recon = abs(render_pkg["rgb"].permute(1, 2, 0) - rgb_target).mean()
            _loss_mask = abs(render_pkg["alpha"].squeeze(0) - mask_target).mean()
            _loss_normal = self.normal_loss(render_pkg["normal"].permute(1, 2, 0), normal_target)
            _loss_normal_sc = self.normal_sc_loss(render_pkg["normal_sc"].permute(1, 2, 0), normal_target)
            _loss_normal_both = abs(render_pkg["normal"].permute(1, 2, 0) - render_pkg["normal_sc"].permute(1, 2, 0)).mean()
            _loss_depth = abs(render_pkg['dep'].permute(1, 2, 0) - depth_target).mean()
            inputs = F.interpolate(render_pkg["rgb"].unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

            
            outputs = self.model_style(inputs)
            style_feature_hat = outputs.last_hidden_state # torch.Size([1, 257, 768])
            _loss_style = self.style_loss(style_feature_hat, style_feature.to(style_feature_hat))
            
            
            
            # * ssim
            if self.LAMBDA_SSIM > 0:
                _loss_ssim = 1.0 - ssim(
                    render_pkg["rgb"][None],
                    rgb_target.permute(2, 0, 1)[None],
                )
                loss_ssim = loss_ssim + _loss_ssim

            loss_recon = loss_recon + _loss_recon
            loss_mask = loss_mask + _loss_mask
            loss_normal = loss_normal + _loss_normal
            loss_normal_sc = loss_normal_sc + _loss_normal_sc
            loss_normal_both = loss_normal_both + _loss_normal_both
            loss_dwt = loss_dwt + _loss_dwt
            loss_dct = loss_dct + _loss_dct
            loss_fft = loss_fft + _loss_fft
            loss_depth = loss_depth + _loss_depth
            loss_style = loss_style + _loss_style
            
            render_pkg_list.append(render_pkg)
            rgb_target_list.append(rgb_target)
            mask_target_list.append(mask_target)
            normal_target_list.append(normal_target)
            depth_target_list.append(depth_target)
        
        loss_scale_flat = abs(torch.min(sc[i], dim=1).values).mean()

        loss_recon = loss_recon / len(gt_rgb)
        loss_mask = loss_mask / len(gt_rgb)
        loss_normal = loss_normal / len(gt_rgb)
        loss_normal_sc = loss_normal_sc / len(gt_rgb)
        loss_normal_both = loss_normal_both / len(gt_rgb)
        loss_dwt = loss_dwt / len(gt_rgb)
        loss_dct = loss_dct / len(gt_rgb)
        loss_fft = loss_fft / len(gt_rgb)
        loss_depth = loss_depth / len(gt_rgb)
        loss_style = loss_style / len(gt_rgb)
        loss_lpips = loss_lpips / len(gt_rgb)
        loss_ssim = loss_ssim / len(gt_rgb)

        loss = (
            loss_recon + self.LAMBDA_SSIM * loss_ssim + self.LAMBDA_LPIPS * loss_lpips
        )

        return (
            loss,
            loss_mask,
            loss_normal,
            loss_normal_sc,
            loss_normal_both,
            loss_dwt,
            loss_dct,
            loss_fft,
            loss_depth,
            loss_style,
            loss_scale_flat,
            render_pkg_list,
            rgb_target_list,
            mask_target_list,
            normal_target_list,
            depth_target_list,
            (mu, fr, sc, op, sph, normal, normal_sc),
            {
                "loss_l1_recon": loss_recon,
                "loss_lpips": loss_lpips,
                "loss_ssim": loss_ssim,
            },
        )

    def compute_reg3D(self, model: GaussianTemplateModelStyle):
        K = getattr(self, "CANONICAL_SPACE_REG_K", 10)
        (
            q_std,
            s_std,
            o_std,
            cd_std,
            ch_std,
            w_std,
            w_rest_std,
            f_std,
            w_norm,
            w_rest_norm,
            dist_sq,
            max_s_sq,
            nn_ind,
        ) = model.compute_reg(K)

        lambda_std_q = getattr(self, "LAMBDA_STD_Q", 0.0)
        lambda_std_s = getattr(self, "LAMBDA_STD_S", 0.0)
        lambda_std_o = getattr(self, "LAMBDA_STD_O", 0.0)
        lambda_std_cd = getattr(self, "LAMBDA_STD_CD", 0.0)
        lambda_std_ch = getattr(self, "LAMBDA_STD_CH", lambda_std_cd)
        lambda_std_w = getattr(self, "LAMBDA_STD_W", 0.3)
        lambda_std_w_rest = getattr(self, "LAMBDA_STD_W_REST", lambda_std_w)
        lambda_std_f = getattr(self, "LAMBDA_STD_F", lambda_std_w)

        lambda_w_norm = getattr(self, "LAMBDA_W_NORM", 0.1)
        lambda_w_rest_norm = getattr(self, "LAMBDA_W_REST_NORM", lambda_w_norm)

        lambda_knn_dist = getattr(self, "LAMBDA_KNN_DIST", 0.0)

        lambda_small_scale = getattr(self, "LAMBDA_SMALL_SCALE", 0.0)

        reg_loss = (
            lambda_std_q * q_std
            + lambda_std_s * s_std
            + lambda_std_o * o_std
            + lambda_std_cd * cd_std
            + lambda_std_ch * ch_std
            + lambda_knn_dist * dist_sq
            + lambda_std_w * w_std
            + lambda_std_w_rest * w_rest_std
            + lambda_std_f * f_std
            + lambda_w_norm * w_norm
            + lambda_w_rest_norm * w_rest_norm
            + lambda_small_scale * max_s_sq
        )

        details = {
            "q_std": q_std.detach(),
            "s_std": s_std.detach(),
            "o_std": o_std.detach(),
            "cd_std": cd_std.detach(),
            "ch_std": ch_std.detach(),
            "w_std": w_std.detach(),
            "w_rest_std": w_rest_std.detach(),
            "f_std": f_std.detach(),
            "knn_dist_sq": dist_sq.detach(),
            "w_norm": w_norm.detach(),
            "w_rest_norm": w_rest_norm.detach(),
            "max_s_sq": max_s_sq.detach(),
        }
        return reg_loss, details, nn_ind

    def add_scalar(self, *args, **kwargs):
        if self.FAST_TRAINING:
            return
        if getattr(self, "NO_TB", False):
            return
        self.writer.add_scalar(*args, **kwargs)
        return

    def run(
        self,
        real_data_provider=None,
        test_every_n_step=-1,
        test_dataset=None,
        guidance=None,
    ):
        torch.cuda.empty_cache()

        init_beta = real_data_provider.betas
        total_t = real_data_provider.total_t
        (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        ) = self._get_model_optimizer(betas=init_beta, add_bones_total_t=total_t) # model 정의

        optimizer_pose, scheduler_pose = self._get_pose_optimizer(
            real_data_provider, model.add_bones
        )

        # * Optimization Loop
        stat_n_list = []
        active_sph_order, last_reset_step = 0, -1
        seed_everything(self.SEED)
        running_start_t = time.time()
        logging.info(f"Start training at {running_start_t}")
        loss_print = 0
        for step in tqdm(range(self.TOTAL_steps)):
            update_learning_rate(xyz_scheduler_func(step), "xyz", optimizer)
            update_learning_rate(sph_scheduler_func(step), "f_dc", optimizer)
            update_learning_rate(sph_rest_scheduler_func(step), "f_rest", optimizer)

            update_learning_rate(
                w_dc_scheduler_func(step), ["w_dc", "w_dc_vox"], optimizer
            )
            update_learning_rate(
                w_rest_scheduler_func(step), ["w_rest", "w_rest_vox"], optimizer
            )
            for k, v in scheduler_pose.items():
                update_learning_rate(v(step), k, optimizer_pose)

            if step in self.INCREASE_SPH_STEP:
                active_sph_order += 1
                logging.info(f"active_sph_order: {active_sph_order}")

            # * Recon fitting step
            model.train()
            optimizer.zero_grad(), optimizer_pose.zero_grad()

            loss = 0.0

            real_data_pack = real_data_provider(
                self.N_POSES_PER_STEP, continuous=False
            )
            (
                loss_recon, loss_mask, loss_normal, loss_normal_sc, loss_normal_both, loss_dwt, loss_dct, loss_fft, loss_depth, loss_style, loss_scale_flat, render_list, gt_rgb, gt_mask, gt_normal, gt_depth, model_ret, loss_dict,
            ) = self._fit_step(
                model,
                data_pack=real_data_pack,
                act_sph_ord=active_sph_order,
                random_bg=self.RAND_BG_FLAG,
                use_box_crop_pad=getattr(self, "BOX_CROP_PAD", -1),
                default_bg=getattr(self, "DEFAULT_BG", [0.0, 0.0, 0.0]),
            )
            loss = loss + loss_recon
            for k, v in loss_dict.items():
                self.add_scalar(k, v.detach(), step)

            loss = loss + self.LAMBDA_NORMAL * loss_normal
            loss = loss + self.LAMBDA_NORMAL_SC * loss_normal_sc
            loss = loss + self.LAMBDA_NORMAL_BOTH * loss_normal_both
            
            loss = loss + self.LAMBDA_DWT * loss_dwt
            loss = loss + self.LAMBDA_DCT * loss_dct
            loss = loss + self.LAMBDA_FFT * loss_fft 
            loss = loss + self.LAMBDA_SCALE_FLAT * loss_scale_flat
            loss = loss + self.LAMBDA_STYLE * loss_style
            
            loss = loss + self.LAMBDA_DEPTH * loss_depth
            
            if (last_reset_step < 0
                or step - last_reset_step > self.MASK_LOSS_PAUSE_AFTER_RESET
                and step > getattr(self, "MASK_START_STEP", 0)
            ):
                loss = loss + self.LAMBDA_MASK * loss_mask
                self.add_scalar("loss_mask", loss_mask.detach(), step)
            self.add_scalar("loss", loss.detach(), step)

            # * Reg Terms
            reg_loss, reg_details, knn_ind = self.compute_reg3D(model)
            loss = reg_loss + loss


            ###### normal smoothing loss ###########################################################
            normal, normal_sc = model_ret[-2].squeeze(0), model_ret[-1].squeeze(0)
            nn1 = normal[knn_ind[:,1].tolist(),:]
            nn2 = normal[knn_ind[:,2].tolist(),:]
            nn1_sc = normal_sc[knn_ind[:,1].tolist(),:]
            nn2_sc = normal_sc[knn_ind[:,2].tolist(),:]
            
            loss_normal_smoothing = 0
            loss_normal_smoothing = loss_normal_smoothing + 1 - F.cosine_similarity(normal, nn1, dim=1)
            loss_normal_smoothing = loss_normal_smoothing + 1 - F.cosine_similarity(normal, nn2, dim=1)
            loss_normal_smoothing = loss_normal_smoothing + 1 - F.cosine_similarity(normal_sc, nn1_sc, dim=1)
            loss_normal_smoothing = loss_normal_smoothing + 1 - F.cosine_similarity(normal_sc, nn2_sc, dim=1)
            loss_normal_smoothing = loss_normal_smoothing.mean()/4
            loss = loss + self.LAMBDA_NORMAL_SMOOTHING * loss_normal_smoothing
            
            
            

            for k, v in reg_details.items():
                self.add_scalar(k, v.detach(), step)

            loss.backward()
            optimizer.step()


            if step > getattr(self, "POSE_OPTIMIZE_START_STEP", -1):
                optimizer_pose.step()

            self.add_scalar("N", model.N, step)

            # * Gaussian Control
            if step > self.DENSIFY_START:
                for render_pkg in render_list:
                    model.record_xyz_grad_radii(
                        render_pkg["viewspace_points"],
                        render_pkg["radii"],
                        render_pkg["visibility_filter"],
                    )

            if (
                step > self.DENSIFY_START
                and step < getattr(self, "DENSIFY_END", 10000000)
                and step % self.DENSIFY_INTERVAL == 0
            ):
                N_old = model.N
                model.densify(
                    optimizer=optimizer,
                    max_grad=self.MAX_GRAD,
                    percent_dense=self.PERCENT_DENSE,
                    extent=0.5,
                    verbose=True,
                )
                logging.info(f"Densify: {N_old}->{model.N}")

            if step > self.PRUNE_START and step % self.PRUNE_INTERVAL == 0:
                N_old = model.N
                model.prune_points(
                    optimizer,
                    min_opacity=self.OPACIT_PRUNE_TH,
                    max_screen_size=1e10,  # ! disabled
                )
                logging.info(f"Prune: {N_old}->{model.N}")
            if step in getattr(self, "RANDOM_GROW_STEPS", []):
                model.random_grow(
                    optimizer,
                    num_factor=getattr(self, "NUM_FACTOR", 0.1),
                    std=getattr(self, "RANDOM_GROW_STD", 0.1),
                    init_opa_value=getattr(self, "RANDOM_GROW_OPA", 0.01),
                )

            if (step + 1) in self.RESET_OPACITY_STEPS:
                model.reset_opacity(optimizer, self.OPACIT_RESET_VALUE)
                last_reset_step = step

            if (step + 1) in getattr(self, "REGAUSSIAN_STEPS", []):
                model.regaussian(optimizer, self.REGAUSSIAN_STD)

            stat_n_list.append(model.N)
            if self.FAST_TRAINING:
                continue

            # * Viz
            if (step + 1) % getattr(self, "VIZ_INTERVAL", 100) == 0 or step == 0:
                mu, fr, s, o, sph = model_ret[:5]
                save_path = f"{self.log_dir}/viz_step/step_{step}.png"
                viz_render(
                    gt_rgb[0], gt_mask[0], render_list[0], save_path=save_path
                )
                # viz the spinning in the middle
                (
                    _,
                    _,
                    K,
                    pose_base,
                    pose_rest,
                    global_trans,
                    time_index,
                    style_feature,
                ) = real_data_pack
                viz_spinning(
                    model,
                    torch.cat([pose_base, pose_rest], 1)[:1],
                    global_trans[:1],
                    real_data_provider.H,
                    real_data_provider.W,
                    K[0],
                    save_path=f"{self.log_dir}/viz_step/spinning_{step}.gif",
                    time_index=time_index,
                    active_sph_order=active_sph_order,
                    # bg_color=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                    bg_color=[0.0, 0.0, 0.0],
                )

                can_pose = model.template.canonical_pose.detach()
                if self.mode == "human":
                    can_pose[0] = self.viz_base_R.to(can_pose.device)
                    can_pose = matrix_to_axis_angle(can_pose)[None]

                can_trans = torch.zeros(len(can_pose), 3).to(can_pose)
                can_trans[:, -1] = 3.0
                viz_H, viz_W = 512, 512
                viz_K = fov2K(60, viz_H, viz_W)
                viz_spinning(
                    model,
                    can_pose,
                    can_trans,
                    viz_H,
                    viz_W,
                    viz_K,
                    save_path=f"{self.log_dir}/viz_step/spinning_can_{step}.gif",
                    time_index=None,  # canonical pose use t=None
                    active_sph_order=active_sph_order,
                    # bg_color=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                    bg_color=[0.0, 0.0, 0.0],
                )

                # viz the distrbution
                plt.figure(figsize=(15, 3))
                scale = model.get_s
                for i in range(3):
                    _s = scale[:, i].detach().cpu().numpy()
                    plt.subplot(1, 3, i + 1)
                    plt.hist(_s, bins=100)
                    plt.title(f"scale {i}")
                    plt.grid(), plt.ylabel("count"), plt.xlabel("scale")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/s_hist_step_{step}.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                opacity = (model.get_o).squeeze(-1)
                plt.hist(opacity.detach().cpu().numpy(), bins=100)
                plt.title(f"opacity")
                plt.grid(), plt.ylabel("count"), plt.xlabel("opacity")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/o_hist_step_{step}.png")
                plt.close()

            # * test
            if test_every_n_step > 0 and (
                (step + 1) % test_every_n_step == 0 or step == self.TOTAL_steps - 1
            ):
                logging.info("Testing...")
                self._eval_save_error_map(model, test_dataset, f"test_{step}")
                test_results = self.eval_dir(f"test_{step}")
                for k, v in test_results.items():
                    self.add_scalar(f"test_{k}", v, step)

        running_end_t = time.time()
        logging.info(
            f"Training time: {(running_end_t - running_start_t):.3f} seconds i.e. {(running_end_t - running_start_t)/60.0 :.3f} minutes"
        )

        # * save
        logging.info("Saving model...")
        model.eval()
        ckpt_path = f"{self.log_dir}/model.pth"
        torch.save(model.state_dict(), ckpt_path)

        pose_path = f"{self.log_dir}/training_poses.pth"
        torch.save(real_data_provider.state_dict(), pose_path)
        model.to("cpu")

        # * stat
        plt.figure(figsize=(5, 5))
        plt.plot(stat_n_list)
        plt.title("N"), plt.xlabel("step"), plt.ylabel("N")
        plt.savefig(f"{self.log_dir}/N.png")
        plt.close()

        model.summary()
        return model, real_data_provider

    def testtime_pose_optimization(
        self,
        data_pack,
        model,
        evaluator,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        check_every_n_step=5,
        viz_fn=None,
        As=None,
        pose_As_lr=1e-3,
    ):
        # * Like Instant avatar, optimize the smpl pose f
        # * will optimize all poses in the data_pack
        torch.cuda.empty_cache()
        seed_everything(self.SEED)
        model.eval()  # to get gradients, but never optimized
        evaluator.eval()
        gt_rgb, gt_mask, gt_normal, gt_depth, K, pose_b, pose_r, trans, style_feature = data_pack[:9]
        pose_b = pose_b.detach().clone()
        pose_r = pose_r.detach().clone()
        trans = trans.detach().clone()
        pose_b.requires_grad_(True)
        pose_r.requires_grad_(True)
        trans.requires_grad_(True)
        gt_rgb, gt_mask, gt_normal, gt_depth, style_feature = gt_rgb.to(self.device), gt_mask.to(self.device), gt_normal.to(self.device), gt_depth.to(self.device), style_feature.to(self.device)

        optim_l = [
            {"params": [pose_b], "lr": pose_base_lr},
            {"params": [pose_r], "lr": pose_rest_lr},
            {"params": [trans], "lr": trans_lr},
        ]
        if As is not None:
            As = As.detach().clone()
            As.requires_grad_(True)
            optim_l.append({"params": [As], "lr": pose_As_lr})
        optimizer_smpl = torch.optim.SGD(optim_l)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_smpl, step_size=decay_steps, gamma=decay_factor
        )

        loss_list, psnr_list, ssim_list, lpips_list = [], [], [], []
        viz_step_list = []
        for inner_step in range(steps):
            # optimize
            optimizer_smpl.zero_grad()
            loss_recon, _, _, _, _, _, _, _, _, _, _,rendered_list, _, _, _, _, _, _ = self._fit_step(
                model,
                [gt_rgb, gt_mask, gt_normal, gt_depth, K, pose_b, pose_r, trans, None, style_feature],
                act_sph_ord=model.max_sph_order,
                random_bg=False,
                default_bg=getattr(self, "DEFAULT_BG", [0.0, 0.0, 0.0]),
                add_bones_As=As,
            )
            loss = loss_recon
            loss.backward()
            optimizer_smpl.step()
            scheduler.step()
            loss_list.append(float(loss))

            if (
                inner_step % check_every_n_step == 0
                or inner_step == steps - 1
                and viz_fn is not None
            ):
                viz_step_list.append(inner_step)
                with torch.no_grad():
                    _check_results = [
                        evaluator(
                            torch.unsqueeze(rendered_list[0]["rgb"], 0).permute(0,2,3,1),
                            gt_rgb,
                        )
                    ]
                psnr = torch.stack([r["psnr"] for r in _check_results]).mean().item()
                ssim = torch.stack([r["ssim"] for r in _check_results]).mean().item()
                lpips = torch.stack([r["lpips"] for r in _check_results]).mean().item()
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                lpips_list.append(float(lpips))

        if viz_fn is not None:
            os.makedirs(osp.dirname(viz_fn), exist_ok=True)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.plot(viz_step_list, psnr_list)
            plt.title(f"PSNR={psnr_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 2)
            plt.plot(viz_step_list, ssim_list)
            plt.title(f"SSIM={ssim_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 3)
            plt.plot(viz_step_list, lpips_list)
            plt.title(f"LPIPS={lpips_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 4)
            plt.plot(loss_list)
            plt.title(f"Loss={loss_list[-1]}"), plt.grid()
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(viz_fn)
            plt.close()

        if As is not None:
            As.detach().clone()
        return (
            pose_b.detach().clone(),
            pose_r.detach().clone(),
            trans.detach().clone(),
            As,
        )

    @torch.no_grad()
    def eval_fps(self, model, real_data_provider, rounds=1):
        model.eval()
        model.cache_for_fast()
        logging.info(f"Model has {model.N} points.")
        N_frames = len(real_data_provider.rgb_list)
        ret = real_data_provider(N_frames)
        gt_rgb, gt_mask, gt_normal, gt_depth, K, pose_base, pose_rest, global_trans, time_index, style_feature = ret
        pose = torch.cat([pose_base, pose_rest], 1)
        H, W = gt_rgb.shape[1:3]
        sph_o = model.max_sph_order
        logging.info(f"FPS eval using active_sph_order: {sph_o}")
        # run one iteration to check the output correctness

        mu, fr, sc, op, normal, normal_sc, sph, additional_ret = model(
            pose[0:1],
            global_trans[0:1],
            style_feature,
            additional_dict={},
            active_sph_order=sph_o,
            fast=True,
        )
        bg = [1.0, 1.0, 1.0]
        render_pkg = render_cam_pcl(
            mu[0], fr[0], sc[0], op[0], normal[0], normal_sc[0], sph[0], style_feature, H, W, K[0], False, sph_o, bg
        )
        pred = render_pkg["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        imageio.imsave(osp.join(self.log_dir, "fps_eval_sample.png"), pred)

        logging.info("Start FPS test...")
        start_t = time.time()

        for j in tqdm(range(int(N_frames * rounds))):
            i = j % N_frames
            mu, fr, sc, op, normal, normal_sc, sph, additional_ret = model(
                pose[i : i + 1],
                global_trans[i : i + 1],
                style_feature,
                additional_dict={"t": time_index},
                active_sph_order=sph_o,
                fast=True,
            )
            bg = [1.0, 1.0, 1.0]
            render_pkg = render_cam_pcl(
                mu[0], fr[0], sc[0], op[0], normal[0], normal_sc[0], sph[0], style_feature, H, W, K[0], False, sph_o, bg
            )
        end_t = time.time()

        fps = (rounds * N_frames) / (end_t - start_t)
        logging.info(f"FPS: {fps}")
        with open(osp.join(self.log_dir, "fps.txt"), "w") as f:
            f.write(f"FPS: {fps}")
        return fps


def tg_fitting_eval(solver, dataset_mode, seq_name, optimized_seq):
    if dataset_mode == "personal_data":
        test(
                solver,
                seq_name=seq_name,
                tto_flag=True,
                tto_step=160,
                tto_decay=50,
                dataset_mode=dataset_mode,
                pose_base_lr=3e-3,
                pose_rest_lr=3e-3,
                trans_lr=3e-3,
            )
    else:
        pass
    # solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)
    return

def seed_everything(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--profile", type=str, default="./profiles/people/people_1m.yaml")
    args.add_argument("--dataset_mode", type=str, default="personal_data")
    args.add_argument("--seq", type=str, default="jogging")
    args.add_argument("--logbase", type=str, default="debug")
    # remove the viz during optimization
    args.add_argument("--fast", action="store_true")
    args.add_argument("--no_eval", action="store_true")
    # for eval
    args.add_argument("--eval_only", action="store_true")
    args.add_argument("--log_dir", default="", type=str)
    args.add_argument("--skip_eval_if_exsists", action="store_true")
    # for viz
    args.add_argument("--viz_only", action="store_true")
    args = args.parse_args()
    

    device = torch.device("cuda:0")
    dataset_mode = args.dataset_mode
    seq_name = args.seq
    profile_fn = args.profile
    base_name = args.logbase
    seed_everything()
    # wandb.init(project=base_name, entity='jjang98')
    
    
    

    if len(args.log_dir) > 0:
        log_dir = args.log_dir
    else:
        os.makedirs(f"./logs/{base_name}", exist_ok=True)
        profile_name = osp.basename(profile_fn).split(".")[0]
        log_dir = (
            f"./logs/{base_name}/seq={seq_name}_prof={profile_name}_data={dataset_mode}"
        )
    if dataset_mode == "personal_data":
        mode = "human"
        smpl_path = "./data/smpl_model/SMPL_NEUTRAL.pkl"
    else:
        raise NotImplementedError()
    
    

    solver = TGFitter(
        log_dir=log_dir,
        profile_fn=profile_fn,
        mode=mode,
        template_model_path=smpl_path,
        device=device,
        FAST_TRAINING=args.fast,
    )
    
    
        
    data_provider, trainset = solver.prepare_real_seq(
        seq_name,
        dataset_mode,
        split='train',
        ins_avt_wild_start_end_skip=getattr(solver, "START_END_SKIP", None),
    )

    if args.eval_only:
        assert len(args.log_dir) > 0, "Please specify the log_dir for eval only mode"
        assert osp.exists(log_dir), f"{log_dir} not exists!"
        logging.info(f"Eval only mode, load model from {log_dir}")
        if args.skip_eval_if_exsists:
            test_dir = osp.join(log_dir, "test")
            if osp.exists(test_dir):
                logging.info(f"Eval already exists, skip")
                sys.exit(0)
        data_provider.load_state_dict(
            torch.load(osp.join(log_dir, "training_poses.pth")), strict=True
        )
        # solver.eval_fps(solver.load_saved_model(), data_provider, rounds=10)
        tg_fitting_eval(solver, dataset_mode, seq_name, data_provider)
        logging.info("Done")
        sys.exit(0)
    elif args.viz_only:
        assert len(args.log_dir) > 0, "Please specify the log_dir for viz only mode"
        assert osp.exists(log_dir), f"{log_dir} not exists!"
        data_provider.load_state_dict(
            torch.load(osp.join(log_dir, "training_poses.pth")), strict=True
        )
        logging.info(f"Viz only mode, load model from {log_dir}")
        if mode == "human":
            viz_human_all(solver, data_provider, viz_name="viz_only")
        logging.info("Done")
        sys.exit(0)

    logging.info(f"Optimization with {profile_fn}")
    _, optimized_seq = solver.run(data_provider)

    if mode == "human":
        viz_human_all(solver, optimized_seq)

    solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)
    if args.no_eval:
        logging.info("No eval, done!")
        sys.exit(0)

    tg_fitting_eval(solver, dataset_mode, seq_name, optimized_seq)

    logging.info("done!")
