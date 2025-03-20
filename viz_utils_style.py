from matplotlib import pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
import imageio
import torch
from tqdm import tqdm
import numpy as np
import warnings, os, os.path as osp, shutil, sys
from transforms3d.euler import euler2mat
from lib_data.data_provider import RealDataOptimizablePoseProviderPoseStyle
import torch.nn.functional as F

from lib_toonsplat.optim_utils import *
from lib_render.gauspl_renderer import render_cam_pcl
from lib_toonsplat.model_utils import transform_mu_frame

from utils.misc import *
from utils.viz import viz_render
import pdb


@torch.no_grad()
def viz_spinning(
    model,
    pose,
    trans,
    style_feature,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    mu, fr, s, o, normal, normal_sc, sph, additional_ret = model(
        pose, trans, style_feature, {"t": time_index}, active_sph_order=active_sph_order
    )
    if model_mask is not None:
        assert len(model_mask) == mu.shape[1]
        mu = mu[:, model_mask.bool()]
        fr = fr[:, model_mask.bool()]
        s = s[:, model_mask.bool()]
        o = o[:, model_mask.bool()]
        normal = normal[:, model_mask.bool()]
        normal_sc = normal_sc[:, model_mask.bool()]
        sph = sph[:, model_mask.bool()]

    viz_frames = []
    viz_frames_normal = []
    viz_frames_normal_sc = []
    for vid in range(n_spinning):
        spin_R = (
            torch.from_numpy(euler2mat(0, 2 * np.pi * vid / n_spinning, 0, "sxyz"))
            .to(device)
            .float()
        )
        spin_t = mu.mean(1)[0]
        spin_t = (torch.eye(3).to(device) - spin_R) @ spin_t[:, None]
        spin_T = torch.eye(4).to(device)
        spin_T[:3, :3] = spin_R
        spin_T[:3, 3] = spin_t.squeeze(-1)
        viz_mu, viz_fr = transform_mu_frame(mu, fr, spin_T[None])
        

        render_pkg = render_cam_pcl(
            viz_mu[0],
            viz_fr[0],
            s[0],
            o[0],
            normal[0],
            normal_sc[0], 
            sph[0],
            style_feature,
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
        
        ##################################################################
        viz_frame_normal = (
            torch.clamp((render_pkg["normal"] + 1 ) / 2, 0.0, 1.0)
            # torch.clamp(render_pkg["normal"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame_normal = (viz_frame_normal * 255).astype(np.uint8)
        viz_frames_normal.append(viz_frame_normal)
        ##################################################################
        
        # ##################################################################
        viz_frame_normal_sc = (
            torch.clamp((render_pkg["normal_sc"] + 1 ) / 2, 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame_normal_sc = (viz_frame_normal_sc * 255).astype(np.uint8)
        viz_frames_normal_sc.append(viz_frame_normal_sc)
        # ##################################################################
        
    # save_path_normal_ = save_path.split('/')[:-1]
    # save_path_normal_.append(save_path.split('/')[-1][:-4] + '_normal.gif')
    # save_path_normal = os.path.join(*save_path_normal_)
    
    
    # save_path_normal_sc_ = save_path.split('/')[:-1]
    # save_path_normal_sc_.append(save_path.split('/')[-1][:-4] + '_normal_sc.gif')
    # save_path_normal_sc = os.path.join(*save_path_normal_sc_)
    
    save_path_normal_ = save_path.split('/')[:-1]
    save_path_normal_.append(save_path.split('/')[-1][:-4] + '_normal')
    save_path_normal = os.path.join(*save_path_normal_)
    
    
    save_path_normal_sc_ = save_path.split('/')[:-1]
    save_path_normal_sc_.append(save_path.split('/')[-1][:-4] + '_normal_sc')
    save_path_normal_sc = os.path.join(*save_path_normal_sc_)
    
    # pdb.set_trace()
    # import cv2
    # for i in range(len(viz_frames_normal_from_sc)):
    #     cv2.imwrite('tmp'+str(i)+'.png', viz_frames_normal_from_sc[i])
    os.makedirs(save_path_normal, exist_ok=True)
    for i, img in enumerate(viz_frames_normal):
        imageio.imsave(f"{save_path_normal}/{i:04d}.png", img)
    
    os.makedirs(save_path_normal_sc, exist_ok=True)
    for i, img in enumerate(viz_frames_normal_sc):
        imageio.imsave(f"{save_path_normal_sc}/{i:04d}.png", img)
    
    # imageio.mimsave(save_path, viz_frames)
    # imageio.mimsave(save_path_normal, viz_frames_normal)
    # imageio.mimsave(save_path_normal_sc, viz_frames_normal_sc)
    return


@torch.no_grad()
def viz_spinning_self_rotate(
    model,
    base_R,
    pose,
    trans,
    style_feature,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    viz_frames = []
    # base_R = base_R.detach().cpu().numpy()
    first_R = axis_angle_to_matrix(pose[:, 0])[0].detach().cpu().numpy()
    for vid in range(n_spinning):
        rotation = euler2mat(0.0, 2 * np.pi * vid / n_spinning, 0.0, "sxyz")
        rotation = torch.from_numpy(first_R @ rotation).float().to(device)
        pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]

        mu, fr, s, o, mask, sph, additional_ret = model(
            pose, trans, style_feature, {"t": time_index}, active_sph_order=active_sph_order
        )
        if model_mask is not None:
            assert len(model_mask) == mu.shape[1]
            mu = mu[:, model_mask.bool()]
            fr = fr[:, model_mask.bool()]
            s = s[:, model_mask.bool()]
            o = o[:, model_mask.bool()]
            mask = mask[:, model_mask.bool()]
            sph = sph[:, model_mask.bool()]

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            mask[0],
            sph[0],
            style_feature,
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


@torch.no_grad()
def viz_human_all(
    solver,
    data_provider: RealDataOptimizablePoseProviderPoseStyle = None,
    ckpt_dir=None,
    training_skip=1,
    n_spinning=40,
    novel_pose_dir="novel_poses",
    novel_skip=2,
    model=None,
    model_mask=None,
    viz_name="",
    export_mesh_flag=False,  # remove this from release version
):
    if model is None:
        model = solver.load_saved_model(ckpt_dir)
    model.eval()

    viz_dir = osp.join(solver.log_dir, f"{viz_name}_human_viz")
    os.makedirs(viz_dir, exist_ok=True)

    active_sph_order = int(model.max_sph_order)
    if data_provider is not None:
        # if ckpt_dir is None:
        #     ckpt_dir = solver.log_dir
        # pose_path = osp.join(ckpt_dir, "pose.pth")
        pose_base_list = data_provider.pose_base_list
        pose_rest_list = data_provider.pose_rest_list
        global_trans_list = data_provider.global_trans_list
        pose_list = torch.cat([pose_base_list, pose_rest_list], 1)
        pose_list, global_trans_list = pose_list.to(
            solver.device
        ), global_trans_list.to(solver.device)
        rgb_list = data_provider.rgb_list
        mask_list = data_provider.mask_list
        normal_list = data_provider.normal_list
        K_list = data_provider.K_list
        style_feature = data_provider.style_feature
        H, W = rgb_list.shape[1:3]
    else:
        H, W = 512, 512
        K_list = [torch.from_numpy(fov2K(45, H, W)).float().to(solver.device)]
        global_trans_list = torch.zeros(1, 3).to(solver.device)
        global_trans_list[0, -1] = 3.0

    # viz training
    if data_provider is not None:
        print("Viz training...")
        viz_frames = []
        for t in range(len(pose_list)):
            if t % training_skip != 0:
                continue
            pose = pose_list[t][None]
            K = K_list[t]
            trans = global_trans_list[t][None]
            time_index = torch.Tensor([t]).long().to(solver.device)
            mu, fr, s, o, normal, normal_sc, sph, _ = model(
                pose,
                trans,
                style_feature,
                {"t": time_index},  # use time_index from training set
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                normal = normal[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                normal[0],
                normal_sc[0], 
                sph[0],
                style_feature,
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [0.0, 0.0, 0.0]),
            )
            viz_frame = viz_render(rgb_list[t], mask_list[t], render_pkg)
            viz_frames.append(viz_frame)
        imageio.mimsave(f"{viz_dir}/training.gif", viz_frames)

    # viz static spinning
    # print("Viz spinning...")
    # can_pose = model.template.canonical_pose.detach()
    # viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
    # viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
    # can_pose[0] = viz_base_R_opencv.to(can_pose.device)
    # can_pose = matrix_to_axis_angle(can_pose)[None]
    # dapose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    # dapose[:, 1, -1] = np.pi / 4
    # dapose[:, 2, -1] = -np.pi / 4
    # dapose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    # tpose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    # tpose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    # to_viz = {"cano-pose": can_pose, "t-pose": tpose, "da-pose": dapose}
    # vis_index = 0
    # if data_provider is not None:
    #     to_viz["first-frame"] = pose_list[vis_index][None]
    # pdb.set_trace()

    # for name, pose in to_viz.items():
    #     print(f"Viz novel {name}...")
        # if export_mesh_flag:
        #     from lib_marchingcubes.gaumesh_utils import MeshExtractor
        #     # also extract a mesh
        #     mesh = solver.extract_mesh(model, pose)
        #     mesh.export(f"{viz_dir}/mc_{name}.obj", "obj")

        # # for making figures, the rotation is in another way
        # viz_spinning_self_rotate(
        #     model,
        #     solver.viz_base_R.detach(),
        #     pose,
        #     global_trans_list[0][None],
        #     H,
        #     W,
        #     K_list[0],
        #     f"{viz_dir}/{name}_selfrotate.gif",
        #     time_index=None,  # if set to None and use t, the add_bone will hand this
        #     n_spinning=n_spinning,
        #     active_sph_order=model.max_sph_order,
        # )
        # viz_spinning(
        #     model,
        #     pose,
        #     global_trans_list[vis_index][None],
        #     style_feature,
        #     H,
        #     W,
        #     K_list[vis_index],
        #     f"{viz_dir}/{name}.gif",
        #     time_index=None,  # if set to None and use t, the add_bone will hand this
        #     n_spinning=n_spinning,
        #     active_sph_order=model.max_sph_order,
        #     bg_color=getattr(solver, "DEFAULT_BG", [0.0, 0.0, 0.0]),
        # )

    # viz novel pose dynamic spinning
    
    print("Viz novel seq...")
    novel_pose_names = [
        f[:-4] for f in os.listdir(novel_pose_dir) if f.endswith(".npy")
    ]
    seq_viz_todo = {}
    for name in novel_pose_names:
        novel_pose_fn = osp.join(novel_pose_dir, f"{name}.npy")
        novel_poses = np.load(novel_pose_fn, allow_pickle=True)
        novel_poses = novel_poses[::novel_skip]
        N_frames = len(novel_poses)
        novel_poses = torch.from_numpy(novel_poses).float().to(solver.device)
        novel_poses = novel_poses.reshape(N_frames, 24, 3)

        seq_viz_todo[name] = (novel_poses, N_frames)
    if data_provider is not None:
        seq_viz_todo["training"] = [pose_list, len(pose_list)]

    for name, (novel_poses, N_frames) in seq_viz_todo.items():
        base_R = solver.viz_base_R.detach().cpu().numpy()
        viz_frames = []
        viz_frames_normal = []
        K = K_list[0]
        for vid in range(N_frames):
            pose = novel_poses[vid][None]
            pose = novel_poses[0][None] # debug
            rotation = euler2mat(2 * np.pi * vid / N_frames, 0.0, 0.0, "syxz")
            rotation = torch.from_numpy(rotation @ base_R).float().to(solver.device)
            pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]
            trans = global_trans_list[0][None]
            mu, fr, s, o, normal, normal_sc, sph, _ = model(
                pose,
                trans,
                style_feature,
                # not pass in {}, so t is auto none
                additional_dict={},
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                normal = normal[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                normal[0],
                normal_sc[0], 
                sph[0],
                style_feature,
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [0.0, 0.0, 0.0]),
                # bg_color=[1.0, 1.0, 1.0],  # ! use white bg for viz
            )
            viz_frame = (
                torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            viz_frame = (viz_frame * 255).astype(np.uint8)
            viz_frames.append(viz_frame)
        os.makedirs(f"{viz_dir}/{name}_one_frame", exist_ok=True)
        for i, img in enumerate(viz_frames):
            imageio.imsave(f"{viz_dir}/{name}_one_frame/novel_pose_{name}_{i:04d}.png", img)
            ################################################################
        #     viz_frame_normal = (
        #         torch.clamp((render_pkg["normal"] + 1 ) / 2, 0.0, 1.0)
        #         # torch.clamp(render_pkg["normal"], 0.0, 1.0)
        #         .permute(1, 2, 0)
        #         .detach()
        #         .cpu()
        #         .numpy()
        #     )
        #     viz_frame_normal = (viz_frame_normal * 255).astype(np.uint8)
        #     viz_frames_normal.append(viz_frame_normal)
            ##################################################################
        # os.makedirs(f"{viz_dir}/{name}_normal", exist_ok=True)
        # for i, img in enumerate(viz_frames_normal):
        #     imageio.imsave(f"{viz_dir}/{name}_normal/{i:04d}.png", img)
        # imageio.mimsave(f"{viz_dir}/novel_pose_{name}.gif", viz_frames)
    return

