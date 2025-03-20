import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.abspath(__file__)))

import torch
import numpy as np
from eval_utils_instant_avatar import Evaluator as EvalAvatar


from typing import Union
from lib_render.gauspl_renderer import render_cam_pcl
import cv2, glob
import pandas as pd
from tqdm import tqdm
import imageio
from lib_data.personal_data_style import Dataset as PersonalDatasetStyle
from lib_gart.model_utils import transform_mu_frame
from matplotlib import pyplot as plt
from transforms3d.euler import euler2mat
import logging
import pdb


def get_evaluator(mode, device):
    if mode == "avatar":
        evaluator = EvalAvatar()
    else:
        raise NotImplementedError()
    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


class TrainingSeqWrapper:
    def __init__(self, seq) -> None:
        self.seq = seq

    def __len__(self):
        return self.seq.total_t

    def __getitem__(self, idx):
        data = {}
        data["rgb"] = self.seq.rgb_list[idx]
        data["mask"] = self.seq.mask_list[idx]
        data["normal"] = self.seq.normal_list[idx]
        data["K"] = self.seq.K_list[idx]
        data["style_feature"] = self.seq.style_feature[idx]
        data["smpl_pose"] = torch.cat(
            [self.seq.pose_base_list[idx], self.seq.pose_rest_list[idx]], dim=0
        )
        data["smpl_trans"] = self.seq.global_trans_list[idx]
        return data, {}


def test(
    solver,
    seq_name: str,
    tto_flag=True,
    tto_step=300,
    tto_decay=60,
    tto_decay_factor=0.5,
    pose_base_lr=3e-3,
    pose_rest_lr=3e-3,
    trans_lr=3e-3,
    dataset_mode="personal_data",
    training_optimized_seq=None,
):
    device = solver.device
    model = solver.load_saved_model()

    assert dataset_mode in ["personal_data"], f"Unknown dataset mode {dataset_mode}"


    if dataset_mode == "personal_data":
        eval_mode = "avatar"
        bg = [0.0, 0.0, 0.0]
        test_dataset = PersonalDatasetStyle(
            video_name=seq_name,
            split="test",
        )            
    else:
        raise NotImplementedError()

    evaluator = get_evaluator(eval_mode, device)

    _save_eval_maps(
        solver.log_dir,
        "test",
        model,
        solver,
        test_dataset,
        dataset_mode=dataset_mode,
        device=device,
        bg=bg,
        tto_flag=tto_flag,
        tto_step=tto_step,
        tto_decay=tto_decay,
        tto_decay_factor=tto_decay_factor,
        tto_evaluator=evaluator,
        pose_base_lr=pose_base_lr,
        pose_rest_lr=pose_rest_lr,
        trans_lr=trans_lr,
    )

    if tto_flag:
        _evaluate_dir(evaluator, solver.log_dir, "test_tto")
    else:
        _evaluate_dir(evaluator, solver.log_dir, "test")

    return


def _save_eval_maps(
    log_dir,
    save_name,
    model,
    solver,
    test_dataset,
    dataset_mode="people_snapshot",
    bg=[1.0, 1.0, 1.0],
    # tto
    tto_flag=False,
    tto_step=300,
    tto_decay=60,
    tto_decay_factor=0.5,
    tto_evaluator=None,
    pose_base_lr=3e-3,
    pose_rest_lr=3e-3,
    trans_lr=3e-3,
    device=torch.device("cuda:0"),
):
    model.eval()
    
    if tto_flag:
        test_save_dir_tto = osp.join(log_dir, f"{save_name}_tto")
        test_save_dir_tto_gt = osp.join(log_dir, f"{save_name}_tto_pseudo_gt")
        test_save_dir_tto_normal = osp.join(log_dir, f"{save_name}_tto_normal")
        test_save_dir_tto_normal_sc = osp.join(log_dir, f"{save_name}_tto_normal_sc")
        test_save_dir_tto_depth = osp.join(log_dir, f"{save_name}_tto_depth")
        os.makedirs(test_save_dir_tto, exist_ok=True)
        os.makedirs(test_save_dir_tto_gt, exist_ok=True)
        os.makedirs(test_save_dir_tto_normal, exist_ok=True)
        os.makedirs(test_save_dir_tto_normal_sc, exist_ok=True)
        os.makedirs(test_save_dir_tto_depth, exist_ok=True)
        
    else:
        test_save_dir = osp.join(log_dir, save_name)
        test_save_dir_gt = osp.join(log_dir, f"{save_name}_gt")
        os.makedirs(test_save_dir, exist_ok=True)
        os.makedirs(test_save_dir_gt, exist_ok=True)

    iter_test_dataset = test_dataset

    logging.info(
        f"Saving images [TTO={tto_flag}] [N={len(iter_test_dataset)}]..."
    )
    for batch_idx, batch in tqdm(enumerate(iter_test_dataset)):
        # get data
        data, meta = batch


        rgb_gt = torch.as_tensor(data["rgb"])[None].float().to(device)
        mask_gt = torch.as_tensor(data["mask"])[None].float().to(device)
        normal_gt = torch.as_tensor(data["normal"])[None].float().to(device)
        depth_gt = torch.as_tensor(data["depth"])[None].float().to(device)
        H, W = rgb_gt.shape[1:3]
        K = torch.as_tensor(data["K"]).float().to(device)
        pose = torch.as_tensor(data["smpl_pose"]).float().to(device)[None]
        trans = torch.as_tensor(data["smpl_trans"]).float().to(device)[None]
        style_feature = torch.as_tensor(data["style_feature"]).to(device)
            

        fn = f"{batch_idx}.png"
        if tto_flag:
            # change the pose from the dataset to fit the test view
            pose_b, pose_r = pose[:, :1], pose[:, 1:]
            model.eval()
            # * for delta list
            try:
                list_flag = model.add_bones.mode in ["delta-list"]
            except:
                list_flag = False
            if list_flag:
                As = model.add_bones(t=batch_idx)  # B,K,4,4, the nearest pose
            else:
                As = None  # place holder
            new_pose_b, new_pose_r, new_trans, As = solver.testtime_pose_optimization(
                data_pack=[
                    rgb_gt,
                    mask_gt,
                    normal_gt,
                    depth_gt,
                    K[None],
                    pose_b,
                    pose_r,
                    trans,
                    style_feature,
                    None,
                ],
                model=model,
                evaluator=tto_evaluator,
                pose_base_lr=pose_base_lr,
                pose_rest_lr=pose_rest_lr,
                trans_lr=trans_lr,
                steps=tto_step,
                decay_steps=tto_decay,
                decay_factor=tto_decay_factor,
                As=As,
            )
            pose = torch.cat([new_pose_b, new_pose_r], dim=1).detach()
            trans = new_trans.detach()

            save_fn = osp.join(test_save_dir_tto, fn)
            save_gt_fn = osp.join(test_save_dir_tto_gt, fn)
            save_normal_fn = osp.join(test_save_dir_tto_normal, fn)
            save_normal_sc_fn = osp.join(test_save_dir_tto_normal_sc, fn)
            save_depth_fn = osp.join(test_save_dir_tto_depth, fn)
            _save_render_image_from_pose(
                model,
                pose,
                trans,
                style_feature,
                H,
                W,
                K,
                bg,
                rgb_gt,
                save_fn,
                save_gt_fn,
                save_normal_fn,
                save_normal_sc_fn,
                save_depth_fn,
                time_index=batch_idx,
                As=As,
            )
        else:
            save_fn = osp.join(test_save_dir, fn)
            _save_render_image_from_pose(
                model, pose, trans, style_feature, H, W, K, bg, rgb_gt, save_fn, save_gt_fn, time_index=batch_idx
            )
    return

def _compute_rotation_translation(eye, at, up):
    # Calculate the direction vector from eye to at
    direction = np.array(at) - np.array(eye)
    direction /= np.linalg.norm(direction)

    # Calculate the right vector by taking the cross product of direction and up
    right = np.cross(direction, np.array(up))
    right /= np.linalg.norm(right)

    # Calculate the up vector by taking the cross product of right and direction
    new_up = np.cross(right, direction)

    # Construct the rotation matrix
    R = np.array([right, new_up, -direction])  # Note the negation for the direction

    # Construct the translation vector
    T = np.array(eye)

    return R, T

@torch.no_grad()
def _save_render_image_from_pose(
    model, pose, trans, style_feature, H, W, K, bg, rgb_gt, save_fn, save_gt_fn, save_normal_fn, save_normal_sc_fn, save_depth_fn, time_index=None, As=None
):
    act_sph_order = model.max_sph_order
    device = pose.device
    additional_dict = {"t": time_index}
    if As is not None:
        additional_dict["As"] = As
    mu, fr, sc, op, normal, normal_sc, sph, _ = model(
        pose, trans, style_feature, additional_dict=additional_dict, active_sph_order=act_sph_order
    )  
    
    
    

    
    render_pkg = render_cam_pcl(
        mu[0], fr[0], sc[0], op[0], normal[0], normal_sc[0], sph[0], style_feature, H, W, K, False, act_sph_order, bg
    )
    mask = (render_pkg["alpha"].squeeze(0) > 0.0).bool()
    render_pkg["rgb"][:, ~mask] = bg[0]  # either 0.0 or 1.0
    pred_rgb = render_pkg["rgb"]  # 3,H,W
    pred_rgb = pred_rgb.permute(1, 2, 0)[None]  # 1,H,W,3
    pred_normal = render_pkg["normal"]
    pred_normal = pred_normal.permute(1,2,0)[None]
    pred_normal_sc = render_pkg["normal_sc"]
    pred_normal_sc = pred_normal_sc.permute(1,2,0)[None]
    pred_depth = render_pkg["dep"].unsqueeze(-1)
    


    errmap = (pred_rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
    errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    errmap = torch.from_numpy(errmap).to(device)[None] / 255
    pred_img = pred_rgb[..., [2,1,0]]
    pred_normal = pred_normal[..., [2,1,0]]
    pred_normal_sc = pred_normal_sc[..., [2,1,0]]
    gt_img = rgb_gt[..., [2, 1, 0]]
    img = torch.cat(
        [rgb_gt[..., [2, 1, 0]], pred_rgb[..., [2, 1, 0]], errmap], dim=2
    )  # ! note, here already swapped the channel order
    # cv2.imwrite(save_total_fn, img.cpu().numpy()[0] * 255)
    cv2.imwrite(save_fn, pred_img.cpu().numpy()[0] * 255)
    cv2.imwrite(save_gt_fn, gt_img.cpu().numpy()[0] * 255)
    cv2.imwrite(save_normal_fn, ((pred_normal.cpu().numpy()[0] + 1) / 2) * 255)
    cv2.imwrite(save_normal_sc_fn, ((pred_normal_sc.cpu().numpy()[0] + 1) / 2) * 255)
    cv2.imwrite(save_depth_fn, pred_depth.cpu().numpy()[0] * 255)
    # plt.savefig(save_depth_fn)
    return


@torch.no_grad()
def _evaluate_dir(evaluator, log_dir, dir_name="test", device=torch.device("cuda:0")):
    imgs = [cv2.imread(fn) for fn in glob.glob(f"{osp.join(log_dir, dir_name)}/*.png")]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [torch.tensor(img).float() / 255.0 for img in imgs]
    
    gt_dir_name = dir_name + '_pseudo_gt'
    imgs_gt = [cv2.imread(fn) for fn in glob.glob(f"{osp.join(log_dir, gt_dir_name)}/*.png")]
    imgs_gt = [cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB) for img_gt in imgs_gt]
    imgs_gt = [torch.tensor(img_gt).float() / 255.0 for img_gt in imgs_gt]

    evaluator = evaluator.to(device)
    evaluator.eval()

    if isinstance(evaluator, EvalAvatar):
        eval_mode = "instant-avatar"
    else:
        eval_mode = "unknown"

    H, W = imgs[0].shape[:2]
    logging.info(f"Image size: {H}x{W}")
    W //= 3
    # evaluator(pred, gt)
    with torch.no_grad():  
        results = [
            evaluator(
                torch.unsqueeze(img,0).to(device),
                torch.unsqueeze(img_gt,0).to(device),
            )
            for img, img_gt in zip(imgs, imgs_gt)
        ]

    ret = {}
    logging.info(f"Eval with {eval_mode} Evaluator from their original code release")
    with open(
        osp.join(log_dir, f"results_{dir_name}_evaluator_{eval_mode}.txt"), "w"
    ) as f:
        psnr = torch.stack([r["psnr"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] PSNR: {psnr:.2f}")
        f.write(f"[{dir_name}] PSNR: {psnr:.2f}\n")
        ret["psnr"] = psnr

        ssim = torch.stack([r["ssim"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] SSIM: {ssim:.4f}")
        f.write(f"[{dir_name}] SSIM: {ssim:.4f}\n")
        ret["ssim"] = ssim

        lpips = torch.stack([r["lpips"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] LPIPS: {lpips:.4f}")
        f.write(f"[{dir_name}] LPIPS: {lpips:.4f}\n")
        ret["lpips"] = lpips
    # save a xls of the per frame results
    for i in range(len(results)):
        for k in results[i].keys():
            results[i][k] = float(results[i][k].cpu())
    df = pd.DataFrame(results)
    df.to_excel(osp.join(log_dir, f"results_{dir_name}_evaluator_{eval_mode}.xlsx"))
    metrics = {
        "psnr": [r["psnr"] for r in results],
        "ssim": [r["ssim"] for r in results],
        "lpips": [r["lpips"] for r in results],
    }
    np.save(osp.join(log_dir, f"{dir_name}_{eval_mode}.npy"), metrics)
    return ret


# if __name__ == "__main__":
#     # debug for brightness eval