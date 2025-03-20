from torch.utils.data import Dataset
import logging
import json
import os
import numpy as np
from os.path import join
import os.path as osp
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
import imageio
import cv2
from plyfile import PlyData
from tqdm import tqdm
from transforms3d.euler import euler2mat
import glob
import pdb
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import torch.nn.functional as F


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        # data_root = "./data/neuman_raw/",               ## data 경로 수정 ./data/InstantAvatar_preprocessed_data_all/
        data_root = "./data/ToonVid/",
        split="train",
        image_zoom_ratio=1.0,
        start_end_skip=None,
        video_name = ""
    ) -> None:
        super().__init__()
        self.data_root = data_root + video_name
        self.video_name = video_name

        root = data_root + video_name

        image_path = os.path.join(root, "images")
        test_idx = self.get_test_idx(video_name)

        idx = [i for i in range(len(os.listdir(image_path))) if i not in test_idx]
        
        

        if start_end_skip is not None:
            start, end, skip = start_end_skip
        else:
            # raise NotImplementedError("Must specify, check the end+1")
            if split == "train": 
                idx = idx
            elif split == 'test': 
                idx = test_idx
            
            self.img_lists = [sorted(glob.glob(f"{root}/images/*.png"))[i] for i in idx]
            self.msk_lists = [sorted(glob.glob(f"{root}/masks/*.png"))[i] for i in idx]
            
            self.normal_lists = [sorted(glob.glob(f"{root}/normal/*.png"))[i] for i in idx]
            self.depth_lists = [sorted(glob.glob(f"{root}/depths_images/*.png"))[i] for i in idx]
                

        self.image_zoom_ratio = image_zoom_ratio

        # camera = np.load(osp.join(root, "cameras.npz"))
        camera = np.load(osp.join(root, "cameras_ScoreHMR.npz"))
        K = camera["intrinsic"]
        T_wc = np.linalg.inv(camera["extrinsic"])
        assert np.allclose(T_wc, np.eye(4))

        height = camera["height"]
        width = camera["width"]

        self.downscale = 1.0 / self.image_zoom_ratio
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale
        self.K = K


        # pose_fn = osp.join(root, "poses_optimized.npz")
        pose_fn = osp.join(root, "poses_ScoreHMR.npz")
        smpl_params = load_smpl_param(pose_fn)
        smpl_params["body_pose"] = [smpl_params["body_pose"][i] for i in idx]
        smpl_params["global_orient"] = [smpl_params["global_orient"][i] for i in idx]
        smpl_params["transl"] = [smpl_params["transl"][i] for i in idx]
        self.smpl_params = smpl_params

        self.img_buffer, self.msk_buffer, self.normal_buffer, img_style_feature_buffer, self.depth_buffer = [], [], [], [], []
        for idx in tqdm(range(len(self.img_lists))):
            img = cv2.imread(self.img_lists[idx])[..., ::-1]
            # msk = np.load(self.msk_lists[idx])
            msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE)
            
            if self.downscale > 1:
                img = cv2.resize(
                    img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )
                msk = cv2.resize(
                    msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )

            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32) / 255.0
            
            normal = cv2.imread(self.normal_lists[idx])[..., ::-1]
            normal = (normal[..., :3] / 255).astype(np.float32)
            normal = normal * 2 - 1
            
            
            depth = cv2.imread(self.depth_lists[idx], cv2.IMREAD_UNCHANGED)
            # pdb.set_trace()
            depth = (depth / 255).astype(np.float32)

            bg_color = np.zeros_like(img).astype(np.float32)
            img = img * msk[..., None] #+ (1 - msk[..., None])
            self.img_buffer.append(img)
            self.msk_buffer.append(msk)
            self.normal_buffer.append(normal)
            self.depth_buffer.append(depth)
            img_style_feature_buffer.append(torch.tensor(np.transpose(img, (2, 1, 0)))) # masked image
            # pdb.set_trace()
            
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        for param in model.parameters():
            param.requires_grad = False
        img_style_feature_buffer = torch.stack(img_style_feature_buffer)
        inputs = F.interpolate(img_style_feature_buffer, size=(224, 224), mode='bilinear', align_corners=False)
        # inputs = processor(images=img_style_feature_buffer, return_tensors="pt", do_rescale=False)
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state
        self.feature_buffer = [last_hidden_states[i] for i in range(last_hidden_states.size(0))]
        # self.feature_buffer = last_hidden_states
        
        return

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = self.img_buffer[idx]
        msk = self.msk_buffer[idx]
        normal = self.normal_buffer[idx]
        depth = self.depth_buffer[idx]
        style_feature = self.feature_buffer[idx]

        pose = self.smpl_params["body_pose"][idx].reshape((23, 3))
        pose = np.concatenate([self.smpl_params["global_orient"][idx][None], pose], 0)

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk,
            "normal": normal,
            "depth" : depth,
            "K": self.K.copy(),
            "smpl_beta": self.smpl_params["betas"][0],  # ! use the first beta!
            "smpl_pose": pose,
            "smpl_trans": self.smpl_params["transl"][idx],
            "idx": idx,
            "style_feature": style_feature,
        }

        meta_info = {
            "video": self.video_name,
        }
        viz_id = f"video{self.video_name}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info

    def get_test_idx(self, video_name):
        if video_name == 'zelda' or video_name == 'AladdinDance' or video_name == '3d_girl_walk' or video_name == 'seattle'or video_name == 'parkinglot' or video_name =='3DPW_downtown_walkdownhill_00_GoEnhance_pixar_style':
            return [2,7,12,17]
        elif video_name == 'haku' or video_name == '3DPW_outdoors_slalom_00_illustration_v1': return [2,7,12,17,22]
        elif video_name == 'citron': return [2,7,12]
        elif video_name == '3DPW_courtyard_jumpBench_01_illustration_v13_1' or video_name == '3DPW_outdoors_fencing_01_GoEnhance_Illustration' or video_name == 'GOT10K_val39_GoEnhance_pixel_style' or video_name == 'GOT10K_val104_Domo_Anime_v1': return [2,7,12,17,22,27,32]
        elif video_name == '3DPW_outdoors_climbing_00_GoEnhance_JapaneseAnimeStyle': return [2,7,12,17,22,27,32,37]
        elif video_name == 'gta_human': return [2,7,12,17,22,27,32,37,42]
        elif video_name == 'lab' or video_name == 'bike' or video_name == 'jogging': return [2,7,12,17,22,27,32,37,42,47]
        else: 
            print('type right video name')
            return None
        


if __name__ == "__main__":
    dataset = Dataset(
    )
    ret = dataset[0]