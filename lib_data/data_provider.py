import torch, numpy as np
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
from transforms3d.axangles import axangle2mat, mat2axangle
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from sklearn.neighbors import KernelDensity
from torch import nn
from matplotlib import pyplot as plt
import os
import pdb

class DatabasePoseProvider(nn.Module):
    def __init__(
        self,
        pose_dirs: list,
        da_pose_prob=0.1,
        da_range=[0.0, np.pi / 4],
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.device = device
        self.base_R = matrix_to_axis_angle(
            torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[None]
        )[0]
        self.base_R = self.base_R.float().to(self.device)

        self.da_pose_prob = da_pose_prob
        self.da_range = da_range

        self.data = []

        # cache the poses
        for d in pose_dirs:
            print(f"Caching {d} ...")
            for subject in tqdm(os.listdir(d)):
                sub_dir = os.path.join(d, subject)
                if not os.path.isdir(sub_dir):
                    continue
                npz_files = [f for f in os.listdir(sub_dir) if f.endswith(".npz")]
                npz_files.sort()
                for fn in npz_files:
                    try:
                        npz_fn = os.path.join(sub_dir, fn)
                        pose_data = np.load(npz_fn)
                        amass_len = pose_data["poses"].shape[0]
                        smplx_to_smpl = list(range(66)) + [72, 73, 74, 117, 118, 119]
                        poses = pose_data["poses"][:, smplx_to_smpl].reshape(
                            amass_len, 24, 3
                        )
                        self.data.append(poses.astype(np.float16))
                    except:
                        # print(f"Error in {npz_fn}, skip!")
                        pass
        self.data = np.concatenate(self.data, axis=0)
        print(
            f"Database has poses {len(self.data)} with DA-pose prob {self.da_pose_prob} and range {self.da_range}"
        )
        return

    def forward(self, N: int):
        pose, trans = self.sample_pose(N)
        return pose, trans

    def sample_pose(self, N: int):
        # da pose
        pose_list = []
        for i in range(N):
            seed = np.random.rand()
            if seed > self.da_pose_prob:
                # from database
                idx = np.random.randint(len(self.data))
                pose = torch.from_numpy(self.data[idx]).float().to(self.device)
            else:
                # da pose
                pose = torch.zeros(24, 3).to(self.device)
                da_theta = float(np.random.uniform(*self.da_range))
                pose[1, -1] = da_theta
                pose[2, -1] = -da_theta
            pose[0] = self.base_R
            pose_list.append(pose)
        pose = torch.stack(pose_list, dim=0)
        trans = torch.zeros(N, 3).to(self.device)
        return pose, trans


class TPoseProvider:
    # ! naive one
    def __init__(self, device) -> None:
        self.device = device
        self.base_R = matrix_to_axis_angle(
            torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[None]
        )[0]
        self.base_R = self.base_R.float().to(self.device)
        return

    def sample_pose(self, N: int):
        # da pose
        pose = torch.zeros(N, 24, 3).to(self.device)
        pose[:, 0] = self.base_R[None]
        trans = torch.zeros(N, 3).to(self.device)
        return pose, trans




class RealDataOptimizablePoseProviderPoseStyle(nn.Module):
    # separate the base_R and rest_R
    def __init__(self, dataset, balance=True):
        super().__init__()
        self.dataset = dataset
        self.balance = balance
        (
            rgb_list,
            mask_list,
            normal_list,
            depth_list,
            K_list,
            pose_list,
            global_trans_list,
            betas,
            style_feature,
        ) = self.prepare_for_fitting(dataset)
        for name, tensor in zip(
            ["rgb_list", "mask_list", "normal_list", "depth_list", "K_list", "betas", "style_feature"],
            [rgb_list, mask_list, normal_list, depth_list, K_list, betas, style_feature],
        ):
            # self.register_buffer(name, tensor)
            # * don't register buffer, just save them in RAM
            setattr(self, name, tensor)

        self.pose_base_list = nn.Parameter(pose_list[:, :1])
        self.pose_rest_list = nn.Parameter(pose_list[:, 1:])
        self.global_trans_list = nn.Parameter(global_trans_list)
        self.register_buffer("pose_list_original", pose_list.clone())
        self.register_buffer("global_trans_list_original", global_trans_list.clone())


        return

    def move_images_to_device(self, device):
        self.rgb_list = self.rgb_list.to(device)
        self.mask_list = self.mask_list.to(device)
        self.normal_list = self.normal_list.to(device)
        self.depth_list = self.depth_list.to(device)
        self.K_list = self.K_list.to(device)
        self.betas = self.betas.to(device)

    @property
    def total_t(self):
        return len(self.rgb_list)



    @property
    def pose_diff(self):
        optim_pose = self.pose_list
        ori_pose = self.pose_list_original
        pose_diff = optim_pose - ori_pose
        optim_trans = self.global_trans_list
        ori_trans = self.global_trans_list_original
        trans_diff = optim_trans - ori_trans
        return pose_diff, trans_diff

    def forward(
        self,
        N=None,
        continuous=False,
        index=None,
        return_index=False,
        force_uniform=False,
    ):
        device = self.pose_base_list.device
        # for uniform is for pose optimization
        # TODO: support multi-clips
        if index is not None:
            if not isinstance(index, torch.Tensor):
                index = torch.from_numpy(np.asarray(index)).long()
                if index.dim() == 0:
                    index = index[None]
            t = index
        else:
            assert N is not None
            if N == len(self.rgb_list) and not self.balance:
                t = torch.arange(N).long()
            else:
                if continuous and N > 1:
                    t0 = torch.randint(
                        low=0, high=len(self.rgb_list) - N, size=(1,)
                    ).squeeze()
                    t = torch.arange(t0, t0 + N).long()
                else:
                    prob = (
                        self.selection_prob
                        if not force_uniform
                        else (
                            np.ones_like(self.selection_prob) / len(self.selection_prob)
                        )
                    )
                    t = torch.from_numpy(
                        np.random.choice(len(self.rgb_list), N, p=prob)
                    ).long()
        if return_index:
            return t
        gt_rgb = self.rgb_list[t]  # B,H,W,3
        gt_mask = self.mask_list[t]  # B,H,W
        gt_normal = self.normal_list[t] # B,H,W,3
        gt_depth = self.depth_list[t]
        K = self.K_list[t]  # B,3,3
        pose_base = self.pose_base_list[t]
        pose_rest = self.pose_rest_list[t]
        global_trans = self.global_trans_list[t]
        style_feature = self.style_feature[t]

        return (
            gt_rgb.to(device),
            gt_mask.to(device),
            gt_normal.to(device),
            gt_depth.to(device),
            K.to(device),
            pose_base,
            pose_rest,
            global_trans,
            t.to(device),
            style_feature.to(device)
        )

    def prepare_for_fitting(self, dataset):
        rgb_list, mask_list, normal_list, depth_list, K_list, pose_list, global_trans_list, style_feature_list = [], [], [], [], [], [], [], []
        for t in tqdm(range(len(dataset))):
            poses, meta = dataset[t]
            betas, pose, global_trans = (
                poses["smpl_beta"],
                poses["smpl_pose"],
                poses["smpl_trans"],
            )
            betas, pose, global_trans = (
                torch.from_numpy(betas),
                torch.from_numpy(pose),
                torch.from_numpy(global_trans),
            )
            pose_list.append(pose), global_trans_list.append(global_trans)
            K = torch.from_numpy(poses["K"])
            gt_rgb = torch.from_numpy(poses["rgb"])
            mask = torch.from_numpy(poses["mask"])
            normal = torch.from_numpy(poses["normal"])
            depth = torch.from_numpy(poses["depth"])
            depth = depth * mask
            gt_rgb = gt_rgb * mask.unsqueeze(-1) #+ (1 - mask.unsqueeze(-1))
            # ! white background
            self.H, self.W = gt_rgb.shape[:2]  # H,W,3
            rgb_list.append(gt_rgb), K_list.append(K)
            mask_list.append(mask)
            normal_list.append(normal)
            depth_list.append(depth)
            style_feature_list.append(poses["style_feature"])
        pose_list = torch.stack(pose_list)  # T, 24, 3
        global_trans_list = torch.stack(global_trans_list)
        rgb_list = torch.stack(rgb_list)  # T, H, W, 3
        mask_list = torch.stack(mask_list)  # T, H, W
        normal_list = torch.stack(normal_list) # T, H, W, 3
        depth_list = torch.stack(depth_list)
        K_list = torch.stack(K_list)  # T, 3, 3
        style_feature_list = torch.stack(style_feature_list)
        

        if not self.balance:
            self.selection_prob = np.ones(rgb_list.shape[0]) / rgb_list.shape[0]
            return rgb_list, mask_list, normal_list, depth_list, K_list, pose_list, global_trans_list, betas, style_feature_list

        # * Weight the views
        if pose_list.ndim == 3:
            global_rot = [p[0].detach().cpu().numpy() for p in pose_list]
        else: # dog
            global_rot = [p[:3].detach().cpu().numpy() for p in pose_list]
        rot_list = []
        for w in tqdm(global_rot):
            angle = np.linalg.norm(w)
            dir = w / (angle + 1e-6)
            mat = axangle2mat(dir, angle)
            rot_list.append(mat)
        # use first frame as eye
        angle_list = []
        BASE_R = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
        for R in rot_list:
            dR = np.matmul(BASE_R.T, R)
            angle = mat2axangle(dR)[1]
            angle_list.append(angle)
        angle_list = np.array(angle_list)

        # angle_list = np.sort(angle_list)
        kde = KernelDensity(kernel="gaussian", bandwidth=np.pi / 36.0).fit(
            np.concatenate(
                [angle_list - 2 * np.pi, angle_list, angle_list + 2 * np.pi]
            )[:, None]
        )  # Pad
        log_dens = kde.score_samples(angle_list[:, None])
        viz_angle = np.linspace(-np.pi, np.pi, 1000)
        log_viz_dens = kde.score_samples(viz_angle[:, None])

        _view_prob = np.exp(log_dens)
        _view_prob = _view_prob / np.sum(_view_prob)
        selection_prob = 1.0 / _view_prob
        selection_prob = selection_prob / np.sum(selection_prob)
        # print(_view_prob)

        self.selection_prob = selection_prob

        # for viz purpose
        self.viz_angle = viz_angle
        self.angle_list = angle_list
        self.log_viz_dens = log_viz_dens

        return rgb_list, mask_list, normal_list, depth_list, K_list, pose_list, global_trans_list, betas, style_feature_list
    
    def viz_selection_prob(self, save_fn):
        if not self.balance:
            return
        fig = plt.figure(figsize=(12, 2))
        plt.subplot(1, 3, 1)
        plt.hist(self.angle_list, bins=100)
        plt.title("View angle histogram")
        plt.subplot(1, 3, 2)
        plt.plot(self.viz_angle, np.exp(self.log_viz_dens))
        plt.title("angle density")
        plt.subplot(1, 3, 3)
        plt.plot(self.selection_prob), plt.title("Selection Prob")
        plt.savefig(save_fn)
        # plt.show()
        plt.close()