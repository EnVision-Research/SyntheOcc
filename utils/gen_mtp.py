import numpy as np
import cv2
import torch
import os
import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pickle
from scipy import sparse


data_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# data_file = './data/nuscenes/nuscenes_occ_infos_train.pkl'

with open(data_file, "rb") as file:
    nus_pkl = pickle.load(file)



dataroot = "./data/nuscenes"
save_name = "samples_syntheocc_surocc"
gt_path = os.path.join(dataroot, save_name)
os.makedirs(gt_path, exist_ok=True)


CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]
for j in CAM_NAMES:
    os.makedirs(gt_path + "/" + j, exist_ok=True)


def process_func(idx, rank):

    info = nus_pkl["infos"][idx]

    curr_name = info["lidar_path"].split("/")[-1]
    occ_path = f"data/nuscenes/dense_voxels_with_semantic_z-5/{curr_name}.npy"

    occ = np.load(occ_path)[:, [2, 1, 0, 3]]
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]

    num_classes = 16
    occupancy_size = [0.5, 0.5, 0.5]
    grid_size = [200, 200, 16]

    occupancy_size = [0.2, 0.2, 0.2]
    grid_size = [500, 500, 40]

    pc_range = torch.tensor(point_cloud_range)
    voxel_size = (pc_range[3:] - pc_range[:3]) / torch.tensor(grid_size)

    raw_w = 1600
    raw_h = 900

    img_w = 100  # target reso
    img_h = 56

    # img_w = 800
    # img_h = 448

    mtp_num = 96

    f = 0.0055

    def voxel2world(voxel):
        return voxel * voxel_size[None, :] + pc_range[:3][None, :]

    def world2voxel(wolrd):
        return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]

    colors_map = torch.tensor(
        [
            [0, 0, 0, 255],  # unknown
            [255, 158, 0, 255],  #  1 car  orange
            [255, 99, 71, 255],  #  2 truck  Tomato
            [255, 140, 0, 255],  #  3 trailer  Darkorange
            [255, 69, 0, 255],  #  4 bus  Orangered
            [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
            [220, 20, 60, 255],  #  6 bicycle  Crimson
            [255, 61, 99, 255],  #  7 motorcycle  Red
            [0, 0, 230, 255],  #  8 pedestrian  Blue
            [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
            [112, 128, 144, 255],  #  10 barrier  Slategrey
            [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green
            [175, 0, 75, 255],  #  12 other_flat
            [75, 0, 75, 255],  #  13  sidewalk
            [112, 180, 60, 255],  # 14 terrain
            [222, 184, 135, 255],  # 15 manmade Burlywood
            [0, 175, 0, 255],  # 16 vegetation  Green
        ]
    ).type(torch.uint8)

    c, r = np.meshgrid(np.arange(img_w), np.arange(img_h))
    uv = np.stack([c, r])
    uv = torch.tensor(uv)

    depth = (
        torch.arange(0.2, 51.4, 0.2)[..., None][..., None]
        .repeat(1, img_h, 1)
        .repeat(1, 1, img_w)
    )
    
    image_paths = []
    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
    cam_positions = []
    focal_positions = []
    for cam_type, cam_info in info["cams"].items():
        image_paths.append(cam_info["data_path"])
        cam_info["sensor2lidar_rotation"] = torch.tensor(
            cam_info["sensor2lidar_rotation"]
        )
        cam_info["sensor2lidar_translation"] = torch.tensor(
            cam_info["sensor2lidar_translation"]
        )
        cam_info["cam_intrinsic"] = torch.tensor(cam_info["cam_intrinsic"])
        # obtain lidar to image transformation matrix
        lidar2cam_r = torch.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = torch.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info["cam_intrinsic"]
        viewpad = torch.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        lidar2img_rts.append(lidar2img_rt)

        cam_intrinsics.append(viewpad)
        lidar2cam_rts.append(lidar2cam_rt.T)

        cam_position = torch.linalg.inv(lidar2cam_rt.T) @ torch.tensor(
            [0.0, 0.0, 0.0, 1.0]
        ).reshape([4, 1])
        cam_positions.append(cam_position.flatten()[:3])

        focal_position = torch.linalg.inv(lidar2cam_rt.T) @ torch.tensor(
            [0.0, 0.0, f, 1.0]
        ).reshape([4, 1])

        focal_positions.append(focal_position.flatten()[:3])

    occ = torch.tensor(occ)

    dense_vox = torch.zeros(grid_size).type(torch.uint8)
    occ_tr = occ[..., [2, 1, 0, 3]]

    dense_vox[occ_tr[:, 0], occ_tr[:, 1], occ_tr[:, 2]] = occ_tr[:, 3].type(torch.uint8)

    for cam_i in range(len(cam_intrinsics)):

        all_pcl = []
        all_col = []
        all_img_fov = []

        final_img = torch.zeros((img_h, img_w, 3)).type(torch.uint8)

        fuse_img = torch.zeros(
            (
                img_h,
                img_w,
            )
        ).type(torch.uint8)
        depth_map = torch.zeros((img_h, img_w, 1)).type(torch.uint8)

        curr_tr = lidar2cam_rts[cam_i]
        cam_in = cam_intrinsics[cam_i]
        c_u = cam_in[0, 2] / (raw_w / img_w)
        c_v = cam_in[1, 2] / (raw_h / img_h)
        f_u = cam_in[0, 0] / (raw_w / img_w)
        f_v = cam_in[1, 1] / (raw_h / img_h)

        b_x = cam_in[0, 3] / (-f_u)  # relative
        b_y = cam_in[1, 3] / (-f_v)

        dep_num = depth.shape[0]
        mtp_vis = []
        for _ in range(mtp_num):
            mtp_vis.append(
                torch.zeros(
                    (
                        img_h,
                        img_w,
                    )
                ).type(torch.uint8)
            )

        for dep_i in range(dep_num):
            # for dep_i in tqdm.tqdm(range(depth.shape[0])):
            dep_i = dep_num - 1 - dep_i

            uv_depth = (
                torch.cat([uv, depth[dep_i : dep_i + 1]], 0)
                .reshape((3, -1))
                .transpose(1, 0)
            )
            n = uv_depth.shape[0]
            x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
            y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
            pts_3d_rect = torch.zeros((n, 3))
            pts_3d_rect[:, 0] = x
            pts_3d_rect[:, 1] = y
            pts_3d_rect[:, 2] = uv_depth[:, 2]

            new_pcl = torch.cat([pts_3d_rect, torch.ones_like(pts_3d_rect[:, :1])], 1)

            new_pcl = torch.einsum("mn, an -> am", torch.linalg.inv(curr_tr), new_pcl)
            # new_pcl = torch.einsum("mn, an -> am", curr_tr, new_pcl)

            # new_pcl[:, :3] -= 0.1
            new_pcl[:, :3] -= occupancy_size[0] / 2

            new_pcl = world2voxel(new_pcl[:, :3])
            new_pcl = torch.round(new_pcl, decimals=0).type(torch.int32)

            pts_index = torch.zeros((new_pcl.shape[0])).type(torch.uint8)

            valid_flag = (
                ((new_pcl[:, 0] < grid_size[0]) & (new_pcl[:, 0] >= 0))
                & ((new_pcl[:, 1] < grid_size[1]) & (new_pcl[:, 1] >= 0))
                & ((new_pcl[:, 2] < grid_size[2]) & (new_pcl[:, 2] >= 0))
            )

            if valid_flag.max() > 0:
                pts_index[valid_flag] = dense_vox[
                    new_pcl[valid_flag][:, 0],
                    new_pcl[valid_flag][:, 1],
                    new_pcl[valid_flag][:, 2],
                ]

            col_pcl = torch.index_select(colors_map, 0, pts_index.type(torch.int32))

            img_fov = col_pcl[:, :3].reshape((img_h, img_w, 3))
            # cv2.imwrite(f"./exp/mtp/{dep_i:06d}.jpg", img_fov.cpu().numpy()[..., [2,1,0]])
            pts_index = pts_index.reshape(
                (
                    img_h,
                    img_w,
                )
            )
            img_flag = pts_index[..., None].repeat(1, 1, 3)
            final_img[img_flag != 0] = img_fov[img_flag != 0]

            all_img_fov.append(pts_index[None])

            mtp_idx = int(dep_i // (dep_num / mtp_num))
            mtp_vis[mtp_idx][pts_index != 0] = pts_index[pts_index != 0]
            fuse_img[pts_index != 0] = pts_index[pts_index != 0]

            depth_map[pts_index != 0] = dep_i

        save_path = image_paths[cam_i]

        if "samples" in save_path:
            save_path = save_path.replace("samples", save_name)
        if "sweeps" in save_path:
            save_path = save_path.replace(
                "sweeps", save_name.replace("samples", "sweeps")
            )

        final_img = final_img[..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(save_path[:-4] + "_occrgb.png", final_img)

        # rgb_img = cv2.imread(image_paths[cam_i])
        # rgb_img = cv2.resize(rgb_img, (img_w, img_h))
        # final_img = np.concatenate([final_img, rgb_img], 0)
        # # raw_occ_rgb = cv2.imread(save_path[:-4].replace(save_name, 'samples_syntheocc') + '_occrgb.jpg')
        # # final_img = np.concatenate([raw_occ_rgb, final_img], 0)
        # cv2.imwrite('output.jpg', final_img)

        if 1:
            all_img_fov = torch.cat(all_img_fov, 0).type(torch.uint8).flip(0)
            mtp_96 = torch.cat([x[None] for x in mtp_vis], 0).type(torch.uint8).flip(0)

            mtp_96_path = save_path[:-4] + "_mtp96.npz"
            mtp_256_path = save_path[:-4] + "_mtp256.npz"

            sparse_mat = mtp_96.cpu().numpy().reshape((-1, mtp_96.shape[-1]))
            # allmatrix_sp = sparse.coo_matrix(sparse_mat) # 采用行优先的方式压缩矩阵
            allmatrix_sp = sparse.csr_matrix(sparse_mat)  # 采用行优先的方式压缩矩阵
            sparse.save_npz(mtp_96_path, allmatrix_sp)  # 保存稀疏矩阵

            sparse_mat = all_img_fov.cpu().numpy().reshape((-1, all_img_fov.shape[-1]))
            # allmatrix_sp = sparse.coo_matrix(sparse_mat) # 采用行优先的方式压缩矩阵
            allmatrix_sp = sparse.csr_matrix(sparse_mat)  # 采用行优先的方式压缩矩阵
            sparse.save_npz(mtp_256_path, allmatrix_sp)  # 保存稀疏矩阵

            # allmatrix_sp = sparse.load_npz('allmatrix_sparse.npz')
            # allmatrix = allmatrix_sp.toarray().reshape(mtp_96.shape)

            fuse_path = save_path[:-4] + "_fuseweight.png"
            cv2.imwrite(fuse_path, fuse_img.cpu().numpy())

            depth_map_path = save_path[:-4] + "_depthmap.png"
            cv2.imwrite(depth_map_path, depth_map[..., 0].cpu().numpy())


def run_inference(rank, world_size, pred_results, input_datas):
    if rank is not None:
        # dist.init_process_group("gloo", rank=rank, world_size=world_size)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
    print(rank)

    torch.set_default_device(rank)

    all_list = input_datas[rank]  # [::6]

    for i in tqdm.tqdm(all_list):
        process_func(i, rank)


if __name__ == "__main__":
    os.system("export NCCL_SOCKET_IFNAME=eth1")

    from torch.multiprocessing import Manager

    world_size = 8

    all_len = len(nus_pkl["infos"])
    val_len = all_len // 8 * 8
    print(all_len, val_len)

    all_list = torch.arange(val_len).cpu().numpy()
    # all_list = torch.arange(16).cpu().numpy()

    all_list = np.split(all_list, 8)

    input_datas = {}
    for i in range(world_size):
        input_datas[i] = list(all_list[i])
        print(len(input_datas[i]))

    input_datas[0] += list(np.arange(val_len, all_len))

    for i in range(world_size):
        print(len(input_datas[i]))

    # run_inference(0, 1, None, input_datas)

    with Manager() as manager:
        pred_results = manager.list()
        mp.spawn(
            run_inference,
            nprocs=world_size,
            args=(
                world_size,
                pred_results,
                input_datas,
            ),
            join=True,
        )
