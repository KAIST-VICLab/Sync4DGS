import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
path_to_add = os.path.join(project_root, "submodules", "spatrackerv2")
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from submodules.spatrackerv2.models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import numpy as np
import cv2
import torch
import torchvision.transforms as tvt
import moviepy.editor as mp
from submodules.spatrackerv2.models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from submodules.spatrackerv2.models.SpaTrackV2.models.utils import get_points_on_a_grid
from rich import print
import argparse
import decord
from submodules.spatrackerv2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import (
    VGGT4Track,
)
from submodules.spatrackerv2.models.SpaTrackV2.models.vggt4track.utils.load_fn import (
    preprocess_image,
)
from submodules.spatrackerv2.models.SpaTrackV2.models.vggt4track.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.colmap_utils import align_depth_with_colmap, align_dyntrack_with_colmap, getNerfppNorm
from submodules.spatrackerv2.motion_utils import *

from scene.dataset_readers import storePly, fetchPly

import imageio.v2 as iio

# UINT16_MAX = 65535


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="infer")  # "load" or "infer"
    parser.add_argument("--data_dir", type=str, default="assets/example0")  # unsync.
    parser.add_argument("--mask_dir", type=str, default="assets/example0")  # sync.
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument(
        "--grid_size", type=int, default=70
    )  # make (grid_size X grid_size)-개 query points
    parser.add_argument(
        "--query_interval", type=int, default=-1
    )  # make dense grid query points for every args.query_interval frame in N_frame_origin's fps # -1 means to make dense grid query points only at the first frame
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument(
        "--fps", type=int, default=2
    )  # higher value for lower GPU consumption
    parser.add_argument(
        "--global_pose_from", type=str, default="vggt"
    )  # "vggt" or "colmap"
    parser.add_argument("--start_timestamp", type=int, default="151") # frame: [args.start_timestamp, args.end_timestamp)
    parser.add_argument("--end_timestamp", type=int, default="201")
    parser.add_argument(
        "--vggt4track_mode", type=str, default="once"
    )  # "once" or "chunk"
    parser.add_argument("--use_mask_rescale", action='store_false') # default: True
    parser.add_argument("--use_mask_query", action='store_true') # default: False
    return parser.parse_args()


"""
[ Argument ]
args.data_dir
  - videos
    - f"{args.video_name}.mp4"(RGB) 또는 f"{args.video_name}.npz"(RGBD)
  - masks
    - f"{args.video_name}.png" (= mask at the first frame which is used for making 2D query points at the first frame)
  - spatrackerv2
    - f"{args.video_name}.npz"
------------------------------------------------------------
[ Save ]
data_npz_load["coords"] # (N_frame, N_query, 6) where 6: (x, y, z) in spatracker model's world-coord. + (r, g, b)
data_npz_load["visibs"] # (N_frame, N_query)
data_npz_load["extrinsics"] # (N_frame, 4, 4) # world-to-cam
data_npz_load["intrinsics"] # (N_frame, 3, 3)

data_npz_load["depths"] # (N_frame, H_new, W_new)
data_npz_load["unc_metric"] # (N_frame, H_new, W_new)

data_npz_load["video"] # (N_frame, 3, H_origin, W_origin) # range [0, 1]

data_npz_load["colmap-aligned_coords"] # (N_frame, N_query, 6) where 6: (x, y, z) in colmap's world-coord. + (r, g, b)
data_npz_load["coords_2d"] # (N_frame, N_query, 3) where 3: (x, y, depth) where x in [0, W_new), y in [0, H_new)
data_npz_load["extrinsics_global"] # (N_frame, 4, 4)
data_npz_load["intrinsics_global"] # (N_frame, 3, 3)
"""

def interpolate_tensor(input_tensor, N_frame_origin):
    """linear interpolate along time-axis(dim=0)"""
    origin_shape = input_tensor.shape
    N_frame_old = origin_shape[0]
    if N_frame_old == N_frame_origin:
        return input_tensor

    # (N_frame_old, ...) -> (N_frame_old, C) -> (C, N_frame_old) -> (1, C, N_frame_old)
    input_tensor = input_tensor.view(N_frame_old, -1).permute(1, 0).unsqueeze(0).float()

    # linear interpolation
    interpolated_tensor = torch.nn.functional.interpolate(
        input_tensor, size=N_frame_origin, mode="linear", align_corners=False
    )

    # (1, C, N_frame_origin) -> (C, N_frame_origin) -> (N_frame_origin, C) -> (N_frame_origin, ...)
    interpolated_tensor = (
        interpolated_tensor.squeeze(0)
        .permute(1, 0)
        .view(N_frame_origin, *origin_shape[1:])
    )
    return interpolated_tensor.to(dtype=input_tensor.dtype)


def load_pose(args):
    # Obtain Camera Pose at multi-view global world coord. from COLMAP or VGGT output
    colmap_0_dir = (
        args.data_dir
        + f"/dataset_{args.global_pose_from}/colmap_{args.start_timestamp:05d}"
    )
    try:
        cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    cam_id = int(args.video_name.split("_")[1])  # cam_03 -> 3
    extr = cam_extrinsics[cam_id + 1]
    intr = cam_intrinsics[extr.camera_id]

    R = np.transpose(qvec2rotmat(extr.qvec))  # (3, 3) # C2W
    T = np.array(extr.tvec)
    # TODO: This is for Technicolor dataset @ readColmapSceneInfoTechnicolor() @ scene/dataset_readers.py
    nerf_normalization = getNerfppNorm(cam_extrinsics)
    print(
        f"Maximum Camera Center Radius: {nerf_normalization['radius']}\nThis will be used as distance unit"
    )
    T = T / nerf_normalization["radius"]

    K = np.eye(3)
    K[0, 0] = intr.params[0]  # * 0.5
    K[0, 2] = intr.params[2]  # * 0.5
    K[1, 1] = intr.params[1] if intr.model == "PINHOLE" else intr.params[0]  # * 0.5
    K[1, 2] = intr.params[3]  # * 0.5

    return R, T, K, nerf_normalization


def load_pcd(args, nerf_normalization):
    colmap_0_dir = (
        args.data_dir
        + f"/dataset_{args.global_pose_from}/colmap_{args.start_timestamp:05d}"
    )
    ply_path = os.path.join(colmap_0_dir, "sparse/0", "points3D.ply")
    bin_path = os.path.join(colmap_0_dir, "sparse/0", "points3D.bin")
    txt_path = os.path.join(colmap_0_dir, "sparse/0", "points3D.txt")

    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        # TODO: This is for Technicolor dataset @ readColmapSceneInfoTechnicolor() @ scene/dataset_readers.py
        xyz = xyz / nerf_normalization["radius"]
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    return pcd

if __name__ == "__main__":
    args = parse_args()
    print("Arguments:")
    print(f"--query_interval {args.query_interval} --vo_points {args.vo_points} --grid_size {args.grid_size} --fps {args.fps} --start_timestamp {args.start_timestamp} --end_timestamp {args.end_timestamp} --global_pose_from {args.global_pose_from} --use_mask_rescale {args.use_mask_rescale} --use_mask_query {args.use_mask_query}")
    out_dir = args.data_dir + f"/spatrackerv2/{args.video_name}/masked_query" if args.use_mask_query else args.data_dir + f"/spatrackerv2/{args.video_name}/randomgrid_query"
    # fps
    fps = int(args.fps)
    mask_dir = args.mask_dir + f"/masks/{args.video_name}"

    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    """ Load unsync. info to obtain mask """
    cam_id = int(args.video_name.split("_")[1])  # cam_03 -> 3
    start_frame_i = None
    with open(os.path.join(args.data_dir, "unsync_data_info.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Camera"):
                continue
            parts = line.split("|")
            camera_str = parts[0].strip()  # e.g. Camera 06
            cam_i = int(camera_str.split()[1])  # e.g. 6
            frames_str = parts[3].strip()  # e.g. Frames: 8–279
            frame_start, frame_end = map(
                int, frames_str.split(":")[1].strip().replace("–", "-").split("-")
            )
            if cam_i == cam_id:
                start_frame_i = frame_start

    if args.data_type == "load":
        npz_dir = args.data_dir + f"/spatrackerv2/{args.video_name}.npz"
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        # TODO: tapip format
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)[
            args.start_timestamp : args.end_timestamp
        ]  # frame range [args.start_timestamp, args.end_timestamp)
        N_frame_origin = len(video_tensor)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"][
            args.start_timestamp : args.end_timestamp
        ]
        depth_tensor = depth_tensor[::fps]
        intrs = data_npz_load["intrinsics"][args.start_timestamp : args.end_timestamp]
        intrs = intrs[::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])[
            args.start_timestamp : args.end_timestamp
        ]
        extrs = extrs[::fps]
        unc_metric = None
    elif args.data_type == "infer":
        vid_dir = args.data_dir + f"/videos/{args.video_name}.mp4"
        video_reader = decord.VideoReader(vid_dir)
        video_tensor = torch.from_numpy(
            video_reader.get_batch(range(len(video_reader))).asnumpy()
        ).permute(0, 3, 1, 2)[
            args.start_timestamp : args.end_timestamp
        ]  # Convert to tensor and permute to (N, C, H, W)
        origin_video_tensor = video_tensor.clone()
        N_frame_origin = len(origin_video_tensor)
        video_tensor = video_tensor[::fps].float()

        """ Load extrinsic, intrinsic, scene scale """
        R, T, K, nerf_normalization = load_pose(args)
        pcd = load_pcd(args, nerf_normalization)

        """ Downscale image and intrinsic for efficient inference """
        # process the image tensor
        _, _, H_orig, W_orig = video_tensor.shape
        # For (H_orig, W_orig)=(1088, 2048), "resize" occurred and "crop" didn't occur and "pad" didn't occur.
        video_tensor = preprocess_image(video_tensor)[None]
        _, _, _, H_down, W_down = video_tensor.shape  # (1, T, C, H, W)

        scale_w, scale_h = W_down / W_orig, H_down / H_orig
        scale_h, scale_w = H_down / H_orig, W_down / W_orig
        K_down = K.copy()
        K_down[0, 0] *= scale_w  # fx
        K_down[1, 1] *= scale_h  # fy
        K_down[0, 2] *= scale_w  # cx
        K_down[1, 2] *= scale_h  # cy
        K = K_down

        if args.vggt4track_mode == "chunk":
            batch_size = 150  # TODO: hard-coded according to GPU VRAM
            extrinsic_list, intrinsic_list, depth_map_list, depth_conf_list = (
                [],
                [],
                [],
                [],
            )
            with torch.no_grad():
                # Predict attributes including cameras, depth maps, and point maps.
                N_frame_total = video_tensor.shape[1]
                for i in tqdm.tqdm(
                    range(0, N_frame_total, batch_size),
                    desc="Processing video in batches",
                ):
                    start_idx = i
                    end_idx = min(i + batch_size, N_frame_total)
                    chunk = video_tensor[:, start_idx:end_idx]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictions_chunk = vggt4track_model(chunk.cuda() / 255)
                        extrinsic_list.append(predictions_chunk["poses_pred"].squeeze())
                        intrinsic_list.append(predictions_chunk["intrs"].squeeze())
                        depth_map_list.append(
                            predictions_chunk["points_map"][..., 2].squeeze()
                        )
                        depth_conf_list.append(
                            predictions_chunk["unc_metric"].squeeze()
                        )
            # extrinsic, intrinsic from vggt4track_model are not utilized (Instead, colmap extrinsic, intrinsic is utilized)
            extrinsic, intrinsic, depth_map, depth_conf = (
                torch.cat(extrinsic_list, dim=0),
                torch.cat(intrinsic_list, dim=0),
                torch.cat(depth_map_list, dim=0),
                torch.cat(depth_conf_list, dim=0),
            )
        elif args.vggt4track_mode == "once":
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Predict attributes including cameras, depth maps, and point maps.
                    predictions = vggt4track_model(video_tensor.cuda() / 255)
                    extrinsic, intrinsic = (
                        predictions["poses_pred"],
                        predictions["intrs"],
                    )
                    depth_map, depth_conf = (
                        predictions["points_map"][..., 2],
                        predictions["unc_metric"],
                    )
        video_tensor = video_tensor.squeeze()

        """ extrinsic, intrinsic: Use colmap ver. instead of vggt4track_model ver. """
        # extrs = extrinsic.squeeze().cpu().numpy() # (N_frame, 4, 4)
        extrs = np.repeat(
            getWorld2View2(R, T)[None, :, :], video_tensor.shape[0], axis=0
        )
        extrs_colmap = np.repeat(
            getWorld2View2(R, T)[None, :, :], video_tensor.shape[0], axis=0
        )
        # intrs = intrinsic.squeeze().cpu().numpy() # (N_frame, 3, 3)
        intrs = np.repeat(K[None, :, :], video_tensor.shape[0], axis=0)
        intrs_colmap = np.repeat(K[None, :, :], video_tensor.shape[0], axis=0)

        """ Align depth map to colmap """
        depth_tensor = depth_map.squeeze().cpu().numpy()  # (N_frame, H, W)
        depth_tensor = align_depth_with_colmap(
            args, depth_tensor, pcd, extrs, intrs, start_frame_i
        )

        # NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        data_npz_load = {}
    torch.cuda.empty_cache()

    # get all data pieces
    viz = True
    os.makedirs(out_dir, exist_ok=True)

    # with open(cfg_dir, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = easydict.EasyDict(cfg)
    # cfg.out_dir = out_dir
    # cfg.model.track_num = args.vo_points
    # print(f"Downloading model from HuggingFace: {cfg.ckpts}")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points

    model.eval()
    model.to("cuda")
    viser = Visualizer(
        save_dir=out_dir, grayscale=True, fps=30, pad_value=0, tracks_leave_trace=5
    )

    grid_size = args.grid_size

    # get frame H W
    if video_tensor is None:
        video_path = None
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]

    grid_pts = get_points_on_a_grid(
        grid_size, (frame_H, frame_W), device="cpu"
    )  # (1, grid_size*grid_size, 2) where 2: (x, y)

    query_list = []
    query_interval = (
        N_frame_origin + 100
        if args.query_interval == -1
        else args.query_interval
    )
    scene_name = os.path.basename(args.data_dir)
    cam_id = int(args.video_name.split("_")[1])  # cam_03 -> 3
    for frame_i in range(0, N_frame_origin, query_interval):
        # mask는 sync. data로 만들었기 때문에 unsync. frame index를 sync. frame index로 mapping해야 함
        sync_frame_i = (
            start_frame_i + (args.start_timestamp - 1) + frame_i
        )  # TODO: fps-ratio도 1/2배 해서 unsync. data 만들었다면 frame_i / (1/2) + start_frame_i로 수정
        # frame_i: range [0, N_frame_origin - 1]
        # start_frame_i: range [00001, ...]
        mask_files = (
            mask_dir + f"/{scene_name}_undist_{sync_frame_i:05d}_{cam_id:02d}.png"
        )
        if os.path.exists(mask_files) and args.use_mask_query:
            mask = cv2.imread(mask_files)  # (H, W)
            mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
            mask = mask.sum(axis=-1) > 0
            
            vis_mask = np.zeros((video_tensor.shape[2], video_tensor.shape[3]), dtype=np.uint8)
            vis_mask[mask == True] = 1
            vis_mask = (vis_mask * 255).astype(np.uint8)
            out_path = os.path.join(out_dir, f"mask_{args.video_name}.png")
            cv2.imwrite(out_path, vis_mask)

            # Sample mask values at grid points and filter out points where mask=0
            grid_pts_int = grid_pts[0].long()
            grid_pts_int[..., 1] = torch.clamp(grid_pts_int[..., 1], 0, mask.shape[0] - 1) # [0, H)
            grid_pts_int[..., 0] = torch.clamp(grid_pts_int[..., 0], 0, mask.shape[1] - 1) # [0, W)
            mask_values = mask[
                grid_pts_int[..., 1], grid_pts_int[..., 0]
            ]  # (grid_size*grid_size,) # value: 0(False) or 1(True)
            grid_pts = grid_pts[
                :, mask_values
            ]  # (1, N_masked_query, 2) where 2: (x, y) where N_masked_query < grid_size*grid_size
        else:
            mask = np.ones_like(video_tensor[0, 0].numpy()) > 0  # (H, W)

        frame_idx = torch.ones_like(grid_pts[:, :, :1]) * int(frame_i / args.fps)
        query_list.append(
            torch.cat([frame_idx, grid_pts], dim=2)[0].cpu().numpy()
        )  # list of (N_masked_query, 3) where 3: (t, x, y)

        if frame_i == 0:
            out_path = os.path.join(out_dir, f"query_{args.video_name}.png")
            vis_points(args, grid_pts[0].long().cpu().numpy(), out_path)
    if len(query_list) > 1:
        query_xyt = np.concatenate(
            query_list, axis=0
        )  # (N_masked_query_total, 3) where 3: (t, x, y)
    else:
        query_xyt = query_list[0]

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj,  # (N_frame, 4, 4)
            intrs,  # (N_frame, 3, 3)
            point_map,  # (N_frame, 3, H, W) # uplifted pixel
            conf_depth,  # (N_frame, H, W) # range [0, 1]
            track3d_pred,  # (N_frame, N_query, 6) where 6: (x,y,z) in cam-coord. + (r,g,b)
            track2d_pred,  # (N_frame, N_query, 3) where 3: (x,y) in pixel-coord. + depth
            vis_pred,  # visibility # (N_frame, N_query, 1) # range [0+0.2, 1+0.2]
            conf_pred,  # confidence on model prediction # (N_frame, N_query, 1) # range [0, 1]
            video,  # (N_frame, 3, H, W)
        ) = model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=1,
            full_point=False,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=unc_metric,
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.2,
        )

        vis_pred = vis_pred.reshape(
            track3d_pred.shape[0], track3d_pred.shape[1]
        )  # (N_frame, N_query) # range [0+0.2, 1+0.2]
        conf_pred = conf_pred.reshape(
            track3d_pred.shape[0], track3d_pred.shape[1]
        )  # (N_frame, N_query) # range [0, 1]

        if args.fps > 1:
            c2w_traj = interpolate_tensor(c2w_traj, N_frame_origin)
            intrs = interpolate_tensor(intrs, N_frame_origin)
            point_map = interpolate_tensor(point_map, N_frame_origin)
            conf_depth = interpolate_tensor(conf_depth, N_frame_origin)
            track3d_pred = interpolate_tensor(track3d_pred, N_frame_origin)
            track2d_pred = interpolate_tensor(track2d_pred, N_frame_origin)
            vis_pred = interpolate_tensor(vis_pred, N_frame_origin)
            video = interpolate_tensor(video, N_frame_origin)

        vis_pred_binary = (vis_pred > 0.5).float()  # range [0+0.2, 1+0.2] -> 0 or 1

        print(
            f"original track3d shape for {cam_id:02d} as (N_frame, N_query, 6)={track3d_pred.shape}"
        )
        if not args.use_mask_query:
            T_c2w = -R @ T  # R: C2W
            # track3d_cam @ R_c2w.T == (R_c2w @ track3d_cam.T).T
            track3d_colmap = (
                np.einsum(
                    "ij,tnj->tni",
                    R,
                    track3d_pred[:, :, :3].cpu().numpy(),
                )
                + T_c2w[None, None, :]
            )
            track3d_spatrack = (
                torch.einsum(
                    "tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[:, :, :3].cpu()
                )
                + c2w_traj[:, :3, 3][:, None, :]
            ).cpu().numpy()
            # motion_mag = get_scene_motion_2d_displacement(
            #     track3d_colmap,
            #     vis_pred_binary.cpu().numpy().astype(bool),
            #     extrs_colmap[0],
            #     intrs_colmap[0],
            #     video_tensor.shape[2],
            #     video_tensor.shape[3],
            #)
            motion_mag = get_scene_motion_3d_displacement(
                track3d_spatrack,
                vis_pred_binary.cpu().numpy().astype(bool)
            )
            track_mask = (motion_mag > 0.16).any(axis=0) # hard-coded # 16 for 2d_displacement # (N_query,)
            # motion_mag.mean(), min(), max(): 0.031, 0.0, 0.619
            # motion_mag.mean(), min(), max(): 0.036, 0.0, 0.646
            track3d_pred = track3d_pred[:, track_mask]  # (N_frame, N_query_reduced, 6)
            track2d_pred = track2d_pred[:, track_mask]  # (N_frame, N_query_reduced, 3)
            vis_pred = vis_pred[:, track_mask]  # (N_frame, N_query_reduced)
            vis_pred_binary = vis_pred_binary[
                :, track_mask
            ]  # (N_frame, N_query_reduced)
            print(
                f"dynamic track3d shape for {cam_id:02d} as (N_frame, N_query, 6)={track3d_pred.shape}"
            )

            # track3d_colmap = (
            #     np.einsum(
            #         "ij,tnj->tni",
            #         R,
            #         track3d_pred[:, :, :3].cpu().numpy(),
            #     )
            #     + T_c2w[None, None, :]
            # )
            # location = track3d_colmap[
            #     np.argmax(
            #         (
            #             ~np.isnan(track3d_colmap).any(axis=-1)
            #             & vis_pred_binary.cpu().numpy().astype(bool)
            #         ),
            #         axis=0,
            #     ),
            #     np.arange(track3d_colmap.shape[1]),
            #]  # (N_query, 3) # 각 query가 첫 번째로 나타나는 frame의 xyz좌표
            # index = np.argsort(
            #     location[:, 1]
            # )  # 각 query가 첫번째로 나타나는 frame의 y좌표
            # track3d_pred = track3d_pred[:, index]
            # track2d_pred = track2d_pred[:, index]
            # vis_pred = vis_pred[:, index]
            # vis_pred_binary = vis_pred_binary[:, index]
            # visible_length = (
            #     vis_pred_binary.cpu().numpy().sum(axis=0)
            # )  # (N_query_reduced,)
            # print(visible_length[0:10])
            # masks = np.zeros(track3d_pred.shape[1])  # (N_query_reduced,)
            # masks[visible_length > np.percentile(visible_length, 10)] = (
            #     1  # sampling tracks that have long visibilities
            # )
            # masks = masks.astype(bool)
            # track3d_pred = track3d_pred[:, masks]
            # track2d_pred = track2d_pred[:, masks]
            # vis_pred = vis_pred[:, masks]
            # vis_pred_binary = vis_pred_binary[:, masks]
            # print(
            #     f"long dynamic track3d shape for {cam_id:02d} as (N_frame, N_query, 6)={track3d_pred.shape}"
            # )
            # masks = np.random.uniform(size=track3d_pred.shape[1]) < min(
            #     256 / track3d_pred.shape[1], 1
            # )
            # masks = masks.astype(bool)
            # track3d_pred = track3d_pred[:, masks]
            # track2d_pred = track2d_pred[:, masks]
            # vis_pred = vis_pred[:, masks]
            # vis_pred_binary = vis_pred_binary[:, masks]
            # print(
            #     f"long dynamic sampled track3d shape for {cam_id:02d} as (N_frame, N_query, 6)={track3d_pred.shape}"
            # )

        try:
            video_tensor = origin_video_tensor
        except NameError:
            video_tensor = video

        extrs_colmap = np.repeat(
            getWorld2View2(R, T)[None, :, :], N_frame_origin, axis=0
        )
        intrs_colmap = np.repeat(K[None, :, :], N_frame_origin, axis=0)

        # # resize the results to avoid too large I/O Burden
        # # depth and image, the maximum side is 336
        # max_size = 336
        # h, w = video.shape[2:]
        # scale = min(max_size / h, max_size / w)
        # if scale < 1:
        #     new_h, new_w = int(h * scale), int(w * scale)
        #     video = tvt.Resize((new_h, new_w))(video)
        #     video_tensor = tvt.Resize((new_h, new_w))(video_tensor)
        #     point_map = tvt.Resize((new_h, new_w))(point_map)
        #     conf_depth = tvt.Resize((new_h, new_w))(conf_depth)
        #     track2d_pred[..., :2] = track2d_pred[..., :2] * scale
        #     intrs[:, :2, :] = intrs[:, :2, :] * scale
        #     intrs_colmap[:, :2, :] = intrs_colmap[:, :2, :] * scale
        #     # if depth_tensor is not None:
        #     #     if isinstance(depth_tensor, torch.Tensor):
        #     #         depth_tensor = tvt.Resize((new_h, new_w))(depth_tensor)
        #     #     else:
        #     #         depth_tensor = tvt.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        if viz:
            filename = f"{args.video_name}_masked_query" if args.use_mask_query else f"{args.video_name}_randomgrid_query"
            viser.visualize(
                video=video[None],
                tracks=track2d_pred[None][..., :2],
                visibility=vis_pred[None],
                filename=filename
            )

        # save as the tapip3d format
        # 3D track at monocular world coord.
        data_npz_load["coords"] = np.concatenate(((torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[:, :, :3].cpu()) + c2w_traj[:, :3, 3][:, None, :]).cpu().numpy(), track3d_pred[:, :, 3:].cpu().numpy()), axis=2)
        # monocular world-to-cam
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:, 2, ...]
        depth_save[conf_depth < 0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy() / 255
        data_npz_load["visibs"] = vis_pred_binary.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()

        data_npz_load["coords_2d"] = track2d_pred.cpu().numpy()

        T_c2w = -R @ T  # R: C2W
        # 3D track at multi-view global world coord. [radius]
        # track3d_cam @ R_c2w.T == (R_c2w @ track3d_cam.T).T
        data_npz_load["colmap-aligned_coords"] = np.concatenate((
            np.einsum(
                "ij,tnj->tni",
                R,
                track3d_pred[:, :, :3].cpu().numpy(),
            )
            + T_c2w[None, None, :], track3d_pred[:, :, 3:].cpu().numpy()), axis=2
        )
        data_npz_load["extrinsics_colmap"] = extrs_colmap
        data_npz_load["intrinsics_colmap"] = intrs_colmap

        """ Rescale SpaTrackerv2 3D track depth to align with colmap pcd' dynamic part """
        track3d_pos = data_npz_load["colmap-aligned_coords"][..., :3]  # (N_frame, N_query, 3)
        track3d_color = data_npz_load["colmap-aligned_coords"][..., 3:]  # (N_frame, N_query, 3)
        visib = data_npz_load["visibs"].astype(bool)  # (N_frame, N_query)
        extrinsics = data_npz_load["extrinsics_colmap"]  # (N_frame, 4, 4)
        intrinsics = data_npz_load["intrinsics_colmap"]  # (N_frame, 3, 3)
        track3d_pos = align_dyntrack_with_colmap(args, track3d_pos, visib, pcd, extrinsics, intrinsics, frame_H, frame_W, start_frame_i)
        data_npz_load["colmap-aligned_coords"] = np.concatenate((track3d_pos, track3d_color), axis=2)

        out_file = out_dir + f"/{args.video_name}.npz"
        np.savez(out_file, **data_npz_load)
        a=0
        print(
            f"Results saved to {out_file}.\nTo visualize them with tapip3d, run: [bold yellow]bash tapip3d_viz.sh[/bold yellow]"
        )
