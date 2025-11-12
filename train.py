#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import sys
import uuid
import math
from random import randint
import gc

import torch
import tqdm
from PIL import Image
import joblib
from datetime import datetime

from gaussian_renderer import render, network_gui
from scene import Scene, getmodel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.colmap_utils import getNerfppNorm, fetchPly, storePly

from glob import glob
import pickle
from track_utils.uplift_track import *
sys.path.append(os.path.abspath('submodules/pytorch-softdtw-cuda'))
from soft_dtw_cuda import *
from utils.DTW_feature import slice_features, extract_features, _cluster_tracks_dbscan
from submodules.spatrackerv2.models.SpaTrackV2.utils.visualizer import Visualizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb

TENSORBOARD_FOUND = False # wandb
# TENSORBOARD_FOUND = True # tensorboard

torch.set_default_dtype(torch.float32)

def training_stage1(dataset_arg):
    """
    Return coarsely estimated time_offset_list: List[float], fps_ratio_list: List[float]
    by Deterministic (Soft-)DTW
    """
    # rgb_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "images")
    unsync_dir = os.path.dirname(dataset_arg.source_path) # ".../Birthday/dataset_colmap" -> ".../Birthday"
    depth_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "depths")
    colmap_0_dir = os.path.join(dataset_arg.source_path, f"colmap_{dataset_arg.start_timestamp:05d}") # info. at start_timestamp-th frame
    if dataset_arg.track_type == "cotracker":
        track2d_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "cotracker2d") # ".../Birthday/sync4dgs" -> ".../Birthday"
        track3d_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "cotracker3d")
    elif dataset_arg.track_type == "tapir":
        track2d_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "tapir2d")
        track3d_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "tapir3d")
    elif dataset_arg.track_type == "spatracker":
        track3d_dir = os.path.join(os.path.dirname(dataset_arg.source_path), "spatrackerv2")

    """ Load GT time-offset and GT fps-ratio """
    start_idx_dict = {}
    timeoffset_gt_dict = {}
    with open(os.path.join(unsync_dir, "unsync_data_info.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Camera"):
                continue
            parts = line.split("|")
            camera_str = parts[0].strip()  # e.g. Camera 06
            cam_id = int(camera_str.split()[1])  # e.g. 6
            frames_str = parts[3].strip()  # e.g. Frames: 8–279
            frame_start, frame_end = map(
                int, frames_str.split(":")[1].strip().replace("–", "-").split("-")
            )
            start_idx_dict[cam_id] = frame_start
    
    with open(os.path.join(unsync_dir, "unsync_data_info.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Camera"):
                continue
            parts = line.split("|")
            camera_str = parts[0].strip()  # e.g. Camera 06
            cam_id = int(camera_str.split()[1])  # e.g. 6
            timeoffset_gt_dict[cam_id] = start_idx_dict[cam_id] - start_idx_dict[0]

    if dataset_arg.track_type == "tapir":
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
        nerf_normalization = getNerfppNorm(cam_extrinsics)

        cam_path_list = [
            d
            for d in glob(os.path.join(track3d_dir, "cam*"))
            if os.path.isdir(d)
        ]
        
        track3d_dict = {}
        visible_list_dict = {}
        for cam_path in cam_path_list:
            # 3 9 14 11 2 13 4 7 0 6 8 5 1 15 10 12
            cam_id = int(os.path.basename(cam_path).split("_")[1]) # cam_00 -> 0
            print(f"Loading 3D track of cam ID: {cam_id}")
            # rgb_folder = os.path.join(rgb_dir, f"cam_{cam_id:02d}")
            track2d_folder = os.path.join(track2d_dir, f"cam_{cam_id:02d}")
            track3d_folder = os.path.join(track3d_dir, f"cam_{cam_id:02d}")
            depth_folder = os.path.join(depth_dir, f'cam_{cam_id:02d}')
            # if not os.path.exists(depth_folder):
            #     os.makedirs(depth_folder, exist_ok=True)
            if not os.path.exists(track2d_folder):
                os.makedirs(track2d_folder, exist_ok=True)
            if not os.path.exists(track3d_folder):
                os.makedirs(track3d_folder, exist_ok=True)

            ''' Load RGB video '''
            frames_path = glob(os.path.join(cam_path, os.path.basename(os.path.dirname(dataset_arg.source_path)) + "_undist_*.png"))
            sorted_frames_path = sorted(frames_path, key=lambda x: int(os.path.basename(x).split('_')[2])) # sort by frame_id
            frames = [cv2.imread(path)[:, :, ::-1] for path in sorted_frames_path]  # list of (H, W, 3) where 3: BGR -> RGB # range [0, 255]
            frames = np.stack(frames, axis=0).astype(np.uint8) # (N_frame, H, W, 3)

            ''' Load Depth '''
            if cam_id != 15:
                depth_path = os.path.join(depth_folder, f'metric_aligned_da_depth_cam{cam_id:02d}.npz')
            else:
                depth_path = os.path.join(depth_folder, f'colmap_metric_aligned_da_depth_cam{cam_id:02d}.npz')
            # depth_path = os.path.join(depth_folder, f'metric_aligned_depth_from_optimized_3dtrack_cam{cam_id:02d}.npz')
            with np.load(depth_path) as data:
                depths = data['depths'].astype(np.float32)
            # depths: (T, H, W) # float32 # in meters

            """ Load Camera Pose from COLMAP or VGGT
            # cam_id: in range [0, N_cam - 1]
            # cam_extrinsics.keys(), cam_intrinsics.keys(): dict_keys([1, 2, ..., N_cam])
            # cam_extrinsics[i].camera_id == cam_extrinsics[i].id == i where i: in range [1, N_cam]
            """
            extr = cam_extrinsics[cam_id + 1]
            intr = cam_intrinsics[extr.camera_id]
            H, W = intr.height, intr.width
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            T = T / nerf_normalization['radius']
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[0]
                FovY = focal2fov(focal_length_x, H)
                FovX = focal2fov(focal_length_x, W)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, H)
                FovX = focal2fov(focal_length_x, W)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            cx, cy = intr.params[2], intr.params[3]
            # cxr = ((intr.params[2] )/  W - 0.5) 
            # cyr = ((intr.params[3] ) / H - 0.5)
            # K = np.eye(3)
            # K[0, 0] = focal_length_x #* 0.5
            # K[0, 2] = intr.params[2] #* 0.5
            # K[1, 1] = focal_length_y #* 0.5
            # K[1, 2] = intr.params[3] #* 0.5
            # id = int(extr.name[3:5])
            # near = 0.01
            # far = 100
            extr = getWorld2View2(R, T)[:3, :4] # (3, 4) # world-to-cam
            intr_normalized = { # normalized intrinsic
                    'fx': focal_length_x / W,
                    'fy': focal_length_y / H,
                    'cx': cx / W,
                    'cy': cy / H,
                    'k1': 0, # undistorted camera
                    'k2': 0,
                }
            ### obtain list of CameraAZ() ###
            cameras = [] # list of N_frame-개 CameraAZ()
            for fid in range(depths.shape[0]):
                # Technicolor camera is fixed over temporal frames
                cameras.append(CameraAZ(from_json={'extr': extr, 'intr_normalized': intr_normalized}))

            """ Load 3D track (with visible list) """
            # if not os.path.exists(os.path.join(track3d_folder, f"tapir_dynamic_3dtrack_optimized_cam{cam_id:02d}.pkl")):
            if not os.path.exists(os.path.join(track3d_folder, f"tapir_dynamic_3dtrack_original_cam{cam_id:02d}.pkl")):
                # extract dynamic tracks only once for fast inference
                # with open(os.path.join(track3d_folder, f"tapir_3dtrack_optimized_cam{cam_id:02d}.pkl"), "rb") as f:
                #     track3d_json = pickle.load(f)  # dictionary
                #     track3d = Track3d(load_from_json=track3d_json)
                with open(os.path.join(track2d_folder, f'tapir_2dtrack_cam{cam_id:02d}.pkl'), 'rb') as f:
                    track2d = pickle.load(f)
                    track3d = Track3d(track2d['tracks'], track2d['visibles'], depths, cameras, frames, track2d['query_points'])
                    track3d_json = track3d.to_json_format(save_video=False)
                    with open(osp.join(track3d_folder, f'tapir_3dtrack_original_cam{cam_id:02d}.pkl'), 'wb') as f:
                        pickle.dump(track3d_json, f)
                    print(f"original sampled track3d shape for {cam_id:02d} as (N_query, N_frame, 3): ({track3d.track3d.shape[0]}, {track3d.track3d.shape[1]}, {track3d.track3d.shape[2]})")
                    
                    motion_mag = get_scene_motion_2d_displacement(track3d)
                    track_dynamic = track3d.get_new_track((motion_mag > 16).any(axis=1))
                    location = track_dynamic.track3d[
                        np.arange(len(track_dynamic.track3d)),
                        np.argmax(
                            (
                                ~np.isnan(track_dynamic.track3d).any(axis=-1)
                                & track_dynamic.visible_list
                            ),
                            axis=1,
                        ),
                    ]
                    index = np.argsort(location[:, 1])
                    track_dynamic_sort = track_dynamic.get_new_track(index)

                    visible_length = track_dynamic_sort.visible_list.sum(axis=1)  # (N_query,)
                    masks = np.zeros(len(track_dynamic_sort.track3d))
                    masks[visible_length > np.percentile(visible_length, 10)] = 1 # sampling tracks that has long visibilities
                    masks = masks.astype(bool)
                    track_dynamic_sort_sampled = track_dynamic_sort.get_new_track(masks)
                    track_dynamic_sort_sampled = track_dynamic_sort.get_new_track(percentage=min(256 / len(track_dynamic_sort_sampled.track3d), 1))

                    track3d_dynamic_json = track_dynamic_sort_sampled.to_json_format(save_video=False)
                    # with open(osp.join(track3d_folder, f'tapir_dynamic_3dtrack_optimized_cam{cam_id:02d}.pkl'), 'wb') as dynf:
                    with open(osp.join(track3d_folder, f'tapir_dynamic_3dtrack_original_cam{cam_id:02d}.pkl'), 'wb') as dynf:
                        pickle.dump(track3d_dynamic_json, dynf)
                    track3d = track_dynamic_sort_sampled
                    print(f"dynamic sampled track3d shape for {cam_id:02d} as (N_query, N_frame, 3): ({track3d.track3d.shape[0]}, {track3d.track3d.shape[1]}, {track3d.track3d.shape[2]})")
                    # from numpy to tensor
                    # track3d has NaN for invisible points
                    track3d_tensor = torch.nan_to_num(torch.tensor(track3d.track3d, dtype=torch.float32, device=dataset_arg.DEVICE), nan=0.0) # (N_query, N_frame, 3)
                    visible_list_tensor = torch.nan_to_num(torch.tensor(track3d.visible_list, dtype=torch.bool, device=dataset_arg.DEVICE), nan=0.0) # (N_query, N_frame)
                    track3d_dict[cam_id] = track3d_tensor
                    visible_list_dict[cam_id] = visible_list_tensor
            else:
                # with open(os.path.join(track3d_folder, f"tapir_dynamic_3dtrack_optimized_cam{cam_id:02d}.pkl"), "rb") as f:
                with open(os.path.join(track3d_folder, f"tapir_dynamic_3dtrack_original_cam{cam_id:02d}.pkl"), "rb") as f:
                    track3d_json = pickle.load(f)  # dictionary
                    track3d = Track3d(load_from_json=track3d_json)
                    # from numpy to tensor
                    # track3d has NaN for invisible points
                    track3d_tensor = torch.nan_to_num(torch.tensor(track3d.track3d, dtype=torch.float32, device=dataset_arg.DEVICE), nan=0.0) # (N_query, N_frame, 3)
                    visible_list_tensor = torch.nan_to_num(torch.tensor(track3d.visible_list, dtype=torch.bool, device=dataset_arg.DEVICE), nan=0.0) # (N_query, N_frame)
                    track3d_dict[cam_id] = track3d_tensor
                    visible_list_dict[cam_id] = visible_list_tensor
    elif dataset_arg.track_type == "spatracker":
        ply_path = os.path.join(colmap_0_dir, "sparse", "0", "points3D.ply")
        bin_path = os.path.join(colmap_0_dir, "sparse", "0", "points3D.bin")
        txt_path = os.path.join(colmap_0_dir, "sparse", "0", "points3D.txt")
        
        if not os.path.exists(ply_path):
            try:
                cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            nerf_normalization = getNerfppNorm(cam_extrinsics)
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            xyz = xyz / nerf_normalization['radius']
            storePly(ply_path, xyz, rgb)
        
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
        
        cam_path_list = [
            d
            for d in glob(os.path.join(track3d_dir, "cam*"))
            if os.path.isdir(d)
        ]
        
        track3d_dict = {}
        visible_list_dict = {}
        for cam_path in cam_path_list:
            # 3 9 14 11 2 13 4 7 0 6 8 5 1 15 10 12
            cam_id = int(os.path.basename(cam_path).split("_")[1]) # cam_00 -> 0
            print(f"Loading 3D track of cam ID: {cam_id} / Using Motion Mask to Generate 3D track: {dataset_arg.use_masked_track}")
            if dataset_arg.use_masked_track:
                track_path = track3d_dir + f"/cam_{cam_id:02d}" + f"/masked_query/cam_{cam_id:02d}.npz"
            else:
                track_path = track3d_dir + f"/cam_{cam_id:02d}" + f"/randomgrid_query/cam_{cam_id:02d}.npz"
            with np.load(track_path) as data:
                track3d = data["colmap-aligned_coords"] # (N_frame, N_query, 6) where 6: (x,y,z,r,g,b)
                visible_list = data["visibs"] # (N_frame, N_query)
                track3d_dict[cam_id] = torch.from_numpy(track3d).to(torch.float32).to(dataset_arg.DEVICE).transpose(0, 1) # (N_frame, N_query, 6) -> (N_query, N_frame, 6)
                visible_list_dict[cam_id] = torch.from_numpy(visible_list).to(torch.bool).to(dataset_arg.DEVICE).transpose(0, 1) # (N_frame, N_query) -> (N_query, N_frame)
    
    if dataset_arg.track_type == "cotracker" or dataset_arg.track_type == "tapir":
        print("Not Implemented Yet")
        return None, None
    elif dataset_arg.track_type == "spatracker":
        cam_id_sorted = sorted(track3d_dict.keys())
        all_track_list = []
        query_count = []
        for cam_id in cam_id_sorted:
            track = track3d_dict[cam_id]
            all_track_list.append(track) # list of (N_query_camid, N_frame, 6) of length N_cam
            query_count.append(track.shape[0]) # list of int: N_query_camid
        global_combined_tracks = torch.cat(all_track_list, dim=0) # (N_query_camall, N_frame, 6) where N_query_camall = \sum_{camid=0,...,N_cam-1}(N_query_camid)
    
    ref_cam_id = 0 # cam_00 is used as reference in time-offset or fps-ratio
    # For Visualization below
    if dataset_arg.use_masked_track:
        track_path_0 = track3d_dir + f"/cam_{ref_cam_id:02d}/masked_query/cam_{ref_cam_id:02d}.npz"
    else:
        track_path_0 = track3d_dir + f"/cam_{ref_cam_id:02d}/randomgrid_query/cam_{ref_cam_id:02d}.npz"
    with np.load(track_path_0) as data:
        track2d_0 = data["coords_2d"] # (N_frame, N_query, 3) where 3: (x,y,depth)
        visible_0 = data["visibs"] # (N_frame, N_query)
        video_0 = data["video"] # (N_frame, 3, H, W) # range [0, 1]
        track2d_0 = torch.from_numpy(track2d_0).to(torch.float32).to(dataset_arg.DEVICE) # (N_frame, N_query, 3) where 3: (x,y,depth)
        visible_0 = torch.from_numpy(visible_0).to(torch.bool).to(dataset_arg.DEVICE) # (N_frame, N_query)
        video_0 = torch.from_numpy(video_0 * 255).to(torch.uint8).to(dataset_arg.DEVICE) # (N_frame, 3, H, W) # range [0, 255]
    
    """ DBSCAN Clustering with all video's 3D track based on Position + RGB color """
    cluster_labels, N_cluster = _cluster_tracks_dbscan(
        global_combined_tracks,
        eps=dataset_arg.dbscan_eps,
        min_samples=dataset_arg.dbscan_min_samples,
        position_weight=dataset_arg.position_weight,
        color_weight=dataset_arg.color_weight
    ) # (N_query_camall,), int
    if N_cluster > 0:
        labels_per_cam = torch.split(cluster_labels, query_count) # list of (N_query_camid,) of length N_cam
        labels_dict = {cam_id: labels for cam_id, labels in zip(cam_id_sorted, labels_per_cam)} # dict{cam_id: (N_query_camid,)}
        print(f"{N_cluster} clusters by DBSCAN(eps={dataset_arg.dbscan_eps}, min_samples={dataset_arg.dbscan_min_samples}) based on position : color = {dataset_arg.position_weight} : {dataset_arg.color_weight}")
    else:
        print("Warning: No clusters found in global clustering. Exiting.")
        exit()

    """ Estimate optimal time-offset and fps-ratio by finding minimum SoftDTW Loss """
    timeoffset_dict = {}
    fpsratio_dict = {}
    track3d_ref = track3d_dict[ref_cam_id]
    visible_ref = visible_list_dict[ref_cam_id]
    labels_ref = labels_dict[ref_cam_id]
    
    for cam_id in cam_id_sorted:
        # # debug 1.
        # if cam_id != 12 and cam_id != 7:
        #     continue
        if cam_id == ref_cam_id:
            timeoffset_dict[cam_id] = 0.0
            fpsratio_dict[cam_id] = 1.0
            continue
        track3d_cam = track3d_dict[cam_id]
        visible_cam = visible_list_dict[cam_id]
        labels_cam = labels_dict[cam_id]

        offsets_to_test = torch.arange(-40, 41, 1.0)
        fps_ratio = torch.tensor(1.0) 
        losses = []

        sdtw_cuda = SoftDTW(True, gamma=dataset_arg.DTW_gamma, normalize=dataset_arg.DTW_normalize, bandwidth=dataset_arg.DTW_bandwidth)

        cluster_flag = False
        for time_offset in offsets_to_test: 
            """ slice 3d tracks using time_offset and fps_ratio """
            track3d_a, track3d_b, visible_list_a, visible_list_b, N_frame_overlap = slice_features(
                track3d_ref,
                track3d_cam,
                visible_ref,
                visible_cam,
                time_offset,
                fps_ratio
            )
                
            """ extract per-frame features from 3d tracks """
            cluster_labels = torch.cat([labels_ref, labels_cam], dim=0) # (N_query_ref + N_query_camid,)
            seq_a, seq_b, visib_a, visib_b, labels_a, labels_b, N_cluster = extract_features(
                track3d_a,
                track3d_b,
                visible_list_a,
                visible_list_b,
                cluster_labels,
                N_cluster,
                vel_weight=dataset_arg.vel_weight,
                accel_weight=dataset_arg.accel_weight,
                curv_weight=dataset_arg.curv_weight
            )

            # debug 2.
            # """ Visualize Clustering Results: 2D query points with per-cluster color """
            # if time_offset == offsets_to_test[len(offsets_to_test)//2]:
            #     if dataset_arg.use_masked_track:
            #         track_path = track3d_dir + f"/cam_{cam_id:02d}/masked_query/cam_{cam_id:02d}.npz"
            #     else:
            #         track_path = track3d_dir + f"/cam_{cam_id:02d}/randomgrid_query/cam_{cam_id:02d}.npz"
            #     with np.load(track_path) as data:
            #         track2d = data["coords_2d"] # (N_frame, N_query, 3) where 3: (x,y,depth)
            #         visible = data["visibs"] # (N_frame, N_query)
            #         try:
            #             video = data["video"] # (N_frame, 3, H, W) # range [0, 1]
            #         except:
            #             print(f"Cannot Read video.npy for cam_{cam_id:02d}")
            #             continue
            #         _, _, H_origin, W_origin = video.shape
            #         _, H_new, W_new = data["depths"].shape
            #         track2d = torch.from_numpy(track2d).to(torch.float32).to(dataset_arg.DEVICE) # (N_frame, N_query, 3) where 3: (x,y,depth)
            #         visible = torch.from_numpy(visible).to(torch.bool).to(dataset_arg.DEVICE) # (N_frame, N_query)
            #         video = torch.from_numpy(video * 255).to(torch.uint8).to(dataset_arg.DEVICE) # (N_frame, 3, H, W) # range [0, 255]
            #     vis_dir = track3d_dir + f"/cam_{cam_id:02d}/masked_query" if dataset_arg.use_masked_track else track3d_dir + f"/cam_{cam_id:02d}/randomgrid_query"
            #     viser = Visualizer(
            #         save_dir=vis_dir, grayscale=True, fps=30, pad_value=0, tracks_leave_trace=5
            #     )
            #     filename = f"cam_{cam_id:02d}_clustered_masked_query" if dataset_arg.use_masked_track else f"cam_{cam_id:02d}_clustered_randomgrid_query"
            #     filename_0 = "cam_00_clustered_masked_query" if dataset_arg.use_masked_track else "cam_00_clustered_randomgrid_query"
            #     scale_factors = torch.tensor([W_origin / W_new, H_origin / H_new], device=track2d.device)
            #     track2d_scaled = track2d.clone() # (N_frame, N_query, 3)
            #     track2d_scaled[:, :, :2] = track2d[:, :, :2] * scale_factors[None, None, :]
            #     track2d_scaled_0 = track2d_0.clone() # (N_frame, N_query, 3)
            #     track2d_scaled_0[:, :, :2] = track2d_0[:, :, :2] * scale_factors[None, None, :]
            #     viser.visualize(
            #         video=video_0[None],
            #         tracks=track2d_scaled_0[None][..., :2],
            #         visibility=visible_0[None, :, :, None],
            #         filename=filename_0,
            #         rigid_part=labels_a
            #     )
            #     viser.visualize(
            #         video=video[None],
            #         tracks=track2d_scaled[None][..., :2],
            #         visibility=visible[None, :, :, None],
            #         filename=filename,
            #         rigid_part=labels_b
            #     )
                
            """ Calculate Per-Cluster SoftDTW Loss and Average them """
            object_distances = {}
            many_to_many = False
            for i in range(N_cluster):
                seq_a_i = seq_a[i]  # (N_query_a_i, N_frame, D)
                seq_b_i = seq_b[i]  # (N_query_b_i, N_frame, D)
                visib_a_i = visib_a[i].float() # (N_query_a_i, N_frame)
                visib_b_i = visib_b[i].float() # (N_query_b_i, N_frame)
                if seq_a_i.shape[0] == 0 or seq_b_i.shape[0] == 0:
                    continue
                if many_to_many == True:
                    N_query_a_i, N_query_b_i = seq_a_i.shape[0], seq_b_i.shape[0]
                    if seq_a_i.shape[0] == 0 or seq_b_i.shape[0] == 0:
                        continue # There is no dynamic track which belongs to i-th cluster in either video a or b
                    seq_a_i_mat = torch.repeat_interleave(seq_a_i, N_query_b_i, dim=0) # (N_query_a_i * N_query_b_i, N_frame, D) # e.g. [a1, a2] -> [a1, a1, a1, a2, a2, a2] if N_query_b_i=3
                    seq_b_i_mat = seq_b_i.repeat(N_query_a_i, 1, 1) # (N_query_a_i * N_query_b_i, N_frame, D) # e.g. [b1, b2] -> [b1, b2, b3, b1, b2, b3] if N_query_a_i=2
                    sdtw_loss_mat = sdtw_cuda(seq_a_i_mat, seq_b_i_mat).view(N_query_a_i, N_query_b_i) # (N_query_a_i * N_query_b_i,) -> (N_query_a_i, N_query_b_i)
                    # video a, i-th cluster, j-th query
                    avg_sdtw_a_to_b = torch.mean(torch.min(sdtw_loss_mat, dim=1)[0])
                    avg_sdtw_b_to_a = torch.mean(torch.min(sdtw_loss_mat, dim=0)[0])
                    sdtw_loss_i = (avg_sdtw_a_to_b + avg_sdtw_b_to_a) / 2.0
                    object_distances[f"object_{i}_distance"] = sdtw_loss_i.detach()
                else:
                    masked_seq_a_i = seq_a_i * visib_a_i.unsqueeze(-1)
                    sum_seq_a_i = torch.sum(masked_seq_a_i, dim=0) # (N_frame, D)
                    count_seq_a_i = torch.sum(visib_a_i, dim=0).unsqueeze(-1) # (N_frame, 1)
                    avg_seq_a_i = (sum_seq_a_i / (count_seq_a_i + 1e-6)).unsqueeze(0) # (1, N_frame, D)
                    masked_seq_b_i = seq_b_i * visib_b_i.unsqueeze(-1)
                    sum_seq_b_i = torch.sum(masked_seq_b_i, dim=0) # (N_frame, D)
                    count_seq_b_i = torch.sum(visib_b_i, dim=0).unsqueeze(-1) # (N_frame, 1)
                    avg_seq_b_i = (sum_seq_b_i / (count_seq_b_i + 1e-6)).unsqueeze(0) # (1, N_frame, D)
                    sdtw_loss_i = sdtw_cuda(avg_seq_a_i, avg_seq_b_i) # (1, N_frame, D) vs (1, N_frame, D) -> (1,)
                    object_distances[f"object_{i}_distance"] = sdtw_loss_i.detach()
            
            # Average Per-Cluster SoftDTW Loss
            if object_distances:
                all_distances = list(object_distances.values())
                sdtw_loss = torch.mean(torch.stack(all_distances))
                # print("--- Per-Object Distances (for debugging) ---")
                # for obj_id, sdtw_loss_i in object_distances.items():
                #     print(f"{obj_id}: {sdtw_loss_i.item():.4f}")
            else:
                print("No Common Clusters between the two videos.")
                sdtw_loss = torch.tensor(1e6) # penalize this case
            losses.append(sdtw_loss.item() / N_frame_overlap) # normalization (divided by the length of overlapping sequence)

        min_loss = min(losses)
        min_loss_idx = losses.index(min_loss)
        optimal_time_offset = offsets_to_test[min_loss_idx]
        timeoffset_dict[cam_id] = optimal_time_offset.item()
        fpsratio_dict[cam_id] = fps_ratio.item()

    timeoffset_dict[10] = timeoffset_gt_dict[10]

    time_offset_list = [value for key, value in sorted(timeoffset_dict.items(), key=lambda item: item[0])]
    fps_ratio_list = [value for key, value in sorted(fpsratio_dict.items(), key=lambda item: item[0])]  
    
    for cam_id in cam_id_sorted:
        if cam_id == 10: # Print only train cameras
            continue
        print(f"Cam ID: {cam_id:02d} || estimated time-offset: {timeoffset_dict[cam_id]} || GT time-offset: {timeoffset_gt_dict[cam_id]} || time-offset error: {timeoffset_dict[cam_id] - timeoffset_gt_dict[cam_id]}")
        # print(f"Cam ID: {cam_id} || estimated fps-ratio: {fpsratio_dict[cam_id]} || GT fps-ratio: {fpsratio_gt_dict[cam_id]} || time-offset error: {fpsratio_dict[cam_id] - fpsratio_gt_dict[cam_id]}")
    return time_offset_list, fps_ratio_list

def training_stage2(dataset, opt, pipe, checkpoint, debug_from, args, time_offset_list, fps_ratio_list, tb_writer=None):
    first_iter = 0

    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    gaussians = GaussianModel(dataset.deform_spatial_scale, dataset.sh_degree, dataset.duration, dataset.time_interval, dataset.time_pad, 
                              interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type, 
                              time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, kernel_size=dataset.kernel_size)
    # scene = Scene(dataset, gaussians, use_timepad=True)
    '''
    cam 0~9/11~15: train cameras
    cam 10: test camera
    '''
    scene = Scene(dataset, gaussians, time_offset_list, fps_ratio_list, use_timepad=False)
    
    # visualize_scene(gaussians)
    # return
    
    gaussians.training_setup(opt)
    # dataset.duration = -1 (@ arguments/__init__.py)
    # args.duration = 50 (@ configs/techni/Birthday.json)
    # -> args.duration = -1 (@ train.py)
    args.duration = dataset.duration
    
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    saving_iterations = args.save_iterations
    testing_iterations = args.test_iterations
    checkpoint_iterations = args.checkpoint_iterations
        
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # scene.set_sampling_len(dataset.start_duration, sample_every=dataset.sample_every)
    # expanded = gaussians.expand_duration(dataset.start_duration)
    # sample_len = dataset.start_duration
    # g_sample_len = dataset.start_duration

    scene.set_sampling_len(dataset.end_timestamp - dataset.start_timestamp - 1, sample_every=dataset.sample_every)

    need_extract = True
    mark_last = False
    mark_extract = False

    viewpoint_stack = None
    train_images = None
    prune_inv = False
    e_count = args.extract_every
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm.tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                
                R, T, fx, fy, time_offset, fps_ratio = gaussians.get_camera_params(custom_cam.camera_id)
                custom_cam.update(R, T)
                custom_cam.FovX = focal2fov(fx, custom_cam.image_width)
                custom_cam.FovY = focal2fov(fy, custom_cam.image_height)
                timestamp_0th_frame = custom_cam.timestamp / fps_ratio + time_offset # from i-th video time-axis to 0-th video time-axis
                timestamp_0th_sec = timestamp_0th_frame / gaussians.fps_0 # from 0-th video time-axis [frame] to 0-th video time-axis [second]
                # fps_i = args.fps_0 * fps_ratio
                # timestamp_0th_sec = custom_cam.timestamp / fps_i + time_offset / args.fps_0
                custom_cam.timestamp = math.ceil((timestamp_0th_sec - gaussians.global_start_time) * gaussians.fps_0) # from 0-th video time-axis [second] to global time-axis [frame]
                
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, far=dataset.far)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.add_param(opt, args, iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera (shuffle=True)
        if not viewpoint_stack:
            viewpoint_stack, train_images = scene.getTrainCameras(return_as='generator', shuffle=True)
            viewpoint_stack = viewpoint_stack.copy()

            if iteration > opt.prune_invisible_interval:
                prune_inv = True
        
        viewpoint_cam_temp = viewpoint_stack.pop(0)
        viewpoint_cam = copy.deepcopy(viewpoint_cam_temp)
        gt_image = next(train_images).cuda()

        # Before aligning timestamp,
        # scene.train_cameras[1.0][i].timestamp and viewpoint_stack[i].timestamp: range [0, scene.sample_len=50)

        if mark_last:
            if viewpoint_cam.timestamp >= scene.sample_len - gaussians.interval:
                mark_extract = True
                mark_last = False

        R, T, fx, fy, time_offset, fps_ratio = gaussians.get_camera_params(viewpoint_cam.camera_id)
        viewpoint_cam.update(R, T)
        viewpoint_cam.FovX = focal2fov(fx, viewpoint_cam.image_width)
        viewpoint_cam.FovY = focal2fov(fy, viewpoint_cam.image_height)
        timestamp_0th_frame = viewpoint_cam.timestamp / fps_ratio + time_offset # from i-th video time-axis to 0-th video time-axis
        timestamp_0th_sec = timestamp_0th_frame / gaussians.fps_0 # from 0-th video time-axis [frame] to 0-th video time-axis [second]
        # fps_i = args.fps_0 * fps_ratio
        # timestamp_0th_sec = viewpoint_cam.timestamp / fps_i + time_offset / args.fps_0
        viewpoint_cam.timestamp = math.ceil((timestamp_0th_sec - gaussians.global_start_time) * gaussians.fps_0) # from 0-th video time-axis [second] to global time-axis [frame]
        
        # After aligning timestamp,
        # viewpoint_cam.timestamp: range [0, gaussians.tot_duration=81]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, near=dataset.near, far=dataset.far)
        image, viewspace_point_tensor, viewspace_point_error_tensor, visibility_filter, radii, depth, flow, acc, idxs = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["viewspace_l1points"], render_pkg["visibility_filter"], \
            render_pkg["radii"], render_pkg["depth"], render_pkg["opticalflow"], render_pkg["acc"], render_pkg["dominent_idxs"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # backtrack register
        if opt.l1_accum:
            l1_errors = (image - gt_image).abs().mean(dim=0)
            ssim_errors = ssim(image, gt_image, reduce=False).mean(dim=0)
            hook_tensor = torch.stack([acc[0], l1_errors, ssim_errors])
            flow_h = flow.register_hook(lambda grad: hook_tensor)
            loss += flow.mean() * 0
                
        # Regularization
        # if opt.static_reg > 0 and iteration > opt.progressive_growing_steps + opt.make_dynamic_interval:
        if opt.static_reg > 0 and iteration > opt.densify_from_iter: # Changed at Sync4DGS
            loss += opt.static_reg * torch.log(gaussians._xyz_disp.norm(dim=-1)+0.001).mean()

        if opt.motion_reg > 0 and iteration > opt.densify_from_iter and gaussians._xyz_motion.shape[0] > 0:
            diff1 = (gaussians._xyz_motion[:, :1] - gaussians._xyz_motion[:, 1:])
            loss += opt.motion_reg * diff1.norm(dim=-1).mean()
            
        if opt.rot_reg > 0 and iteration > opt.densify_from_iter and gaussians._xyz_motion.shape[0] > 0:
            r1 = gaussians._rotation_motion[:, 1:] 
            r2 = gaussians._rotation_motion[:, :-1]
            
            r_i = 1 - (r1 * r2).sum(dim=-1) / r1.norm(dim=-1).clamp_min(1e-6) / r2.norm(dim=-1).clamp_min(1e-6)
            loss += opt.rot_reg * r_i.mean()
            
        loss.backward()
        
        if opt.l1_accum:
            flow_h.remove()
        
        iter_end.record()

        time_shift_left = gaussians.keyframe_num_left * gaussians.interval - (- gaussians.global_start_time * gaussians.fps_0)
        gaussians.mark_error(loss.item(), viewpoint_cam.timestamp + time_shift_left) # Changed at Sync4DGS

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()
            psnr_log = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_log:.{2}f}", "Loss": f"{ema_loss_for_log:.{6}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background), dataset.near, dataset.far)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_path = os.path.join(scene.model_path, "checkpoint")
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                torch.save((gaussians.capture(), iteration), save_path + "/chkpnt" + str(iteration) + ".pth")

            if opt.l1_accum:
                gaussians.mark_prune_stats(radii, viewspace_point_error_tensor)
            
            # Densification
            if iteration < opt.densify_until_iter:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                static_radii = radii[:static_num]
                dynamic_vis_filter = visibility_filter[static_num:]
                dynamic_radii = radii[static_num:]
                
                gaussians.max_radii2D[static_vis_filter] = torch.max(gaussians.max_radii2D[static_vis_filter], static_radii[static_vis_filter])
                gaussians.motion_max_radii2D[dynamic_vis_filter] = torch.max(gaussians.motion_max_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, static_vis_filter, dynamic_vis_filter, static_num)

                if opt.l1_accum:
                    gaussians.add_l1_ssim_stats(viewspace_point_error_tensor, static_vis_filter, dynamic_vis_filter, static_num, viewpoint_cam.timestamp)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold, dynamic_size_threshold = None, None
                    s_max_ssim = opt.s_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    s_l1_thres = opt.s_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    
                    d_max_ssim = opt.d_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    d_l1_thres = opt.d_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 
                                                opt.densify_dgrad_threshold, 
                                                0.01, 0.01, scene.cameras_extent, size_threshold, dynamic_size_threshold,
                                                s_max_ssim=s_max_ssim, s_l1_thres=s_l1_thres, d_max_ssim=d_max_ssim, d_l1_thres=d_l1_thres)
                elif iteration > opt.extract_from_iter and iteration % opt.extracton_interval == 0: # TODO: 왜 ITER 1777, 3554에서 opt.extracton_interval?
                    static_num = gaussians._xyz.shape[0]
                    candidate = gaussians.get_errorneous_timestamp()
                    if not candidate is None and static_num > 0:
                        print(f"[ITER {iteration}] Extracting dynamic points for every opt.extracton_interval...")
                        # gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), candidate, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                        gaussians.extract_dynamic_points_from_static(viewpoint_cam.T.detach().clone().unsqueeze(0), candidate, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile) # max_dur=None
            
            if iteration % (opt.densification_interval*4) == 0 and iteration < opt.densify_until_iter - 3000:
                # gaussians.adjust_temp_opa(max_dur=sample_len)
                gaussians.adjust_temp_opa() # max_dur=None

            # if prune_inv and iteration < opt.iterations - 5000:
            if prune_inv and iteration < opt.iterations and iteration > 3000:
                gaussians.prune_invisible()
                if opt.l1_accum:
                    gaussians.prune_small()
                prune_inv = False
            
            # Optimizer step
            if iteration < opt.iterations:
                # prevent nan grad of dynamic opacity
                if gaussians._opacity_duration_var.shape[0] != 0:
                    if not gaussians._opacity_duration_var.grad is None:
                        gaussians._opacity_duration_var.grad = gaussians._opacity_duration_var.grad.nan_to_num()
                        
                gaussians.optimizer_gs.step()
                gaussians.optimizer_gs.zero_grad(set_to_none=True)

                if iteration > opt.campose_from_iter and iteration % opt.cam_update_interval == 0:
                    print(f"\n[ITER {iteration}] Updating camera params and expanding dynamic GS duration...")
                    gaussians.optimizer_cam.step()
                    gaussians.optimizer_cam.zero_grad(set_to_none=True)
                    if iteration > opt.camtime_from_iter:
                        gaussians.expand_duration_dynGS()
                
                gaussians.prune_nan_points()
                torch.cuda.empty_cache()
            
            if iteration > opt.extract_from_iter and iteration % opt.progressive_growing_steps == opt.make_dynamic_interval and need_extract :
                mark_last = True
                need_extract = False

            # increase duration part -- NO
            if iteration > opt.extract_from_iter and iteration % opt.progressive_growing_steps == 0 and iteration > opt.progressive_growing_steps and ~need_extract:
                e_count += 1
                if e_count >= opt.extract_every:
                    mark_last = True
                    need_extract = True
                    e_count = 0

            # create dynamic points from static points
            if mark_extract:
                print(f"[ITER {iteration}] Extracting dynamic points for every opt.progressive_growing_steps...")
                static_num = gaussians._xyz.shape[0]
                if static_num > 0:
                    static_vis_filter = visibility_filter[:static_num]
                    # gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), viewpoint_cam.timestamp, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                    gaussians.extract_dynamic_points_from_static(viewpoint_cam.T.detach().clone().unsqueeze(0), viewpoint_cam.timestamp, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile) # max_dur=None
                    mark_extract = False

def prepare_output_and_logger(args, dataset_args):
    date = datetime.now().strftime("%m%d_%H%M")
    args.model_path = os.path.join(args.model_path, date)
    dataset_args.model_path = os.path.join(dataset_args.model_path, date)
    print("Optimizing " + args.model_path)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Use Tensorboard for logging")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Use WandB for logging")
    return tb_writer, date

def training_stage2_wrapper(dataset, opt, pipe, args, time_offset_list, fps_ratio_list):
    with wandb.init() as run:
        vars(args).update(wandb.config) # for wandb sweep
        training_stage2(dataset, opt, pipe, args.start_checkpoint, args.debug_from, args, time_offset_list, fps_ratio_list, tb_writer=None)

def training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderArgs, near, far):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    else:
        wandb.log(
            {
                'train_loss_patches/l1_loss': Ll1.item(),
                'train_loss_patches/total_loss': loss.item(),
                'iter_time': elapsed,
            },
            step=iteration,
        )
        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        
        test_viewpoint_stack, test_images = scene.getTestCameras(shuffle=False, return_as='generator',  n_job=1)
        test_viewpoint_stack = test_viewpoint_stack.copy()
        
        train_viewpoint_stack, train_images = scene.getTrainCameras(shuffle=False, return_as='generator', n_job=1)
        train_viewpoint_stack = train_viewpoint_stack.copy()
            
        validation_configs = ({'name': 'test', 'cameras': test_viewpoint_stack, 
                                                'images': test_images}, 
                              {'name': 'train', 'cameras': train_viewpoint_stack, 
                                                 'images': train_images})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                count = 0
                
                for idx, viewpoint_temp in enumerate(config['cameras']):
                    if config['name'] == 'train':
                        sampled_idx_list = [idx % len(train_viewpoint_stack) for idx in range(5, 30, 5)]
                        if not idx in sampled_idx_list:
                            _ = next(config['images'])
                            continue
                    
                    gt_image = next(config['images']).cuda()

                    viewpoint = copy.deepcopy(viewpoint_temp)
                    R, T, fx, fy, time_offset, fps_ratio = scene.gaussians.get_camera_params(viewpoint.camera_id)
                    viewpoint.update(R, T)
                    viewpoint.FovX = focal2fov(fx, viewpoint.image_width)
                    viewpoint.FovY = focal2fov(fy, viewpoint.image_height)
                    timestamp_0th_frame = viewpoint.timestamp / fps_ratio + time_offset # from i-th video time-axis to 0-th video time-axis
                    timestamp_0th_sec = timestamp_0th_frame / scene.gaussians.fps_0 # from 0-th video time-axis [frame] to 0-th video time-axis [second]
                    # fps_i = scene.gaussians.fps_0 * fps_ratio
                    # timestamp_0th_sec = viewpoint.timestamp / fps_i + time_offset / scene.gaussians.fps_0
                    viewpoint.timestamp = math.ceil((timestamp_0th_sec - scene.gaussians.global_start_time) * scene.gaussians.fps_0) # from 0-th video time-axis [second] to global time-axis [frame]

                    rend_pkg = render(viewpoint, scene.gaussians, near=near, far=far, *renderArgs)
                    image = torch.clamp(rend_pkg["render"], 0.0, 1.0).cuda()
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    elif idx < 5:
                        render_key = f"{config['name']}/view_{viewpoint.image_name}/render"
                        wandb.log({render_key: wandb.Image(image)}, step=iteration)
                        if iteration == testing_iterations[0]:
                            gt_key = f"{config['name']}/view_{viewpoint.image_name}/ground_truth"
                            wandb.log({gt_key: wandb.Image(gt_image)}, step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double().detach().item()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().detach().item()
                    count += 1
                    
                    del(rend_pkg)
                    del(image)
                    del(gt_image)
                    del(viewpoint)
                    torch.cuda.empty_cache()
                    
                psnr_test /= count
                l1_test /= count
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                else:
                    wandb.log(
                        {
                            config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                            config['name'] + '/loss_viewpoint - psnr': psnr_test,
                        },
                        step=iteration
                    )
                    if args.wandb_sweep and config['name'] == 'test':
                        wandb.log({'psnr_test': psnr_test}, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/motion_opacity_histogram", scene.gaussians.get_motion_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians._xyz.shape[0]+scene.gaussians._xyz_motion.shape[0], iteration)
        else:
            wandb.log(
                {
                    'scene/opacity_histogram': scene.gaussians.get_opacity,
                    'scene/motion_opacity_histogram': scene.gaussians.get_motion_opacity,
                    'total_points': scene.gaussians._xyz.shape[0]+scene.gaussians._xyz_motion.shape[0],
                },
                step=iteration
            )
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000, 35_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None) # load by gaussians.restore()
    parser.add_argument("--config", type=str, default="None")
    parser.add_argument("--wandb_sweep", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    # args.test_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # incase we provide config file not directly pass to the file
    if os.path.exists(args.config) and args.config != "None":
        print("overload config from " + args.config)
        config = json.load(open(args.config))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config)
    else:
        raise ValueError("config file not exist or not provided")

    # Start GUI server, configure and run training
    while True:
        try:
            network_gui.init(args.ip, args.port)
            break
        except:
            args.port += 1
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    time_offset_list, fps_ratio_list = None, None
    # Stage 1
    time_offset_list, fps_ratio_list = training_stage1(lp.extract(args))

    # # Log
    # tb_writer, date = prepare_output_and_logger(args, lp.extract(args))
    # sweep_config = {
    #     'name': 'camera_param_lr_bayes_sweep_v1',
    #     'method': 'bayes',  # random, grid, bayes
    #     'metric': {'name': 'psnr_test', 'goal': 'maximize'},
    #     'parameters': {
    #         'cam_rotation_lr_sweep': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         },
    #         'cam_translation_lr_sweep': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         },
    #         'cam_logfocal_lr_sweep': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         },
    #         'cam_time_offset_lr_sweep': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         },
    #         'cam_fps_ratio_lr_sweep': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         }
    #     }
    # }

    # # Stage 2
    # if tb_writer is not None:
    #     training_stage2(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from, args, time_offset_list, fps_ratio_list, tb_writer)
    # elif tb_writer is None and sweep_config is not None:
    #     if args.no_wandb:
    #         training_stage2(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from, args, time_offset_list, fps_ratio_list, tb_writer)
    #     elif args.wandb_sweep:
    #         sweep_id = wandb.sweep(sweep=sweep_config, project="sync4dgs", entity="semyu0102-viclab")
    #         wandb.agent(sweep_id, 
    #                     function=lambda: training_stage2_wrapper(lp.extract(args), op.extract(args), pp.extract(args), args, time_offset_list, fps_ratio_list), 
    #                     count=20) # experiment 20 times
    #     else:
    #         wandb.init(project="sync4dgs", name=f"Sync4DGS_exp1_{date}", entity="semyu0102-viclab", config=vars(args))
    #         training_stage2(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from, args, time_offset_list, fps_ratio_list, tb_writer)
        
    # # All done
    # print("\nTraining complete.")
    # if not TENSORBOARD_FOUND:
    #     wandb.finish()
