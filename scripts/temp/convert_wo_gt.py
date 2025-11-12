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
import logging
from argparse import ArgumentParser
import shutil

def getcolmap(args, folder, offset):
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    location_folder = os.path.join(folder, f"colmap_{offset:05d}")

    if not args.skip_matching:
        os.makedirs(location_folder + "/distorted/sparse", exist_ok=True)

        ## Feature extraction
        feat_extracton_cmd = colmap_command + " feature_extractor "\
            "--database_path " + location_folder + "/distorted/database.db \
            --image_path " + location_folder + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + args.camera + " \
            --SiftExtraction.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + location_folder + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ### Bundle adjustment
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        # mapper_cmd = (colmap_command + " mapper \
        #     --database_path " + location_folder + "/distorted/database.db \
        #     --image_path "  + location_folder + "/input \
        #     --output_path "  + location_folder + "/distorted/sparse \
        #     --Mapper.ba_global_function_tolerance=0.000001"
        # )
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + location_folder + "/distorted/database.db \
            --image_path "  + location_folder + "/input \
            --output_path "  + location_folder + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001 \
            --Mapper.abs_pose_min_num_inliers 10 \
            --Mapper.init_min_num_inliers 10 \
            --Mapper.tri_ignore_two_view_tracks 0"
        )
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + location_folder + "/input \
        --input_path " + location_folder + "/distorted/sparse/0 \
        --output_path " + location_folder + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(location_folder + "/sparse")
    os.makedirs(location_folder + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(location_folder, "sparse", file)
        destination_file = os.path.join(location_folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")

        # Resize images.
        os.makedirs(location_folder + "/images_2", exist_ok=True)
        os.makedirs(location_folder + "/images_4", exist_ok=True)
        os.makedirs(location_folder + "/images_8", exist_ok=True)
        # Get the list of files in the source directory
        files = os.listdir(location_folder + "/images")
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(location_folder, "images", file)

            destination_file = os.path.join(location_folder, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(location_folder, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(location_folder, "images_8", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

    print("Done.")
