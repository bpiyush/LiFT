import os
import sys
from glob import glob 
from collections import defaultdict

import json
from hydra import compose, initialize
from omegaconf import OmegaConf
import PIL, PIL.Image
import decord

from adapt4change.utils.speednet import *
import shared.utils as su

classes_selected = [
    "skipping rope",
    "gymnastics tumbling",
    "somersaulting",
    "cartwheeling",
    "trampolines bouncing",
    "swinging on something",
    "vault",
    "deadlifting",
    "clean and jerk",
    "diving cliff",
]

SAVE_DIR = "/scratch/shared/beegfs/piyush/datasets/SpeedyKinetics/clips"
os.makedirs(SAVE_DIR, exist_ok=True)

def check_all_files_exist(save_paths):
    for path in save_paths:
        if not os.path.exists(path):
            return False
    return True

def save_clips_for_single_video(video_path, show=False):
    """
    Note that I randomly sample 3s clips out of the original video.
    """
    video_id = os.path.basename(video_path).split(".mp4")[0]
    save_paths = [
        f"{SAVE_DIR}/{video_id}-normal.mp4",
        f"{SAVE_DIR}/{video_id}-spedup.mp4",
        f"{SAVE_DIR}/{video_id}-slowdn.mp4",
    ]
    if check_all_files_exist(save_paths):
        return

    try:
        vr = decord.VideoReader(video_path)
    except Exception as e:
        print(f"Error opening video {video_path}: {e}")
        return

    total_frames = len(vr)
    fps = vr.get_avg_fps()

    # Initialize sampler for a video 
    sampler = FrameIndexSampler(total_frames=total_frames)
    
    # Sample clips 
    clip_duration = 3.
    T = int(clip_duration * fps)
    start_frame = sampler.get_valid_start_frame(T)

    # Get all clip indices
    normal_indices, sped_up_indices, slowed_down_indices = sampler.sample_all_clip_indices(start_frame, T)

    try:
        frames_normal = [PIL.Image.fromarray(f) for f in vr.get_batch(normal_indices).asnumpy()]
        frames_spedup = [PIL.Image.fromarray(f) for f in vr.get_batch(sped_up_indices).asnumpy()]
        frames_slowdn = [PIL.Image.fromarray(f) for f in vr.get_batch(slowed_down_indices).asnumpy()]
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return

    su.io.save_video(frames_normal, save_paths[0], fps=vr.get_avg_fps())
    su.io.save_video(frames_spedup, save_paths[1], fps=vr.get_avg_fps())
    su.io.save_video(frames_slowdn, save_paths[2], fps=vr.get_avg_fps())
    if show:
        su.visualize.show_grid_of_videos(
            files=save_paths,
            labels=["Normal", "Sped up", "Slowed down"],
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    args = parser.parse_args()

    data_dir = "/datasets/KineticsClean/"
    verbose = True
    total_train = []
    total_valid = []
    for c in classes_selected:
        files_train = glob(f"{data_dir}/train_split/{c}/*.mp4")
        files_valid = glob(f"{data_dir}/val_split/{c}/*.mp4")
        if verbose:
            print(c)
            print("Train videos: ", len(files_train))
            print("Valid videos: ", len(files_valid))
            print("-" * 80)
        total_train.extend(files_train)
        total_valid.extend(files_valid)
    print("Total train files: ", len(total_train))
    print("Total valid files: ", len(total_valid))
    
    files = total_train + total_valid
    print(f"Total files: {len(files)}")
    print(f"Start index: {args.start_index}")
    print(f"End index: {args.end_index}")
    files = files[args.start_index:args.end_index]
    print(f"Total files to process: {len(files)}")
    
    parallelize = True
    if not parallelize:
        for file in su.log.tqdm_iterator(files, desc="Processing files"):
            save_clips_for_single_video(file)
    else:
        from joblib import Parallel, delayed
        iterator = su.log.tqdm_iterator(files, desc="Processing files")
        Parallel(n_jobs=16)(delayed(save_clips_for_single_video)(file) for file in iterator)
