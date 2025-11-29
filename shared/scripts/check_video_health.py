"""Checks videos in a CSV for health."""
import os
import sys
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from torchcodec.decoders import SimpleVideoDecoder
import decord

import shared.utils as su


def get_video_width(path):
    try:
        return SimpleVideoDecoder(path).metadata.width 
    except:
        return -1


def check_decord_videoreader(path):
    try:
        vr = decord.VideoReader(path)
        return True
    except:
        return False


def check_decord_random_frame(path):
    try:
        vr = decord.VideoReader(path)
        i = np.random.randint(len(vr))
        frame = vr[i]
        return True
    except:
        return False


if __name__ == "__main__":
    # Configure video_dir and csv_path
    data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap"
    video_dir = f"{data_dir}/cut_full_scale"
    csv_path = f"{data_dir}/metadata/"\
        "cleaned_chiral_subset_with_reverse_captions-425K-2025-07-17_18:21:38.csv"
    id_col = "id"
    save_dir = "./outputs/ego4d_video_health"
    os.makedirs(save_dir, exist_ok=True)
    
    health_checks = [
        # "video_exists",
        "video_widths",
        # "decord_videoreader",
        # "decord_random_frame",
    ]
    
    
    assert os.path.isdir(video_dir), f"Video directory {video_dir} does not exist"
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    
    # Load the CSV
    su.log.print_update(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Number of rows in CSV:", len(df))
    
    # Add video_path column
    df['video_path'] = df[id_col].apply(lambda x: f"{video_dir}/{x}.mp4")
    
    
    # 1. Check if the video files exist
    if "video_exists" in health_checks:
        su.log.print_update("Checking if the video files exist")
        iterator = su.log.tqdm_iterator(df.video_path.tolist())
        video_exists = Parallel(n_jobs=-1)(delayed(os.path.exists)(f) for f in iterator)
        print("Fraction of videos that exist: ", np.mean(video_exists))
        ids_with_missing_videos = df.loc[~np.array(video_exists), id_col].tolist()
        np.save(f"{save_dir}/ids_with_missing_videos.npy", ids_with_missing_videos)
    
    # 2. Check if the video widths are valid
    if "video_widths" in health_checks:
        su.log.print_update("Checking if the video widths are valid")
        iterator = su.log.tqdm_iterator(df.video_path.tolist())
        video_widths = Parallel(n_jobs=-1)(delayed(get_video_width)(f) for f in iterator)
        video_widths = np.array(video_widths)
        import ipdb; ipdb.set_trace()
        print("Fraction of videos that are valid: ", np.mean(video_widths != -1))
        ids_with_invalid_widths = df.loc[np.where(video_widths == -1), id_col].tolist()
        np.save(f"{save_dir}/ids_with_invalid_widths.npy", np.array(ids_with_invalid_widths))
    
    # 3. Check if the video files are decodable by decord
    if "decord_videoreader" in health_checks:
        su.log.print_update("Checking if the video files are decodable by decord")
        iterator = su.log.tqdm_iterator(df.video_path.tolist())
        decord_videoreader = Parallel(n_jobs=-1)(delayed(check_decord_videoreader)(f) for f in iterator)
        decord_videoreader = np.array(decord_videoreader)
        import ipdb; ipdb.set_trace()
        print("Fraction of videos that are decodable by decord: ", np.mean(decord_videoreader))
        ids_with_invalid_decord = df.loc[~decord_videoreader, id_col].tolist()
        np.save(f"{save_dir}/ids_with_invalid_decord.npy", np.array(ids_with_invalid_decord))
    
    
    # 4. Check if a random frame can be decoded by decord
    if "decord_random_frame" in health_checks:
        su.log.print_update("Checking if a random frame can be decoded by decord")
        iterator = su.log.tqdm_iterator(df.video_path.tolist())
        decord_random_frame = Parallel(n_jobs=-1)(delayed(check_decord_random_frame)(f) for f in iterator)
        decord_random_frame = np.array(decord_random_frame)
        print("Fraction of videos that have a random frame that can be decoded by decord: ", np.mean(decord_random_frame))
        ids_with_invalid_decord_random_frame = df.loc[~decord_random_frame, id_col].tolist()
        np.save(f"{save_dir}/ids_with_invalid_decord_random_frame.npy", np.array(ids_with_invalid_decord_random_frame))
