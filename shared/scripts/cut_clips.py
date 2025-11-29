"""
Cut video clips from downloaded videos.

Example.
D=/work/piyush/from_nfs2/datasets/Charades
video_dir=$D/Charades_v1_480/

EPIC

S=/datasets/EpicKitchens-100/
D=/work/piyush/from_nfs2/datasets/EPIC-Kitchens-100/cut_clips
csv=$D/../epic-kitchens-100-annotations/EPIC_100_train_with_id.csv
python shared/scripts/cut_clips.py --csv $csv --video_id_key path_id --start_time_key start_sec --end_time_key stop_sec --video_dir $S/ --cut_dir $D/  --ext MP4
"""
import os
from os.path import join, exists
from subprocess import call
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import shared.utils.io as io
import shared.utils.log as log
from video_language.datasets.charades import get_paths, load_main_csv


def time_float_to_str(time_in_seconds):
    import datetime

    # Calculate hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(time_in_seconds, 3600)
    minutes, seconds_with_ms = divmod(remainder, 60)
    seconds, milliseconds = divmod(int(seconds_with_ms * 1000), 1000)

    # Create a timedelta object
    time_delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    # Format the time as HH:MM:SS.mmm
    formatted_time = str(time_delta)
    
    return formatted_time


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to CSV file containing video IDs and timestamps",
    )
    parser.add_argument(
        "--video_id_key", type=str, default="video_id",
    )
    parser.add_argument(
        "--start_time_key", type=str, default="start_time",
    )
    parser.add_argument(
        "--end_time_key", type=str, default="end_time",
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="Path to directory containing downloaded videos",
    )
    parser.add_argument(
        "--cut_dir", type=str, required=True,
        help="Path to directory where cut videos will be saved",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite existing cut videos",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    parser.add_argument(
        "--no_round_times", action="store_true",
        help="Whether to round start and end times to nearest second in filenames",
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    parser.add_argument(
        "--ext", type=str, default="mp4",
    )
    parser.add_argument(
        "--si", type=int, default=0,
    )
    parser.add_argument(
        "--ei", type=int, default=None,
    )
    parser.add_argument(
        "--filter_csv", type=str, default=None, required=False,
    )
    parser.add_argument(
        "--filter_key", type=str, default=None, required=False,
    )
    args = parser.parse_args()
    
    # Make cut_dir
    os.makedirs(args.cut_dir, exist_ok=True)
    
    # Load csv
    assert os.path.exists(args.csv), f"CSV file {args.csv} does not exist."
    df = pd.read_csv(args.csv)
    print(">>> Loaded CSV file with shape", df.shape)
    assert {args.video_id_key, args.start_time_key, args.end_time_key}.issubset(df.columns), \
        f"CSV file must contain columns {args.video_id_key}, {args.start_time_key}, and {args.end_time_key}."

    # Filter CSV
    if args.filter_csv is not None:
        path = args.filter_csv
        assert os.path.exists(path), f"CSV file {path} does not exist."

        key = args.filter_key
        df_filter = pd.read_csv(path)
        assert key in df_filter.columns, f"CSV file must contain column {key}."

        # Only keep the rows in df that match on key with df_filter
        keep_values = df_filter[key].unique()

        df = df[df[key].isin(keep_values)]
        print(">>> Filtered CSV file with shape", df.shape)
    
    # Filter out videos that don't exist
    df["video_path"] = df[args.video_id_key].apply(
        lambda video_id: join(args.video_dir, f"{video_id}.{args.ext}"),
    )
    df["check_video"] = df["video_path"].apply(exists)
    df = df[df["check_video"]]
    del df["check_video"]
    print(">>> Found videos for", df.shape[0], "rows.")

    if len(df) == 0:
        print(">>> No videos to cut.")
        exit()


    si = args.si
    ei = args.ei if args.ei is not None else len(df)
    df = df.iloc[si:ei]
    print("Start index:", si, "End index:", ei)

    # Custom filter
    # df = df[df.split == "validation"]
    # print(">>> Filtered videos for", df.shape[0], "rows.")

    if args.debug:
        args.verbose = True


    # Cut videos
    ext = args.ext
    iterator = tqdm(range(len(df)), desc="Cutting clips")
    for i in iterator:

        row = df.iloc[i].to_dict()
        f = row["video_path"]
        v, s, e = row[args.video_id_key], row[args.start_time_key], row[args.end_time_key]
        s = float(s)
        e = float(e)

        if args.no_round_times:
            clip_filename = f"{v}_{s}_{e}.{ext}"
        else:
            clip_filename = f"{v}_{np.round(s, 1)}_{np.round(e, 1)}.{ext}"
        clip_filepath = join(args.cut_dir, clip_filename)
        os.makedirs(os.path.dirname(clip_filepath), exist_ok=True)

        if os.path.exists(clip_filepath) and not args.overwrite:
            continue
        
        # bring s in HH:MM:SS.mmm format with milliseconds
        s = time_float_to_str(s)
        e = time_float_to_str(e)
        # # bring s in HH:MM:SS. format
        # s = time.strftime("%H:%M:%S", time.gmtime(s))
        # e = time.strftime("%H:%M:%S", time.gmtime(e))

        # ffmpeg code
        # ffmpeg_source = "/users/piyush/install/ffmpeg-06092024/ffmpeg-7.0.2-i686-static/ffmpeg"
        ffmpeg_source = " /users/piyush/install/ffmpeg/ffmpeg-7.0.2-i686-static/ffmpeg"
        # print("FFMpeg version: ", call(f"{ffmpeg_source} -version", shell=True))
        # use ffmpeg to cut the clip + change spatial resolution to have max height
        # NOTE: also changes spatial resolution to have max width as 480
        command = f"{ffmpeg_source} -i {f} -ss {s} -to {e} -strict -2 -c:v libx264 "\
            f"-pix_fmt yuv420p -c:a copy"\
            " -vf 'scale=480:-1' "\
            f"{clip_filepath} "\
            f"-y -format {ext}"
        if not args.verbose:
            command += " -loglevel quiet"
        else:
            print(">>> Cutting clip", clip_filepath)
        call(command, shell=True)

        if args.debug:
            print(command)
            break
    
    print(">>> Number of cut files:", len(os.listdir(args.cut_dir)))
