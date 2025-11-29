"""Cuts multiple clips from a single video using ffmpeg."""
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from subprocess import call
from tqdm import tqdm


def cut_multiple_clips_video_only(
        video_path: str,
        start_times: list,
        end_times: list,
        save_dir: str,
        video_id=None,
        ext=None,
        verbose=False,
    ):
    """Cuts multiple clips from a single video using ffmpeg.

    Args:
        video_path: Path to the video file.
        start_times: List of start times for each clip.
        end_times: List of end times for each clip.
    """
    os.makedirs(save_dir, exist_ok=True)
    assert len(start_times) == len(end_times), \
        'start_times and end_times must have the same length.'

    if video_id is None:
        video_id = os.path.basename(video_path).split(".")[0]
    if ext is None:
        ext = os.path.basename(video_path).split(".")[1]

    item_ids = [
        f"{video_id}_{np.round(s, 1)}_{np.round(e, 1)}" \
            for (s, e) in zip(start_times, end_times)
    ]

    ins = [
        f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,scale=480:-1[v{i}]" \
        for i, (s, e) in enumerate(zip(start_times, end_times))
    ]
    ins = ";".join(ins)
    outs = [
        f"-map [v{i}] {save_dir}/{item_ids[i]}.{ext}" \
        for i in range(len(start_times))
    ]
    outs = " ".join(outs)
    if not verbose:
        suffix = "-loglevel panic"
    else:
        suffix = ""
    command = f"""
        ffmpeg -i {video_path} -filter_complex "{ins}" {outs} -y {suffix}
    """
    call(command, shell=True)


def cut_multiple_clips_audio_and_video(
        video_path: str,
        start_times: list,
        end_times: list,
        save_dir: str,
        video_id=None,
        ext=None,
        verbose=False,
    ):
    """Cuts multiple clips from a single video using ffmpeg.

    Args:
        video_path: Path to the video file.
        start_times: List of start times for each clip.
        end_times: List of end times for each clip.
    """

    if args.verbose:
        print("[:::] Cutting clips from video: ", video_path)
        print("[:::] Number of clips to cut: ", len(start_times))

    os.makedirs(save_dir, exist_ok=True)
    assert len(start_times) == len(end_times), \
        'start_times and end_times must have the same length.'

    if video_id is None:
        video_id = os.path.basename(video_path).split(".")[0]
    if ext is None:
        ext = os.path.basename(video_path).split(".")[1]

    item_ids = [
        f"{video_id}_{np.round(s, 1)}_{np.round(e, 1)}" \
            for (s, e) in zip(start_times, end_times)
    ]

    ins = [
        f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,scale=480:-1[v{i}];"\
        f"[0:a:0]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]" \
        for i, (s, e) in enumerate(zip(start_times, end_times))
    ]
    ins = ";".join(ins)
    outs = [
        f"-map [v{i}] -map [a{i}] {save_dir}/{item_ids[i]}.{ext}" \
        for i in range(len(start_times))
    ]
    outs = " ".join(outs)
    if not verbose:
        suffix = "-loglevel panic"
    else:
        suffix = ""
    command = f"""
        ffmpeg -i {video_path} -filter_complex "{ins}" {outs} -y {suffix}
    """
    call(command, shell=True)




def cut_multiple_clips_audio_and_video_v2(
        video_path: str,
        start_times: list,
        end_times: list,
        save_dir: str,
        video_id=None,
        ext=None,
        verbose=False,
    ):
    """Cuts multiple clips from a single video using ffmpeg.

    Args:
        video_path: Path to the video file.
        start_times: List of start times for each clip.
        end_times: List of end times for each clip.
    """

    if args.verbose:
        print("[:::] Cutting clips from video: ", video_path)
        print("[:::] Number of clips to cut: ", len(start_times))

    os.makedirs(save_dir, exist_ok=True)
    assert len(start_times) == len(end_times), \
        'start_times and end_times must have the same length.'

    if video_id is None:
        video_id = os.path.basename(video_path).split(".")[0]
    if ext is None:
        ext = os.path.basename(video_path).split(".")[1]

    item_ids = [
        f"{video_id}_{np.round(s, 1)}_{np.round(e, 1)}" \
            for (s, e) in zip(start_times, end_times)
    ]

    ins = [
        f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,scale=480:-1[v{i}];"\
        f"[0:a:0]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]" \
        for i, (s, e) in enumerate(zip(start_times, end_times))
    ]
    # ins = ";".join(ins)
    outs = [
        f"-map [v{i}] -map [a{i}] {save_dir}/{item_ids[i]}.{ext}" \
        for i in range(len(start_times))
    ]
    # outs = " ".join(outs)
    if not verbose:
        suffix = "-loglevel panic"
    else:
        suffix = ""

    iterator = tqdm(range(len(start_times)), desc="Cutting clips for {}".format(video_id))
    for i in iterator:
        ins_ = ins[i]
        outs_ = outs[i]
        save_path = f"{save_dir}/{item_ids[i]}.{ext}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            continue
        command = f"""
            ffmpeg -i {video_path} -filter_complex "{ins_}" {outs_} -y {suffix}
        """
        call(command, shell=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--ext", type=str, default="mp4",
    )
    # Arguments for input CSV
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
        "--video_only", action="store_true",
    )
    parser.add_argument(
        "--si", type=int, default=0,
    )
    parser.add_argument(
        "--ei", type=int, default=None,
    )
    args = parser.parse_args()

    if args.sanity:

        # Test without audio
        video_path = "sample_data/folding_paper.mp4"
        start_times = [0, 5, 10]
        end_times = [5, 10, 15]
        cut_multiple_clips_video_only(
            video_path,
            start_times,
            end_times,
            "./sample_data/clips",
            verbose=args.verbose,
            ext=args.ext,
        )

        # Test with audio
        video_path = "sample_data/pouring_water_youtube.mp4"
        start_times = [0, 5, 10]
        end_times = [5, 10, 15]
        cut_multiple_clips_audio_and_video(
            video_path,
            start_times,
            end_times,
            "./sample_data/clips",
            verbose=args.verbose,
            ext=args.ext,
        )

    else:

        # Make cut_dir
        os.makedirs(args.cut_dir, exist_ok=True)
        
        # Load csv
        assert os.path.exists(args.csv), f"CSV file {args.csv} does not exist."
        df = pd.read_csv(args.csv)
        print(">>> Loaded CSV file with shape", df.shape)
        keys = [args.video_id_key, args.start_time_key, args.end_time_key]
        assert set(keys).issubset(df.columns), \
            f"CSV file must contain columns {keys}."
        
        # Filter out videos that don't exist
        df["video_path"] = df[args.video_id_key].apply(
            lambda video_id: join(args.video_dir, f"{video_id}.{args.ext}"),
        )
        df["check_video"] = df["video_path"].apply(exists)
        df = df[df["check_video"]]
        del df["check_video"]
        print(">>> Found videos for", df.shape[0], "rows.")

        si = args.si
        ei = args.ei if args.ei is not None else df.shape[0]
        print("Running from indices", si, "to", ei)
        df = df.iloc[si:ei]

        if args.debug:
            args.verbose = True
        ext = args.ext

        # Iterate over each video
        video_paths = df["video_path"].unique()
        # iterator = tqdm(range(len(video_paths)), desc="Cutting clips")
        print("Number of unique videos:", len(video_paths))
        for i in range(len(video_paths)):
            video_path = video_paths[i]

            # Find rows corresponding to this video
            df_video = df[df["video_path"] == video_path]
            print("Number of clips to cut from video", video_path, ":", df_video.shape[0])
            start_times = df_video[args.start_time_key].values
            end_times = df_video[args.end_time_key].values
            video_id = df_video[args.video_id_key].values[0]

            """
            # Cut to MAXLEN clips per video
            MAX_LEN = 10
            start_times_batches = np.array_split(start_times, MAX_LEN)
            end_times_batches = np.array_split(end_times, MAX_LEN)
            for start_times_, end_times_ in zip(start_times_batches, end_times_batches):
                if args.video_only:
                    cut_multiple_clips_video_only(
                        video_path,
                        start_times_,
                        end_times_,
                        args.cut_dir,
                        video_id=video_id,
                        ext=ext,
                        verbose=args.verbose,
                        # verbose=True,
                    )
                else:
                    cut_multiple_clips_audio_and_video_v2(
                        video_path,
                        start_times_,
                        end_times_,
                        args.cut_dir,
                        video_id=video_id,
                        ext=ext,
                        verbose=args.verbose,
                        # verbose=True,
                    )
            """
            # """
            # Cut videos
            if args.video_only:
                cut_multiple_clips_video_only(
                    video_path,
                    start_times,
                    end_times,
                    args.cut_dir,
                    video_id=video_id,
                    ext=ext,
                    verbose=args.verbose,
                )
            else:
                cut_multiple_clips_audio_and_video_v2(
                    video_path,
                    start_times,
                    end_times,
                    args.cut_dir,
                    video_id=video_id,
                    ext=ext,
                    verbose=args.verbose,
                )
            # """
            
            if args.debug:
                break
