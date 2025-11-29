import os
from glob import glob
import time
import argparse
import numpy as np
import librosa
from decord import VideoReader
import shared.utils.log as log
from os.path import exists, basename
from natsort import natsorted

def resize_video_simple(input_path, output_path, width=480):
    import subprocess
    command = f"""ffmpeg -loglevel quiet -i {input_path} -vf \"scale={width}:-1\" -c:a copy {output_path}"""
    subprocess.call(command, shell=True)

def load_pending_videos(tracker_file):
    """Load list of pending videos from tracker file."""
    if not exists(tracker_file):
        return []
    
    with open(tracker_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_pending_videos(tracker_file, video_paths):
    """Save list of pending videos to tracker file."""
    with open(tracker_file, 'w') as f:
        for path in video_paths:
            f.write(f"{path}\n")

def remove_completed_video(tracker_file, completed_video):
    """Remove a completed video from the tracker file."""
    pending_videos = load_pending_videos(tracker_file)
    if completed_video in pending_videos:
        pending_videos.remove(completed_video)
        save_pending_videos(tracker_file, pending_videos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="Directory containing videos to downsize."
    )
    parser.add_argument(
        "--ext", type=str, default="mp4",
        help="File extension to search for (default: mp4)."
    )
    parser.add_argument(
        "--remove_old", action="store_true",
        help="Remove original video after downsizing."
    )
    parser.add_argument(
        "--width", type=int, default=480,
        help="Width to resize videos to (default: 480)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run debug checks after downsizing."
    )
    parser.add_argument("--si", type=int, default=None)
    parser.add_argument("--ei", type=int, default=None)
    parser.add_argument(
        "--tracker_file", type=str, default="video_resize_tracker.txt",
        help="Tracker file to keep track of pending videos (default: video_resize_tracker.txt)."
    )
    parser.add_argument(
        "--reset_tracker", action="store_true",
        help="Reset the tracker file and start fresh."
    )
    args = parser.parse_args()

    assert os.path.isdir(args.video_dir), f"Directory {args.video_dir} does not exist."
    
    # Handle tracker file
    if args.reset_tracker and exists(args.tracker_file):
        os.remove(args.tracker_file)
        print(f"> Reset tracker file: {args.tracker_file}")
    
    # Check if we have pending videos from previous run
    pending_videos = load_pending_videos(args.tracker_file)
    
    if pending_videos:
        print(f"> Found {len(pending_videos)} pending videos from previous run")
        ifiles = pending_videos
        ofiles = pending_videos  # In-place replacement
    else:
        # Start fresh - find all videos
        pattern = os.path.join(args.video_dir, f"**/*.{args.ext}")
        ifiles = glob(pattern, recursive=True)
        ifiles = natsorted(ifiles)
        ofiles = ifiles  # In-place replacement
        print("> Number of videos in the directory:", len(ifiles))
        
        # Apply start/end index filtering
        si = args.si if args.si is not None else 0
        ei = args.ei if args.ei is not None else len(ifiles)
        print("> Start index:", si)
        print("> End index:", ei)
        ifiles = ifiles[si:ei]
        ofiles = ofiles[si:ei]
        
        # Save to tracker file for future runs
        save_pending_videos(args.tracker_file, ifiles)
        print(f"> Saved {len(ifiles)} videos to tracker file: {args.tracker_file}")
    
    print("> Number of videos to downsize:", len(ifiles))

    iterator = log.tqdm_iterator(
        range(len(ifiles)), total=len(ifiles), desc="Downsizing videos",
    )
    for i in iterator:
        ifile = ifiles[i]
        ofile = ifile
        assert exists(ifile), f"Video {ifile} does not exist."
        tmp_ofile = ifile + ".tmp.mp4"
        start_time = time.time()
        resize_video_simple(ifile, tmp_ofile, width=args.width)
        end_time = time.time()
        time_taken = end_time - start_time
        desc = f"Time taken {time_taken:.2f}s for {basename(ifile)}"
        iterator.set_description(desc)

        if args.debug:
            yold, srold = librosa.load(ifile, offset=1.0, duration=1.0)
            ynew, srnew = librosa.load(tmp_ofile, offset=1.0, duration=1.0)
            assert srold == srnew, "Sampling rate mismatch."
            assert len(yold) == len(ynew), "Length mismatch."
            assert np.allclose(yold, ynew), "Audio mismatch."
            vr = VideoReader(tmp_ofile)
            frames = vr.get_batch(range(0, 10)).asnumpy()
            assert frames.shape[0] == 10, "Length mismatch."
            assert frames.shape[2] == args.width, "Width mismatch."

        # Replace original file
        os.replace(tmp_ofile, ofile)
        
        # Remove completed video from tracker
        remove_completed_video(args.tracker_file, ifile)
        
        if args.remove_old:
            # Already replaced, so nothing to remove
            pass
    
    # Clean up tracker file if all videos are done
    if not load_pending_videos(args.tracker_file):
        os.remove(args.tracker_file)
        print(f"> All videos completed. Removed tracker file: {args.tracker_file}")
    else:
        remaining = len(load_pending_videos(args.tracker_file))
        print(f"> {remaining} videos remaining. Tracker file preserved: {args.tracker_file}") 