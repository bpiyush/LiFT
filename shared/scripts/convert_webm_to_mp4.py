"""
Script to conver webm to mp4

Example:
video_dir=/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/videos/
out_dir=/scratch/shared/nfs2/piyush/datasets/SSv2/videos/
python shared/scripts/convert_webm_to_mp4.py --video_dir $video_dir --out_dir $out_dir
"""

import os
from subprocess import call
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    ext = ".webm"
    
    in_files = glob(os.path.join(args.video_dir, f"*/*{ext}"))
    out_files = [f.replace(f"{ext}", ".mp4") for f in in_files]
    out_files = [f.replace(args.video_dir, args.out_dir) for f in out_files]

    print("> Set to convert", len(in_files), f"files from {ext} to mp4.")
    iterator = tqdm(range(len(in_files)), desc="Converting videos")
    for i in iterator:
        in_file = in_files[i]
        out_file = out_files[i]
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        if not os.path.exists(out_file):
            command = f"ffmpeg -i {in_file} -c:v copy -c:a copy -strict -2 {out_file} -loglevel quiet"
            call(command, shell=True)
        else:
            print(f"Skipping {in_file} as {out_file} already exists.")

        # os.remove(in_file)
        break
    print("> Done converting.")