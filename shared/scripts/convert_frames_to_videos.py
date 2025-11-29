"""
Converts frames to videos.

input_dir=/scratch/shared/beegfs/piyush/datasets/Jester/images/
output_dir=/scratch/shared/beegfs/piyush/datasets/Jester/videos/
python shared/scripts/convert_frames_to_videos.py --input_dir $input_dir --output_dir $output_dir
"""
import os
import sys
import decord
import moviepy.editor as mpy
from glob import glob
from moviepy.editor import ImageSequenceClip

import shared.utils as su


def create_video(image_paths, output_file, fps=12):
    """
    Convert a list of image files into an MP4 video
    
    Parameters:
    - image_paths: List of image file paths (sorted in desired order)
    - output_file: Output filename (e.g., 'output.mp4')
    - fps: Frames per second (default 12)
    """
    clip = ImageSequenceClip(image_paths, fps=fps, load_images=True)
    clip.write_videofile(output_file, codec='libx264', logger=None)



if __name__ == "__main__":
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--extension", type=str, default="jpg")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of all frame folders
    print("Getting list of all frame folders...")
    folders = os.listdir(args.input_dir)

    # Run the process for each folder
    iterator = su.log.tqdm_iterator(folders, desc="Converting frames to videos")
    for folder in iterator:

        # Get output path
        save_path = os.path.join(args.output_dir, f"{folder}.mp4")
        if os.path.exists(save_path) and not args.overwrite:
            continue

        # Get list of frames
        frame_paths = glob(
            os.path.join(args.input_dir, folder, f"*.{args.extension}"),
        )

        # Save it as a video
        create_video(frame_paths, save_path, fps=args.fps)

        if args.debug:
            print("Debugging mode. Exiting after processing one folder.")
            break
