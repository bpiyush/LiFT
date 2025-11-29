"""Downsize videos preserving aspect resolution."""
import torch
import torchvision
from decord import VideoReader
from glob import glob
import os
from os.path import join, basename, exists
import subprocess
import numpy as np
import pandas as pd
import subprocess
import ffmpeg
import time
import librosa
from moviepy.editor import VideoFileClip

import shared.utils.log as log
import shared.utils.io as io


# def downsize(input_path, output_path, width=480, height=None, maintain_aspect_ratio=True):
#     """Downsizes a given video."""

#     # Check if the video exists
#     assert exists(input_path), f"Video {input_path} does not exist."
    
#     # Define ffmpeg command to downsize video with width=480 maintaining aspect ratio
#     # And save it at output_path
#     if maintain_aspect_ratio:
#         assert height is None, "Cannot specify height when maintaining aspect ratio."
#         height = -1

#     (
#         ffmpeg
#         .input(input_path)
#         .output(output_path, preset="ultrafast", vf=f"scale={width}:{height}", loglevel="quiet")
#         .run()
#     )

def resize_video_maintain_aspect_ratio(input_path, output_path):
    (
        ffmpeg
        .input(input_path)
        .filter("scale", w=480, h=-2)
        .output(output_path, crf=18, preset="ultrafast", loglevel="quiet")
        .run()
    )


def resize_video_maintain_aspect_ratio_faster(input_path, output_path, width=480):
    (
        ffmpeg
        .input(input_path)
        .filter('scale', width, -1)
        .output(output_path, vcodec='h264_nvenc', preset='fast', pix_fmt='yuv420p')
        .overwrite_output()
        .run(capture_stdout=True)
    )


def resize_video_maintain_aspect_ratio_vanilla(input_path, output_path, width=480):
    (
        ffmpeg
        .input(input_path)
        .filter('scale', width, -1)
        .output(output_path, c="copy")
        .overwrite_output()
        .run(capture_stdout=True)
    )


def resize_video_moviepy(input_path, output_path, width=480):
    # Load the input video
    video = VideoFileClip(input_path)
    
    # Resize the video
    video_resized = video.resize(width=width)
    
    # Save the resized video
    video_resized.write_videofile(output_path)



def resize_video_simple(input_path, output_path, width=480):
    command = f"""ffmpeg -loglevel quiet -i {input_path} -vf "scale={width}:-1" -c:a copy {output_path} -y"""
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to csv file containing in/out video paths."
    )
    parser.add_argument(
        "--in_colname", type=str, default="input",
        help="column name of input videos.",
    )
    parser.add_argument(
        "--out_colname", type=str, default="output",
        help="column name of output videos.",
    )
    parser.add_argument(
        "--remove_old", action="store_true",
    )
    parser.add_argument(
        "--width", type=int, default=480,
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    parser.add_argument(
        "--si", type=int, default=None,
        help="Start index.",
    )
    parser.add_argument(
        "--ei", type=int, default=None,
        help="End index.",
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    print("Width:", args.width)
    assert exists(args.csv), f"File {args.csv} does not exist."
    df = pd.read_csv(args.csv)
    print("> Number of videos:", len(df))
    
    si = args.si if args.si is not None else 0
    ei = args.ei if args.ei is not None else len(df)
    print("> Start index:", si)
    print("> End index:", ei)
    df = df.iloc[si:ei]
    print("> Number of videos to downsize:", len(df))
    
    ifiles = df[args.in_colname].tolist()
    ofiles = df[args.out_colname].tolist()
    assert len(ifiles) == len(ofiles), \
        "Number of input and output videos must be the same."

    iterator = log.tqdm_iterator(
        range(len(ifiles)), total=len(ifiles), desc="Downsizing videos",
    )
    for i in iterator:
        ifile, ofile = ifiles[i], ofiles[i]
        
        # If ofile == ifile (i.e., edit the same file),
        # then we need to operate on a temporary file
        # which will then be moved to the original file
        replace = ofile == ifile
        if replace:
            ofile_actual = ofile
            ofile = ofile.replace(".mp4", "_temp.mp4")

        # Check if the video exists
        assert exists(ifile), f"Video {ifile} does not exist."

        # If output file already exists, skip
        if exists(ofile) and not args.overwrite:
            continue

        # Make sure output directory exists
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        # resize
        start_time = time.time()
        resize_video_simple(ifile, ofile, width=args.width)
        end_time = time.time()
        time_taken = end_time - start_time
        desc = "Time taken {:.2f}s for {}".format(time_taken, basename(ifile))
        iterator.set_description(desc)
        
        # If replace, move the temporary file to the original file
        if replace:
            if exists(ofile):
                os.rename(ofile, ofile_actual)
            ofile = ofile_actual
        
        if args.debug:
            yold, srold = librosa.load(ifile, offset=1.0, duration=1.0)
            ynew, srnew = librosa.load(ofile, offset=1.0, duration=1.0)
            assert srold == srnew, "Sampling rate mismatch."
            assert len(yold) == len(ynew), "Length mismatch."
            assert np.allclose(yold, ynew), "Audio mismatch."
            
            # Try loading the new video file
            vr = VideoReader(ofile)
            frames = vr.get_batch(range(0, 10)).asnumpy()
            assert frames.shape[0] == 10, "Length mismatch."
            assert frames.shape[2] == 480, "Width mismatch."
        
        # If remove_old, remove old video
        if args.remove_old:
            os.remove(ifile)
