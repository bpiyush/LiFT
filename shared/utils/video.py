import decord
import PIL.Image
import PIL
import numpy as np


def load_frame(video_path, index=0):
    vr = decord.VideoReader(video_path, num_threads=1)
    frame = PIL.Image.fromarray(vr[index].asnumpy())
    return frame


def load_first_and_final_frames(video_path):
    vr = decord.VideoReader(video_path, num_threads=1)
    frame_l = PIL.Image.fromarray(vr[0].asnumpy())
    frame_r = PIL.Image.fromarray(vr[-1].asnumpy())
    return [frame_l, frame_r]


def load_frames_linspace(video_path, st=None, et=None, n=8, num_threads=1, reverse=False, **vr_args):
    decord.bridge.set_bridge('native')

    try:
        vr = decord.VideoReader(video_path, num_threads=num_threads, **vr_args)
    except Exception as e:
        print("Error loading video:", e, "for video:", video_path)
        # Return blank frames
        return [PIL.Image.new("RGB", (480, 256)) for _ in range(n)]

    fps = vr.get_avg_fps()
    if st is None:
        sf = 0
    else:
        sf = max(int(st * fps), 0)
    if et is None:
        ef = len(vr) - 1
    else:
        ef = min(int(et * fps), len(vr) - 1)
        ef = max(ef, sf)
    if n == -1:
        indices = np.arange(sf, ef + 1)
    else:
        indices = np.linspace(sf, ef, n, endpoint=True).astype(int)

    try:
        frames = [PIL.Image.fromarray(vr[i].asnumpy()) for i in indices]
    except Exception as e:
        print("Error loading frames:", e, "for video:", video_path)
        # Return blank frames
        frames = [PIL.Image.new("RGB", (480, 256)) for _ in range(n)]
    
    if reverse:
        frames = frames[::-1]

    # Close the video reader
    del vr

    return frames


def load_frames_linspace_with_first_and_last(video_path, n=8):
    """Loads n frames from a video, including the first and last frames."""
    assert n > 1, "n should be greater than 1"
    vr = decord.VideoReader(video_path, num_threads=1)
    indices = np.linspace(0, len(vr) - 1, n - 2).astype(int)
    frames = [PIL.Image.fromarray(vr[0].asnumpy())]
    frames += [PIL.Image.fromarray(x) for x in vr.get_batch(indices).asnumpy()]
    frames += [PIL.Image.fromarray(vr[-1].asnumpy())]
    return frames


def get_duration(path, return_fps=False):
    vr = decord.VideoReader(path, num_threads=1)
    if not return_fps:
        return len(vr) / vr.get_avg_fps()
    else:
        return len(vr) / vr.get_avg_fps(), vr.get_avg_fps()


def load_frames_at_timestamps(video_path, timestamps):
    """
    Loads frames at given timestamps from a video.

    Args:
        video_path (str): Path to the video file.
        timestamps (list): List of timestamps at which to load frames.
    """
    vr = decord.VideoReader(video_path, num_threads=1)
    duration = len(vr) / vr.get_avg_fps()
    assert max(timestamps) <= duration, \
        "Timestamps should be within the duration of the video."
    indices = [int(t * vr.get_avg_fps()) for t in timestamps]
    frames = [PIL.Image.fromarray(vr[i].asnumpy()) for i in indices]
    return frames



import ffmpeg
import os
from pathlib import Path

def cut_video(video_path, start_time, end_time, save_path):
    """
    Cut a video clip from a source video file.
    
    Args:
        video_path (str): Path to the input video file
        start_time (str or float): Start time in seconds (float) or time format (str like "00:01:30")
        end_time (str or float): End time in seconds (float) or time format (str like "00:02:45")
        save_path (str): Path where the cut video will be saved
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        FileNotFoundError: If input video file doesn't exist
        Exception: For other FFmpeg-related errors
    """
    
    # Check if input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Calculate duration if times are provided as numbers
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            duration = end_time - start_time
        else:
            # Let FFmpeg handle time format strings directly
            duration = None
        
        # Build FFmpeg command
        input_stream = ffmpeg.input(video_path)
        
        if duration is not None:
            # Use start time and duration
            output_stream = input_stream.video.filter('trim', start=start_time, duration=duration)
            audio_stream = input_stream.audio.filter('atrim', start=start_time, duration=duration)
        else:
            # Use start and end time strings
            output_stream = input_stream.video.filter('trim', start=start_time, end=end_time)
            audio_stream = input_stream.audio.filter('atrim', start=start_time, end=end_time)
        
        # Combine video and audio streams
        output = ffmpeg.output(
            output_stream, 
            audio_stream, 
            save_path,
            vcodec='copy',  # Copy video codec to maintain quality and speed
            acodec='copy'   # Copy audio codec to maintain quality and speed
        )
        
        # Run the FFmpeg command (overwrite output file if it exists)
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        
        print(f"Video successfully cut and saved to: {save_path}")
        return True
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error occurred: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Alternative simpler version using subprocess (if ffmpeg-python is not available)
import subprocess

def cut_video_subprocess(video_path, start_time, end_time, save_path):
    """
    Cut a video clip using subprocess to call FFmpeg directly.
    
    Args:
        video_path (str): Path to the input video file
        start_time (str): Start time (e.g., "00:01:30" or "90")
        end_time (str): End time (e.g., "00:02:45" or "165")
        save_path (str): Path where the cut video will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',  # Copy streams without re-encoding
            '-y',  # Overwrite output file
            save_path
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video successfully cut and saved to: {save_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example 1: Using time in seconds
    cut_video(
        video_path="sample_data/folding_paper.mp4",
        start_time=3,      # 30 seconds
        end_time=9,        # 90 seconds
        save_path="output_clip.mp4"
    )
