#!/usr/bin/env python3
"""
Video Grid Visualizer (imageio-based with Grid)

Creates a grid layout of multiple videos with visual separators and saves it
as an MP4 or GIF. This version uses the imageio library for robust video
writing, adds customizable grid gaps, and intelligently resizes cells for a
balanced look.
"""

import cv2
import numpy as np
import argparse
import os
import imageio.v2 as imageio # Use imageio.v2 to avoid deprecation warnings

def get_video_info(video_path):
    """Get video information using imageio with a fallback to OpenCV."""
    try:
        with imageio.get_reader(video_path) as reader:
            meta = reader.get_meta_data()
            fps = meta.get('fps', 30)
            duration = meta.get('duration', 0)
            size = meta.get('size', (0, 0))
            if duration == 0 and fps > 0:
                duration = reader.count_frames() / fps
            frame_count = int(duration * fps) if duration and fps else reader.count_frames()
            
            return {
                'duration': duration, 'fps': fps, 'frame_count': frame_count,
                'width': size[0], 'height': size[1]
            }
    except Exception:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            'duration': duration, 'fps': fps, 'frame_count': frame_count,
            'width': width, 'height': height
        }

def resize_frame(frame, target_width, target_height):
    """Resize frame to fit inside target dimensions, maintaining aspect ratio."""
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        new_w = target_width
        new_h = int(new_w / aspect)
    else:
        new_h = target_height
        new_w = int(new_h * aspect)
        
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def create_video_grid(video_paths, n_rows=1, n_cols=None, save_path="output.mp4", 
                      gap=10, grid_color_name="white", max_cell_width=640):
    """
    Create a grid of videos with separators and save as MP4 or GIF.
    """
    if not video_paths:
        raise ValueError("No video paths provided")
    
    if n_cols is None:
        n_cols = len(video_paths)
    
    if n_rows * n_cols < len(video_paths):
        raise ValueError(f"Grid size ({n_rows}x{n_cols}) is too small for {len(video_paths)} videos")

    grid_color = (255, 255, 255) if grid_color_name.lower() == "white" else (0, 0, 0)

    print("Gathering video information...")
    video_infos = [get_video_info(path) for path in video_paths]
    
    # --- Intelligent Resizing and Dimension Calculation ---
    valid_infos = [info for info in video_infos if info['width'] > 0 and info['height'] > 0]
    if not valid_infos:
        raise ValueError("Could not get valid dimensions from any input video.")
        
    avg_aspect_ratio = sum(info['width'] / info['height'] for info in valid_infos) / len(valid_infos)
    
    cell_width = min(max_cell_width, 1920 // n_cols) # Don't let cells get too big
    cell_height = int(cell_width / avg_aspect_ratio)

    output_width = (cell_width * n_cols) + (gap * (n_cols + 1))
    output_height = (cell_height * n_rows) + (gap * (n_rows + 1))
    
    # Ensure dimensions are even, as required by many video codecs
    output_width += output_width % 2
    output_height += output_height % 2
    # ---

    max_duration = max(info['duration'] for info in video_infos if info)
    target_fps = max(info['fps'] for info in video_infos if info and info['fps'])
    if not target_fps or target_fps <= 0:
        target_fps = 30
    
    total_frames = int(max_duration * target_fps)
    
    print(f"Grid: {n_rows}x{n_cols} with {gap}px {grid_color_name} gaps")
    print(f"Calculated Cell Size: {cell_width}x{cell_height}")
    print(f"Final Output Size: {output_width}x{output_height}")
    print(f"Max duration: {max_duration:.2f}s | Target FPS: {target_fps} | Total frames: {total_frames}")

    caps = [cv2.VideoCapture(path) for path in video_paths]
    last_frames = [None] * len(video_paths)

    writer = imageio.get_writer(save_path, fps=target_fps, codec='libx264', macro_block_size=None)

    try:
        for frame_idx in range(total_frames):
            # Initialize the master frame with the grid color
            output_frame = np.full((output_height, output_width, 3), grid_color, dtype=np.uint8)
            
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    last_frames[i] = frame
                else:
                    if last_frames[i] is None:
                        info = video_infos[i]
                        h = info['height'] if info['height'] > 0 else cell_height
                        w = info['width'] if info['width'] > 0 else cell_width
                        last_frames[i] = np.zeros((h, w, 3), dtype=np.uint8)
                    frame = last_frames[i]
                
                resized_frame = resize_frame(frame, cell_width, cell_height)
                
                row = i // n_cols
                col = i % n_cols
                
                # Calculate position with gaps
                y_start = gap + row * (cell_height + gap)
                x_start = gap + col * (cell_width + gap)
                
                output_frame[y_start : y_start + cell_height, x_start : x_start + cell_width] = resized_frame
            
            rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            
            if frame_idx % int(target_fps) == 0 or frame_idx == total_frames - 1:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"Processing... {progress:.1f}% complete", end='\r')

    finally:
        print("\nCleaning up resources...")
        for cap in caps:
            cap.release()
        writer.close()
    
    print(f"Video grid successfully saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a grid visualization of multiple videos with separators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_paths", nargs="+", help="One or more paths to input videos.")
    parser.add_argument("--n_rows", type=int, default=1, help="Number of rows in the grid.")
    parser.add_argument("--n_cols", type=int, help="Number of columns. Defaults to the number of videos if n_rows is 1.")
    parser.add_argument("--save_path", default="output.mp4", help="Output path for the video (e.g., 'output.mp4' or 'output.gif').")
    parser.add_argument("--gap", type=int, default=10, help="Size of the gap between videos in pixels.")
    parser.add_argument("--grid_color", default="white", choices=["white", "black"], help="Color of the grid gaps.")
    parser.add_argument("--max_cell_width", type=int, default=640, help="Maximum width for each video cell in the grid.")

    args = parser.parse_args()
    
    # Set default n_cols if not provided
    if args.n_cols is None:
        args.n_cols = len(args.video_paths) // args.n_rows
        if len(args.video_paths) % args.n_rows != 0:
            args.n_cols += 1

    try:
        create_video_grid(
            video_paths=args.video_paths,
            n_rows=args.n_rows,
            n_cols=args.n_cols,
            save_path=args.save_path,
            gap=args.gap,
            grid_color_name=args.grid_color,
            max_cell_width=args.max_cell_width
        )
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()