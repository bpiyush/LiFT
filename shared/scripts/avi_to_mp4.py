import os
import glob
import argparse
from moviepy.editor import VideoFileClip
import sys

def convert_avi_to_mp4(src_dir, dst_dir, ext, start_idx, end_idx):
    # Ensure destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Get list of all .avi files in subdirectories
    avi_files = glob.glob(os.path.join(src_dir, f'**/*.{ext}'), recursive=True)
    
    if not avi_files:
        print(f"No .{ext} files found in {src_dir}")
        return

    # Apply start and end index filtering
    avi_files = avi_files[start_idx:end_idx]
    
    for avi_file in avi_files:
        # Get file ID without extension and parent path
        file_id = os.path.splitext(os.path.relpath(avi_file, src_dir))[0]
        mp4_file = os.path.join(dst_dir, f"{file_id}.mp4")
        
        # Create any necessary subdirectories in dst_dir
        os.makedirs(os.path.dirname(mp4_file), exist_ok=True)
        
        try:
            # Suppress moviepy verbose output
            sys.stdout = open(os.devnull, 'w')
            clip = VideoFileClip(avi_file)
            clip.write_videofile(mp4_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            sys.stdout = sys.__stdout__
            print(f"Converted: {avi_file} -> {mp4_file}")
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"Error converting {avi_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .avi files to .mp4")
    parser.add_argument("--src_dir", required=True, help="Source directory containing .avi files")
    parser.add_argument("--dst_dir", required=True, help="Destination directory for .mp4 files")
    parser.add_argument("--ext", default="avi", help="Extension of the source files (default: avi)")
    parser.add_argument("--si", type=int, default=0, help="Start index of files to process (default: 0)")
    parser.add_argument("--ei", type=int, default=None, help="End index of files to process (default: None)")
    
    args = parser.parse_args()

    convert_avi_to_mp4(args.src_dir, args.dst_dir, args.ext, args.si, args.ei)
