import subprocess
import os
from pathlib import Path

def create_side_by_side_gif(video_paths, output_gif, gap_width=20, fps=10, scale_height=240, gap_color="white", verbose=False):
    """
    Create a single GIF with multiple videos placed side by side.
    
    Args:
        video_paths (list): List of paths to input MP4 files
        output_gif (str): Path for output GIF file
        gap_width (int): Width of gap between videos in pixels
        fps (int): Frame rate for output GIF
        scale_height (int): Height to scale all videos to (maintains aspect ratio)
        gap_color (str): Color for gaps between videos (e.g., "white", "black", "red", "#FF0000")
        verbose (bool): Whether to print FFmpeg commands and processing messages
    """
    
    if not video_paths:
        raise ValueError("No video paths provided")
    
    # Verify all input files exist
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    # Create filter complex string for FFmpeg
    num_videos = len(video_paths)
    
    # Input mapping and scaling
    filter_parts = []
    scaled_inputs = []
    
    for i, _ in enumerate(video_paths):
        # Scale each video to same height while maintaining aspect ratio
        filter_parts.append(f"[{i}:v]scale=-1:{scale_height}[v{i}]")
        scaled_inputs.append(f"[v{i}]")
    
    # Create horizontal stack with gaps
    if num_videos == 1:
        hstack_filter = f"{scaled_inputs[0]}copy[stacked]"
    else:
        # Create colored gap between videos
        gap_filters = []
        for i in range(num_videos - 1):
            gap_filters.append(f"color={gap_color}:{gap_width}x{scale_height}:d=1[gap{i}]")
        
        if gap_filters:
            filter_parts.extend(gap_filters)
        
        # Build hstack input list with gaps
        hstack_inputs = []
        for i in range(num_videos):
            hstack_inputs.append(scaled_inputs[i])
            if i < num_videos - 1:  # Add gap after each video except the last
                hstack_inputs.append(f"[gap{i}]")
        
        hstack_filter = f"{''.join(hstack_inputs)}hstack=inputs={len(hstack_inputs)}[stacked]"
    
    filter_parts.append(hstack_filter)
    
    # Complete the filter complex for stacked video
    stacked_filter = ";".join(filter_parts)
    
    # Build FFmpeg command with two-pass palette approach
    cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
    
    # Add input files
    for video_path in video_paths:
        cmd.extend(["-i", video_path])
    
    # Add filter complex and output options
    cmd.extend([
        "-filter_complex", f"{stacked_filter};[stacked]split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        "-r", str(fps),  # Set frame rate
        "-loop", "0",    # Infinite loop
        output_gif
    ])
    
    if verbose:
        print("Running FFmpeg command:")
        print(" ".join(cmd))
        print("\nProcessing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"✓ Successfully created GIF: {output_gif}")
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        if verbose:
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return False

def create_top_to_bottom_gif(video_paths, output_gif, gap_height=20, fps=10, scale_width=320, gap_color="white", verbose=False):
    """
    Create a single GIF with multiple videos stacked vertically (top to bottom).
    
    Args:
        video_paths (list): List of paths to input MP4 files
        output_gif (str): Path for output GIF file
        gap_height (int): Height of gap between videos in pixels
        fps (int): Frame rate for output GIF
        scale_width (int): Width to scale all videos to (maintains aspect ratio)
        gap_color (str): Color for gaps between videos (e.g., "white", "black", "red", "#FF0000")
        verbose (bool): Whether to print FFmpeg commands and processing messages
    """
    
    if not video_paths:
        raise ValueError("No video paths provided")
    
    # Verify all input files exist
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    # Create filter complex string for FFmpeg
    num_videos = len(video_paths)
    
    # Input mapping and scaling
    filter_parts = []
    scaled_inputs = []
    
    for i, _ in enumerate(video_paths):
        # Scale each video to same width while maintaining aspect ratio
        filter_parts.append(f"[{i}:v]scale={scale_width}:-1[v{i}]")
        scaled_inputs.append(f"[v{i}]")
    
    # Create vertical stack with gaps
    if num_videos == 1:
        vstack_filter = f"{scaled_inputs[0]}copy[stacked]"
    else:
        # Create colored gap between videos
        gap_filters = []
        for i in range(num_videos - 1):
            gap_filters.append(f"color={gap_color}:{scale_width}x{gap_height}:d=1[gap{i}]")
        
        if gap_filters:
            filter_parts.extend(gap_filters)
        
        # Build vstack input list with gaps
        vstack_inputs = []
        for i in range(num_videos):
            vstack_inputs.append(scaled_inputs[i])
            if i < num_videos - 1:  # Add gap after each video except the last
                vstack_inputs.append(f"[gap{i}]")
        
        vstack_filter = f"{''.join(vstack_inputs)}vstack=inputs={len(vstack_inputs)}[stacked]"
    
    filter_parts.append(vstack_filter)
    
    # Complete the filter complex for stacked video
    stacked_filter = ";".join(filter_parts)
    
    # Build FFmpeg command with two-pass palette approach
    cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
    
    # Add input files
    for video_path in video_paths:
        cmd.extend(["-i", video_path])
    
    # Add filter complex and output options
    cmd.extend([
        "-filter_complex", f"{stacked_filter};[stacked]split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        "-r", str(fps),  # Set frame rate
        "-loop", "0",    # Infinite loop
        output_gif
    ])
    
    if verbose:
        print("Running FFmpeg command:")
        print(" ".join(cmd))
        print("\nProcessing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"✓ Successfully created GIF: {output_gif}")
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        if verbose:
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return False

def reverse_video(input_video_path, output_filename=None, verbose=False):
    """
    Reverse a video file and save it to /tmp directory.
    
    Args:
        input_video_path (str): Path to input MP4 file
        output_filename (str, optional): Name for output file. If None, generates from input filename
        verbose (bool): Whether to print FFmpeg commands and processing messages
        
    Returns:
        str: Path to the reversed video file in /tmp, or None if failed
    """
    
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Generate output filename if not provided
    if output_filename is None:
        input_name = Path(input_video_path).stem
        output_filename = f"{input_name}_reversed.mp4"
    
    # Ensure output filename has .mp4 extension
    if not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Create output path in /tmp
    output_path = os.path.join('/tmp', output_filename)
    
    # Build FFmpeg command to reverse video
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output file
        "-i", input_video_path,
        "-vf", "reverse",  # Video filter to reverse frames
        "-af", "areverse",  # Audio filter to reverse audio
        "-c:v", "libx264",  # Video codec
        "-c:a", "aac",      # Audio codec
        output_path
    ]
    
    if verbose:
        print("Running FFmpeg command to reverse video:")
        print(" ".join(cmd))
        print("\nProcessing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"✓ Successfully created reversed video: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ FFmpeg error: {e.stderr}")
        return None
    except FileNotFoundError:
        if verbose:
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return None

def add_text_overlay(input_video_path, text, output_filename=None, font_size=12, font_color="white", 
                    background_color="black", position="top", margin=10, duration=None, verbose=False):
    """
    Add a text overlay to a video with a background title bar.
    
    Args:
        input_video_path (str): Path to input MP4 file
        text (str): Text to display
        output_filename (str, optional): Name for output file. If None, generates from input filename
        font_size (int): Font size for the text (default: 24)
        font_color (str): Color of the text (default: "white")
        background_color (str): Color of the background bar (default: "black")
        position (str): Position of text bar - "top", "bottom", "center" (default: "top")
        margin (int): Margin from edge in pixels (default: 10)
        duration (float, optional): Duration to show text in seconds. If None, shows for entire video
        verbose (bool): Whether to print FFmpeg commands and processing messages
        
    Returns:
        str: Path to the video with text overlay in /tmp, or None if failed
    """
    
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Generate output filename if not provided
    if output_filename is None:
        input_name = Path(input_video_path).stem
        output_filename = f"{input_name}_with_text.mp4"
    
    # Ensure output filename has .mp4 extension
    if not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Create output path in /tmp
    output_path = os.path.join('/tmp', output_filename)
    
    # Determine text position based on position parameter
    if position == "top":
        text_position = f"x={margin}:y={margin+5}"  # Add small offset for background box
    elif position == "bottom":
        text_position = f"x={margin}:y=h-th-{margin+5}"  # Add small offset for background box
    elif position == "center":
        text_position = f"x={margin}:y=(h-th)/2"
    else:
        text_position = f"x={margin}:y={margin+5}"  # Default to top with offset
    
    # Build the drawtext filter
    drawtext_filter = f"drawtext=text='{text}':fontsize={font_size}:fontcolor={font_color}:{text_position}"
    
    # Add background box if needed
    if background_color != "transparent":
        # Create a semi-transparent background box with estimated height based on font size
        # Estimate text height as approximately 1.2 * font_size
        estimated_text_height = int(font_size * 1.2)
        box_height = estimated_text_height + 10  # Add padding
        
        # Position box based on text position
        if position == "top":
            box_y = margin
        elif position == "bottom":
            box_y = f"h-{box_height}-{margin}"
        elif position == "center":
            box_y = f"(h-{box_height})/2"
        else:
            box_y = margin
            
        box_filter = f"drawbox=x={margin-5}:y={box_y}:w=iw-{2*(margin-5)}:h={box_height}:color={background_color}@0.7:t=fill"
        drawtext_filter = f"{box_filter},{drawtext_filter}"
    
    # Add duration constraint if specified
    if duration is not None:
        drawtext_filter += f":enable='between(t,0,{duration})'"
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output file
        "-i", input_video_path,
        "-vf", drawtext_filter,
        "-c:v", "libx264",  # Video codec
        "-c:a", "copy",     # Copy audio without re-encoding
        output_path
    ]
    
    if verbose:
        print("Running FFmpeg command to add text overlay:")
        print(" ".join(cmd))
        print("\nProcessing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"✓ Successfully created video with text overlay: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ FFmpeg error: {e.stderr}")
        return None
    except FileNotFoundError:
        if verbose:
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return None

def add_text_strip(input_video_path, text, output_filename=None, font_size=16, font_color="white", 
                  background_color="black", position="top", text_padding=20, max_width_ratio=0.9, verbose=False):
    """
    Add a text strip/bar to a video (increases video height) rather than overlaying text.
    
    Args:
        input_video_path (str): Path to input MP4 file
        text (str): Text to display in the strip
        output_filename (str, optional): Name for output file. If None, generates from input filename
        font_size (int): Font size for the text (default: 16)
        font_color (str): Color of the text (default: "white")
        background_color (str): Color of the background strip (default: "black")
        position (str): Position of text strip - "top" or "bottom" (default: "top")
        text_padding (int): Padding around text in pixels (default: 20)
        max_width_ratio (float): Maximum width of text as ratio of video width (default: 0.9)
        verbose (bool): Whether to print FFmpeg commands and processing messages
        
    Returns:
        str: Path to the video with text strip in /tmp, or None if failed
    """
    
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Generate output filename if not provided
    if output_filename is None:
        input_name = Path(input_video_path).stem
        output_filename = f"{input_name}_with_strip.mp4"
    
    # Ensure output filename has .mp4 extension
    if not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Create output path in /tmp
    output_path = os.path.join('/tmp', output_filename)
    
    # Calculate text strip height based on font size, padding, and estimated line count
    # Estimate characters per line based on font size (roughly 2 characters per font size pixel)
    estimated_chars_per_line = int(font_size * 2)
    text_lines = text.split('\n') if '\n' in text else [text]
    
    # If text is too long, wrap it
    wrapped_lines = []
    for line in text_lines:
        if len(line) <= estimated_chars_per_line:
            wrapped_lines.append(line)
        else:
            # Simple word wrapping
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= estimated_chars_per_line:
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word
            if current_line:
                wrapped_lines.append(current_line)
    
    # Calculate strip height based on number of lines
    line_height = font_size + 5  # Add some line spacing
    strip_height = (len(wrapped_lines) * line_height) + (2 * text_padding)
    
    # Create text strip using a different approach - pad the video and add text
    # This will add padding above the video and put text in that padded area
    if position == "top":
        # Add padding to top of video and put text in the padded area
        text_strip_filter = f"[0:v]pad=iw:ih+{strip_height}:0:{strip_height}:{background_color}[padded];[padded]drawtext=text='{chr(10).join(wrapped_lines)}':fontsize={font_size}:fontcolor={font_color}:x=(w-tw)/2:y={text_padding}:line_spacing={line_height}[stacked]"
    else:  # bottom
        # Add padding to bottom of video and put text in the padded area
        text_strip_filter = f"[0:v]pad=iw:ih+{strip_height}:0:0:{background_color}[padded];[padded]drawtext=text='{chr(10).join(wrapped_lines)}':fontsize={font_size}:fontcolor={font_color}:x=(w-tw)/2:y=h-th-{text_padding}:line_spacing={line_height}[stacked]"
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output file
        "-i", input_video_path,
        "-filter_complex", text_strip_filter,
        "-map", "[stacked]",  # Map the processed video
        "-map", "0:a",        # Map the original audio
        "-c:v", "libx264",    # Video codec
        "-c:a", "copy",       # Copy audio without re-encoding
        output_path
    ]
    
    if verbose:
        print("Running FFmpeg command to add text strip:")
        print(" ".join(cmd))
        print("\nProcessing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"✓ Successfully created video with text strip: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ FFmpeg error: {e.stderr}")
        return None
    except FileNotFoundError:
        if verbose:
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return None

def get_video_info(video_path):
    """Get basic info about a video file."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", 
        "-show_format", "-show_streams", video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        # Find video stream
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return {
                    'width': stream['width'],
                    'height': stream['height'],
                    'duration': float(stream.get('duration', 0)),
                    'fps': eval(stream.get('r_frame_rate', '0/1'))
                }
    except:
        pass
    return None






# Example usage
if __name__ == "__main__":
    # Example video paths - replace with your actual video files
    video_files = [
        "examples/folding_paper.mp4",
        "examples/S008C002P032R002A051.mp4", 
    ]
    
    output_file = "combined_videos.gif"
    
    # Check if example files exist
    existing_files = [f for f in video_files if os.path.exists(f)]
    
    if existing_files:
        print(f"Found {len(existing_files)} video files:")
        for video in existing_files:
            info = get_video_info(video)
            if info:
                print(f"  {video}: {info['width']}x{info['height']}, {info['duration']:.1f}s")
            else:
                print(f"  {video}: (info unavailable)")
        
        # Create the horizontal GIF
        success = create_side_by_side_gif(
            video_paths=existing_files,
            output_gif=output_file,
            gap_width=30,      # 30px gap between videos
            fps=12,            # 12 frames per second
            scale_height=300,  # Scale all videos to 300px height
            gap_color="white"  # White gap between videos
        )
        
        # Also create a vertical GIF
        vertical_output_file = "combined_videos_vertical.gif"
        success_vertical = create_top_to_bottom_gif(
            video_paths=existing_files,
            output_gif=vertical_output_file,
            gap_height=20,     # 20px gap between videos
            fps=12,            # 12 frames per second
            scale_width=320,   # Scale all videos to 320px width
            gap_color="white"  # White gap between videos
        )
        
        if success:
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"\nHorizontal GIF size: {file_size:.1f} MB")
        
        if success_vertical:
            vertical_file_size = os.path.getsize(vertical_output_file) / (1024 * 1024)  # MB
            print(f"Vertical GIF size: {vertical_file_size:.1f} MB")
        
        # Example of reversing a video
        if existing_files:
            print(f"\nReversing first video: {existing_files[0]}")
            reversed_path = reverse_video(existing_files[0])
            if reversed_path:
                print(f"Reversed video saved to: {reversed_path}")
        
        # Example of adding text overlay
        if existing_files:
            print(f"\nAdding text overlay to first video: {existing_files[0]}")
            text_video_path = add_text_overlay(
                input_video_path=existing_files[0],
                text="Sample Title Text",
                font_size=30,
                font_color="white",
                background_color="black",
                position="top",
                margin=15
            )
            if text_video_path:
                print(f"Video with text overlay saved to: {text_video_path}")
        
        # Example of adding text strip
        if existing_files:
            print(f"\nAdding text strip to first video: {existing_files[0]}")
            strip_video_path = add_text_strip(
                input_video_path=existing_files[0],
                text="Video Title Strip",
                font_size=16,
                font_color="white",
                background_color="darkblue",
                position="top",
                text_padding=15
            )
            if strip_video_path:
                print(f"Video with text strip saved to: {strip_video_path}")
    else:
        print("No video files found. Please update the video_files list with your actual MP4 file paths.")
        print("\nExample usage:")
        print("video_files = [")
        print('    "/path/to/your/video1.mp4",')
        print('    "/path/to/your/video2.mp4",')
        print('    "/path/to/your/video3.mp4"')
        print("]")
        print("\n# Create horizontal GIF")
        print("create_side_by_side_gif(video_files, 'horizontal.gif')")
        print("\n# Create vertical GIF")
        print("create_top_to_bottom_gif(video_files, 'vertical.gif')")
        print("\n# Reverse a video")
        print("reversed_path = reverse_video('/path/to/your/video1.mp4')")
        print("print(f'Reversed video: {reversed_path}')")
        print("\n# Add text overlay to video")
        print("text_video = add_text_overlay('/path/to/your/video1.mp4', 'My Title', font_size=30)")
        print("print(f'Video with text: {text_video}')")
        print("\n# Add text strip to video (increases video height)")
        print("strip_video = add_text_strip('/path/to/your/video1.mp4', 'Title Strip', font_size=16)")
        print("print(f'Video with strip: {strip_video}')")