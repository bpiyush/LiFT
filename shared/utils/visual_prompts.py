"""Helpers for visual prompts."""
import os
import random
import numpy as np
import cv2
import decord

from PIL import Image, ImageDraw, ImageFont


def add_absolute_time_bar(video_path, save_path, bar_height=20, bar_color="red", max_duration=10.):
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import crop

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Get video properties
    video_duration = video_clip.duration  # Total duration of the video in seconds
    video_width, video_height = video_clip.size
    
    # Convert bar color (e.g., "red") to RGB
    color_map = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}  # Extendable
    bar_color_rgb = color_map.get(bar_color.lower(), (255, 0, 0))  # Default is red
    
    def add_bar_to_frame(get_frame, t):
        """ Function to add a time bar to each frame based on time t (in seconds). """
        frame = get_frame(t)
        current_time = min(t, max_duration)  # Cap the time at max_duration

        # Calculate the proportional bar length
        bar_length = int((current_time / max_duration) * video_width)
        
        # Create a new bar image with the same width as the frame
        bar = np.zeros((bar_height, video_width, 3), dtype=np.uint8)
        bar[:, :bar_length] = bar_color_rgb
        
        # Stack the bar on top of the current frame
        frame_with_bar = np.vstack([bar, frame])
        return frame_with_bar
    
    # Modify the video frames by adding the time bar
    video_with_bar = video_clip.fl(add_bar_to_frame)
    
    # Crop the extra height from the video (if needed)
    video_with_bar = crop(video_with_bar, height=video_height)  # Keeps original video height
    
    # Save the modified video to the output path
    video_with_bar.write_videofile(save_path, codec="libx264", verbose=False, logger=None)


def add_clock(video_path, save_path, clock_color='red', clock_location=None, clock_radius=25):
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import crop

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Get video properties
    video_width, video_height = video_clip.size
    video_duration = video_clip.duration  # Total duration of the video in seconds
    
    # Default clock location if not provided: 10% height and width from the top-right corner
    if clock_location is None:
        clock_location = [int(video_width * 0.9), int(video_height * 0.1)]

        # Make sure the clock does not go out of the image
        clock_location[0] = min(clock_location[0], video_width - clock_radius)
        clock_location[1] = max(clock_location[1], clock_radius)
        clock_location = tuple(clock_location)
    
    # Convert clock_color from string to BGR for OpenCV
    color_map = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}  # Extendable
    bar_color_rgb = color_map.get(clock_color.lower(), (255, 0, 0))  # Default is red

    def add_clock_to_frame(get_frame, t):
        """ Function to add a clock bar (circle) that fills clockwise as time progresses. """
        frame = get_frame(t)
        
        # Create a copy of the frame to modify
        frame = np.array(frame)
        
        # Calculate the angle based on the current time t relative to the total video duration
        angle = int((t / video_duration) * 360)  # From 0 to 360 degrees
        
        # Draw a partial filled circle (arc) to represent the clock progress
        center = clock_location  # The center of the clock
        thickness = -1  # Solid fill
        
        # Draw the circular progress using cv2.ellipse
        cv2.ellipse(
            frame, 
            center, 
            (clock_radius, clock_radius),  # Clock radius in both x and y direction
            -90,  # Start angle
            0,  # Starting point (0 degrees)
            angle,  # End angle based on progress
            bar_color_rgb,  # Clock color in BGR
            thickness  # Fill the shape
        )
        
        return frame

    # Apply the clock bar to each frame in the video
    video_with_clock = video_clip.fl(add_clock_to_frame)
    
    # Save the modified video with the clock to the output path
    video_with_clock.write_videofile(save_path, codec="libx264", verbose=False, logger=None)


def add_expanding_shape(
    video_path, 
    save_path, 
    shape='circle', 
    max_duration=None, 
    shape_color='blue', 
    shape_location=None, 
    min_size=10, 
    max_size=50,
    thickness=3,
):
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import crop

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Get video properties
    video_width, video_height = video_clip.size
    video_duration = video_clip.duration if max_duration is None else max_duration
    
    # Default shape location if not provided: 10% width and height from the top-right corner
    if shape_location is None:
        shape_location = [int(video_width * 0.9), int(video_height * 0.1)]

        # # Make sure the shape does not go out of the image
        # shape_location[0] = min(shape_location[0], video_width - max_size)
        # shape_location[1] = max(shape_location[1], max_size)

        # Move the location if the shape goes out of the image
        if shape_location[0] + max_size > video_width:
            shape_location[0] = video_width - max_size
        if shape_location[1] - max_size < 0:
            shape_location[1] = max_size

        shape_location = tuple(shape_location)
    else:
        if isinstance(shape_location, str):
            shape_location = shape_location.lower()
            if shape_location == 'center':
                shape_location = (video_width // 2, video_height // 2)
            elif shape_location == 'top-left':
                shape_location = (max_size, max_size)
            elif shape_location == 'top-right':
                shape_location = (video_width - max_size, max_size)
            elif shape_location == 'bottom-left':
                shape_location = (max_size, video_height - max_size)
            elif shape_location == 'bottom-right':
                shape_location = (video_width - max_size, video_height - max_size)
            else:
                raise ValueError(f"Invalid shape_location: {shape_location}")
        else:
            assert len(shape_location) == 2, "shape_location must be a tuple of (x, y) coordinates."
            shape_location = tuple(shape_location)

    
    # Convert shape_color from string to BGR for OpenCV
    color_map = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # Extendable
    shape_color_bgr = color_map.get(shape_color.lower(), (0, 0, 255))  # Default is red

    def add_shape_to_frame(get_frame, t):
        """ Function to add an expanding shape (circle, square, or triangle) to each frame based on time t. """
        frame = get_frame(t)
        
        # Create a copy of the frame to modify
        frame = np.array(frame)
        
        # Calculate the current size of the shape based on the time elapsed
        current_size = int(min_size + (max_size - min_size) * (t / video_duration))
        
        # Get the center for placing the shape
        center = shape_location
        
        # Draw the specified shape
        if shape == 'circle':
            cv2.circle(frame, center, current_size, shape_color_bgr, thickness)  # Filled circle
            
        elif shape == 'square':
            top_left = (center[0] - current_size, center[1] - current_size)
            bottom_right = (center[0] + current_size, center[1] + current_size)
            cv2.rectangle(frame, top_left, bottom_right, shape_color_bgr, thickness)  # Filled square
        
        elif shape == 'triangle':
            # Define triangle points
            p1 = (center[0], center[1] - current_size)  # Top point
            p2 = (center[0] - current_size, center[1] + current_size)  # Bottom-left point
            p3 = (center[0] + current_size, center[1] + current_size)  # Bottom-right point
            pts = np.array([p1, p2, p3], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], shape_color_bgr)  # Filled triangle
        
        return frame

    # Apply the expanding shape to each frame in the video
    video_with_shape = video_clip.fl(add_shape_to_frame)
    
    # Save the modified video with the expanding shape to the output path
    video_with_shape.write_videofile(save_path, codec="libx264", verbose=False, logger=None)
