import os
import csv
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict
from functools import partial
import webdataset as wds
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV file to WebDataset format with video data")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WebDataset shards")
    parser.add_argument("--num_shards", type=int, default=128, help="Number of shards to create")
    parser.add_argument("--samples_per_shard", type=int, default=None, 
                        help="Max samples per shard (overrides num_shards if specified)")
    parser.add_argument("--worker_count", type=int, default=mp.cpu_count(), 
                        help="Number of worker processes")
    parser.add_argument("--shard_prefix", type=str, default="shard", 
                        help="Prefix for shard filenames")
    parser.add_argument("--video_extension", type=str, default=".webm", 
                        help="Extension of video files (default: .webm)")
    parser.add_argument("--debug", action="store_true", 
                        help="Debug mode: create a shard_debug.tar file with max 1000 videos")
    parser.add_argument("--si", type=int, default=0, help="Start index")
    parser.add_argument("--ei", type=int, default=None, help="End index")
    return parser.parse_args()


def read_csv_data(csv_path: str, debug: bool = False) -> List[Dict]:
    """Read the CSV file and return a list of samples."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            samples.append(row)
            # In debug mode, limit to 1000 samples
            if debug and i >= 999:
                break
    
    # Select start and end index if specified
    si = args.si
    ei = args.ei if args.ei is not None else len(samples)
    print("Selected samples from index", si, "to", ei)
    samples = samples[si:ei]

    return samples


def distribute_samples(samples: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Distribute samples across shards."""
    samples_per_shard = len(samples) // num_shards
    remainder = len(samples) % num_shards
    
    distributed_samples = []
    start_idx = 0
    
    for i in range(num_shards):
        # Add one extra sample for the first 'remainder' shards
        shard_size = samples_per_shard + (1 if i < remainder else 0)
        end_idx = start_idx + shard_size
        
        distributed_samples.append(samples[start_idx:end_idx])
        start_idx = end_idx
    
    return distributed_samples


def process_shard(shard_samples: List[Dict], shard_path: str, video_extension: str = ".webm"):
    """Process and write a single shard with actual video data."""
    with wds.TarWriter(shard_path) as sink:
        for sample in tqdm(shard_samples, desc=f"Processing {shard_path}"):
            video_path = sample['video_path']

            vr = VideoReader(video_path, num_threads=1)
            n_frames = len(vr)
            fps = vr.get_avg_fps()
            H, W, _ = vr[0].shape
            
            try:
                # Read video file as binary data
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                
                # Get filename without path for the key
                filename = Path(video_path).stem

                # Create sample with the actual video data
                sample_dict = {
                    "__key__": filename,
                    "video": video_data,  # Actual video binary data
                    "video.extension": video_extension.lstrip('.'),  # Store extension without dot
                    "target": str(sample['target']), # Target/label
                    # "split": str(sample['split']),  # Train/val/test split
                    "json": dict(n_frames=n_frames, fps=fps, H=H, W=W),  # Additional metadata
                }
                
                sink.write(sample_dict)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")


import io
import torchvision

def encode_tensor(tensor):
    """
    Convert tensor to bytes in memory.
    """
    # Convert the tensor to bytes in memory
    with io.BytesIO() as buf:
        if isinstance(tensor, torch.Tensor):
            torch.save(tensor, buf)  # Save tensor to the buffer
        return buf.getvalue()  # Return the byte data


def process_shard_tensor(shard_samples: List[Dict], shard_path: str, video_extension: str = ".webm"):
    """Process and write a single shard with actual video data (actual tensor)."""
    with wds.TarWriter(shard_path) as sink:
        for sample in tqdm(shard_samples, desc=f"Processing {shard_path}"):
            video_path = sample['video_path']

            # Load the entire video as a tensor
            video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
            video_data = encode_tensor(video)
            n_frames = len(video)
            fps = info['video_fps']
            H, W = video.shape[1:-1]
            
            try:
                # # Read video file as binary data
                # with open(video_path, 'rb') as f:
                #     video_data = f.read()
                
                # Get filename without path for the key
                filename = Path(video_path).stem

                # Create sample with the actual video data
                sample_dict = {
                    "__key__": filename,
                    "video": video_data,  # Actual video binary data
                    "video.extension": video_extension.lstrip('.'),  # Store extension without dot
                    "target": str(sample['target']), # Target/label
                    # "split": str(sample['split']),  # Train/val/test split
                    "json": dict(n_frames=n_frames, fps=fps, H=H, W=W),  # Additional metadata
                }

                sink.write(sample_dict)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")


def create_webdataset(csv_path: str, output_dir: str, num_shards: int, 
                      samples_per_shard: int = None, worker_count: int = None,
                      shard_prefix: str = "shard", video_extension: str = ".webm", 
                      debug: bool = False):
    """Convert CSV to WebDataset format with video data and parallel processing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all samples from the CSV
    print(f"Reading samples from {csv_path}...")
    samples = read_csv_data(csv_path, debug=debug)
    total_samples = len(samples)
    print(f"Found {total_samples} samples in the CSV file")
    
    # Handle debug mode with a specific shard_debug.tar file
    if debug:
        print("Debug mode enabled: Creating shard_debug.tar with max 1000 videos")
        debug_shard_path = os.path.join(output_dir, "shard_debug.tar")
        # process_shard(samples, debug_shard_path, video_extension)
        process_shard_tensor(samples, debug_shard_path, video_extension)
        
        # Calculate and display file size
        file_size = os.path.getsize(debug_shard_path)
        print(f"Created debug shard: {debug_shard_path}")
        print(f"Debug shard size: {file_size / (1024**2):.2f} MB")
        
        # Test the debug shard
        test_dataset(output_dir, debug_pattern="shard_debug.tar")
        return
    
    # Determine number of shards based on samples_per_shard if provided
    if samples_per_shard is not None:
        num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard
        print(f"Creating {num_shards} shards with max {samples_per_shard} samples per shard")
    else:
        print(f"Creating {num_shards} shards")
    
    # Distribute samples across shards
    shard_samples = distribute_samples(samples, num_shards)
    
    # Prepare shard paths
    shard_paths = [
        os.path.join(output_dir, f"{shard_prefix}_{i:05d}.tar") 
        for i in range(num_shards)
    ]
    
    # Use all available cores if worker_count is not specified
    if worker_count is None:
        worker_count = mp.cpu_count()
        
    worker_count = min(worker_count, num_shards)  # Don't use more workers than shards
    
    print(f"Using {worker_count} worker processes")
    
    # Process shards in parallel with video extension
    # process_func = partial(process_shard, video_extension=video_extension)
    process_func = partial(process_shard_tensor, video_extension=video_extension)
    
    with mp.Pool(worker_count) as pool:
        list(tqdm(
            pool.starmap(process_func, zip(shard_samples, shard_paths)),
            total=num_shards,
            desc="Creating WebDataset shards with video data"
        ))
    
    print(f"Successfully created {num_shards} WebDataset shards in {output_dir}")
    
    # Calculate and display total dataset size
    total_size = sum(os.path.getsize(path) for path in shard_paths)
    print(f"Total dataset size: {total_size / (1024**2):.2f} MB")


def test_dataset(output_dir: str, shard_prefix: str = "shard", debug_pattern: str = None):
    """Test reading from the created WebDataset."""
    # Find all shard files or use debug pattern
    if debug_pattern:
        shard_pattern = os.path.join(output_dir, debug_pattern)
    else:
        shard_pattern = os.path.join(output_dir, f"{shard_prefix}_*.tar")
    
    # Create a dataset
    dataset = wds.WebDataset(shard_pattern)
    
    # Display sample info
    print("\nTesting dataset:")
    for i, sample in enumerate(dataset):
        print(f"Sample {i}:")
        for key, value in sample.items():
            if key == "video":
                print(f"  {key}: <binary data of length {len(value)}>")
            else:
                print(f"  {key}: {value}")
        
        if i >= 2:  # Just show a few samples
            break


if __name__ == "__main__":
    args = parse_args()
    create_webdataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        worker_count=args.worker_count,
        shard_prefix=args.shard_prefix,
        video_extension=args.video_extension,
        debug=args.debug
    )