"""
Creates multiple shards of videos in a dataset.

Help:

input_dir=/work/piyush/from_nfs2/datasets/SSv2/20bn-something-something-v2/
output_dir=/work/piyush/from_nfs2/datasets/SSv2/ssv2_shards/
python shared/scripts/shard_video_dataset.py -i $input_dir -o $output_dir
"""
import os
import tarfile
from glob import glob

import shared.utils as su


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Input video directory
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default="webm")
    # Output shard directory
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    # Max size of each shard (n.o. videos)
    parser.add_argument("--shard_size", type=int, default=10000)
    args = parser.parse_args()
    return args


# Iterate over videos in your dataset
def write_to_shard(tar, video_path, video_name):
    with open(video_path, 'rb') as f:
        tarinfo = tarfile.TarInfo(name=video_name)
        tarinfo.size = os.path.getsize(video_path)
        tar.addfile(tarinfo, f)


if __name__ == "__main__":

    # Read arguments
    args = read_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all video files
    su.log.print_update("Processing files at " + args.input_dir)
    video_files = glob(os.path.join(args.input_dir, f"*.{args.ext}"))
    print(f"Found {len(video_files)} video files")

    # Iterate
    iterator = su.log.tqdm_iterator(video_files, desc="Sharding videos")

    shard_size = args.shard_size
    output_dir = args.output_dir
    shard_id = 0
    i = 0
    for video_file in iterator:
        if i % shard_size == 0:
            if i > 0:
                tar.close()
            shard_path = os.path.join(output_dir, f"shard-{shard_id:04d}.tar")
            print(f"Creating shard {shard_path}")
            tar = tarfile.open(shard_path, 'w')
            shard_id += 1
        write_to_shard(tar, video_file, os.path.basename(video_file))
        i += 1

    if tar:
        tar.close()