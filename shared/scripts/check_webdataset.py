"""Loads tar files using webdataset."""
import os
import webdataset as wds
import decord
from torch.utils.data import DataLoader
import numpy as np
import einops


# Define a function to decode videos using Decord
def decode_video(video_bytes):
    # Save the video bytes to a temporary file and decode with decord
    vr = decord.VideoReader(video_bytes)
    frames = [vr[i].asnumpy() for i in range(0, len(vr), 5)]  # Frame skip example
    return frames


def convert_bytes_to_frames(video_bytes):
    vr = decord.VideoReader(video_bytes)
    frames = [vr[i].asnumpy() for i in range(0, len(vr), 5)]  # Frame skip example
    return frames



def decode_video(video_bytes):
    """
    Given video bytes, decode them into frames.
    """
    pass


if __name__ == "__main__":
    shard_folder = "/work/piyush/from_nfs2/datasets/SSv2/ssv2_shards/"
    shard_path = os.path.join(shard_folder, "shard-0000.tar")

    # Define your WebDataset path pattern (all shards)
    dataset_path = os.path.join(shard_folder, "shard-{0000..0002}.tar")

    ds = wds.WebDataset(dataset_path)
    sample = next(iter(ds))
    dl = DataLoader(ds, batch_size=16, num_workers=8)
    batch = next(iter(dl))

    # Create a WebDataset loader
    dataset = (
        wds.WebDataset(dataset_path)
        # .decode("rgb")  # Ensure that we decode the video bytes into RGB images
        .to_tuple("webm")  # Ensure that we get the video bytes and metadata
        # .map_tuple(decode_video)  # Apply your video decoding function
    )
    # dataloader = DataLoader(dataset, batch_size=16, num_workers=8)
    # batch = next(iter(dataloader))
    # print(batch[0].shape)  # (16, 32, 256, 256, 3) for example
    # H, W, C = 256, 256, 3
    H, W, C = (240, 427, 3)
    for (video,) in dataset:
        # Get file name


        # print(len(video[0]))
        print(type(video))
        video_array = np.frombuffer(video, dtype=np.uint8)
        reshaped_video = einops.rearrange(video_array, "(t h w c) -> t c h w", h=H, w=W, c=C)
        # print(len(video))
        # print(video[1])

        # np_video_bytes = np.frombuffer(video[0], np.uint8)
        break
    import ipdb; ipdb.set_trace()