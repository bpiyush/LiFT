"""Checks cut files."""
import os
import sys
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import decord
import numpy as np
import pandas as pd


if __name__ == "__main__":
    video_dir = "/work/piyush/from_nfs2/datasets/EPIC-Kitchens-100/cut_clips"
    files = glob(os.path.join(video_dir, "*/*/*.MP4"))
    print("Total files:", len(files))

    parallel = True

    if not parallel:
        failed = []
        iterator = tqdm(files, desc="Checking files")
        for f in iterator:
            try:
                vr = decord.VideoReader(f, ctx=decord.cpu(), num_threads=1)
                random_frame = np.random.randint(0, len(vr))
                random_frame = vr.get_batch([random_frame]).asnumpy()
            except Exception as e:
                failed.append(f)
        import ipdb; ipdb.set_trace()
    else:
        def check_file(f):
            try:
                vr = decord.VideoReader(f, ctx=decord.cpu(), num_threads=1)
                random_frame = np.random.randint(0, len(vr))
                random_frame = len(vr) - 1
                random_frame = vr.get_batch([random_frame]).asnumpy()
                return None
            except Exception as e:
                return f

        status = Parallel(n_jobs=24)(
            delayed(check_file)(f) for f in tqdm(files, desc="Checking files")
        )
        failed = [f for f in status if f is not None]
        print("Number of files on which loading failed:", len(failed))
        import ipdb; ipdb.set_trace()

        for f in failed: os.remove(f)
    