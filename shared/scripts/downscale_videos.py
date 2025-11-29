#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib import parallel as joblib_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downscale videos and save them to a new folder with a given extension."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("/scratch/shared/beegfs/piyush/datasets/NTU/nturgb+d_rgb/"),
        help="Root directory containing videos (searched recursively)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="avi",
        help="Video file extension to search for (without dot)",
    )
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default=0.4,
        help="Factor by which to downscale width and height (e.g., 0.4)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to save downscaled videos. Defaults to video_dir + '-downscaled={factor}'",
    )
    parser.add_argument(
        "--save_ext",
        type=str,
        default="mp4",
        help="Extension to save resulting videos with (without dot)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first 10 videos and print saved paths",
    )
    parser.add_argument(
        "--si",
        type=int,
        default=0,
        help="Start index (inclusive) in the sorted video list",
    )
    parser.add_argument(
        "--ei",
        type=int,
        default=None,
        help="End index (exclusive) in the sorted video list; None means till end",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all cores)",
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = Path(f"{args.video_dir}-downscaled={args.downscale_factor}")

    # Normalize extensions (strip leading dots)
    args.ext = args.ext.lstrip('.')
    args.save_ext = args.save_ext.lstrip('.')
    return args


def ensure_even_dimension(value: int) -> int:
    if value < 1:
        return 1
    return value if value % 2 == 0 else value - 1 if value > 1 else 1


def list_videos(video_dir: Path, ext: str) -> List[Path]:
    pattern = f"**/*.{ext}"
    return sorted(video_dir.rglob(pattern))


def change_extension(path: Path, new_ext: str) -> Path:
    return path.with_suffix('.' + new_ext)


def downscale_video(
    src_path: Path,
    dst_path: Path,
    downscale_factor: float,
    save_ext: str,
) -> bool:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return False

    # Read properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN check
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = ensure_even_dimension(max(1, int(width * downscale_factor)))
    out_h = ensure_even_dimension(max(1, int(height * downscale_factor)))

    # Choose FOURCC based on extension
    save_ext_lower = save_ext.lower()
    if save_ext_lower in {"mp4", "m4v"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif save_ext_lower in {"avi"}:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif save_ext_lower in {"mov"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        # Fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        return False

    ok = True
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    except Exception:
        ok = False
    finally:
        writer.release()
        cap.release()

    # If nothing written, treat as failure
    if dst_path.exists() and dst_path.stat().st_size > 0 and ok:
        return True
    try:
        if dst_path.exists():
            dst_path.unlink()
    except Exception:
        pass
    return False


def process_one(
    src: Path,
    dst: Path,
    downscale_factor: float,
    save_ext: str,
) -> tuple:
    success = downscale_video(src, dst, downscale_factor, save_ext)
    return success, dst


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib_parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = old_cb
        try:
            tqdm_object.close()
        except Exception:
            pass


def main() -> int:
    args = parse_args()

    video_dir: Path = args.video_dir
    save_dir: Path = args.save_dir
    ext: str = args.ext
    save_ext: str = args.save_ext
    downscale_factor: float = args.downscale_factor
    si: int = max(0, int(args.si))
    ei = args.ei if args.ei is None else max(0, int(args.ei))
    n_jobs: int = int(args.n_jobs)

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"ERROR: video_dir does not exist or is not a directory: {video_dir}", file=sys.stderr)
        return 1

    videos = list_videos(video_dir, ext)

    # Slice by [si:ei]
    try:
        videos = videos[si:ei]
    except Exception:
        # Fallback if indexing fails
        videos = []

    if len(videos) == 0:
        print("No videos found.")
        return 0

    # In debug mode, limit to first 10 from the sliced list
    if args.debug:
        videos = videos[:10]

    # Build (src, dst) pairs and count already existing
    tasks = []
    skipped_count = 0
    for src in videos:
        try:
            rel = src.relative_to(video_dir)
        except ValueError:
            rel = src.name
        rel_path = Path(rel)
        rel_with_new_ext = change_extension(rel_path, save_ext)
        dst = (save_dir / rel_with_new_ext).resolve()
        if dst.exists() and dst.stat().st_size > 0:
            skipped_count += 1
            continue
        tasks.append((src, dst))

    total_count = len(videos)
    saved_paths: List[Path] = []
    errors: int = 0

    with tqdm(total=total_count, desc="Downscaling videos", unit="vid") as pbar:
        # account for already existing outputs
        if skipped_count:
            pbar.update(skipped_count)

        if len(tasks) > 0:
            with tqdm_joblib(pbar):
                results = Parallel(n_jobs=n_jobs, backend="loky")( \
                    delayed(process_one)(src, dst, downscale_factor, save_ext) for (src, dst) in tasks
                )
            for success, dst in results:
                if success:
                    saved_paths.append(dst)
                else:
                    errors += 1
    
    if args.debug:
        print("Saved (debug mode):")
        for p in saved_paths:
            print(str(p))

    if errors > 0:
        print(f"Completed with {errors} failures out of {len(videos)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


