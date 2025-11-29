"""Misc utils."""
import os
from shared.utils.log import tqdm_iterator
import numpy as np
from termcolor import colored


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively turn dictionaries into DictToObj instances
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return str(self.__dict__)


def ignore_warnings(type="ignore"):
    import warnings
    warnings.filterwarnings(type)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_youtube_video(youtube_id, ext='mp4', resolution="360p", **kwargs):
    import pytube
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"
    yt = pytube.YouTube(video_url)
    try:
        streams = yt.streams.filter(
            file_extension=ext, res=resolution, progressive=True, **kwargs,
        )
        # streams[0].download(output_path=save_dir, filename=f"{video_id}.{ext}")
        streams[0].download(output_path='/tmp', filename='sample.mp4')
    except:
        print("Failed to download video: ", video_url)
        return None
    return "/tmp/sample.mp4"


def check_audio(video_path):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    try:
        return VideoFileClip(video_path).audio is not None
    except:
        return False


def check_audio_multiple(video_paths, n_jobs=8):
    """Parallelly check if videos have audio"""
    iterator = tqdm_iterator(video_paths, desc="Checking audio")
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
            delayed(check_audio)(video_path) for video_path in iterator
        )


def num_trainable_params(model, round=3, verbose=True, return_count=False):
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    if verbose:
        print(f"::: Number of trainable parameters in {model_name}: {value} {unit}")
    if return_count:
        return n_params


def num_params(model, round=3):
    n_params = sum([p.numel() for p in model.parameters()])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    print(f"::: Number of total parameters in {model_name}: {value}{unit}")


def fix_seed(seed=42):
    """Fix all numpy/pytorch/random seeds."""
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_tensor(x):
    print(x.shape, x.min(), x.max())


import hashlib
def encode_string(input_string, num_chars=4):
    """
    Encodes a given string into a 4-character code using the SHA-256 hash algorithm.

    Args:
        input_string (str): The input string to be encoded.
        num_chars (int): The number of characters to take from the hash digest.

    Returns:
        str: A 4-character code representing the encoded string.
    """
    # Convert the input string to bytes
    input_bytes = input_string.encode('utf-8')

    # Calculate the SHA-256 hash of the input bytes
    hash_object = hashlib.sha256(input_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = hash_object.hexdigest()

    # Take the first 4 characters of the hexadecimal digest as the code
    code = hex_digest[:num_chars]

    return code


def flatten_list_of_lists(xss):
    return [x for xs in xss for x in xs]


import textwrap


def get_terminal_width():
    import shutil
    return shutil.get_terminal_size().columns


def wrap_text(text: str, max_length: int = 100) -> str:
    """
    Wraps a long string to the specified max_length for easier printing.

    Args:
        text (str): The input string to wrap.
        max_length (int): The maximum length of each line. Default is 80.

    Returns:
        str: The wrapped text with lines at most max_length long.
    """
    terminal_width = get_terminal_width()
    max_length = min(max_length, terminal_width)
    wrapped_text = textwrap.fill(text, width=max_length)
    return wrapped_text


def print_colored(heading, text, color="blue", warp=True):
    width = get_terminal_width()
    print(colored(heading + "." * (width - len(heading)), color))
    if warp:
        text = text.split("\n")
        text = [wrap_text(t) for t in text]
        text = "\n".join(text)
        print(text)
        # print(wrap_text(text))
    else:
        print(text)
    print("." * width)


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def get_run_id():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
