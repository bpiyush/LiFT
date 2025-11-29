
"""Defines DINOv2 for videos (per-frame)."""
import torch
import einops
import timm
import numpy as np
from typing import Sequence
from torchvision import transforms
from shared.utils.log import tqdm_iterator
from shared.utils.video import load_frames_linspace


def dinov2_with_registers(model_id='vit_base_patch14_reg4_dinov2.lvd142m'):
    image_backbone = timm.create_model(
        model_id,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    return image_backbone


class DINOv2ForVideo(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.model = dinov2_with_registers(model_id=model_id)
        self.embed_dim = self.model.embed_dim
        self.num_heads = 12
    
    def forward(self, videos):
        """
        Args:
            videos (torch.Tensor): Shape (B, C, T, H, W).
        """
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        z = self.model.forward(images)
        z = einops.rearrange(z, '(b t) d -> b t d', b=b)
        return z
    
    def forward_dense(self, videos):
        """Outputs dense DINO features."""
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        # NOTE: this needs adding patch to timm to get the dense features
        # This is not used currently.
        assert self.model_id.startswith("dense_"), \
            "Use the dense model for dense features."
        # [n c h w] -> [n d h' w']
        z = self.model.get_intermediate_layers(images, reshape=True)[0]
        z = einops.rearrange(z, '(b t) ... -> b t ...', b=b)
        return z
    
    def forward_cls_and_registers(self, videos, pool_registers="concat"):
        """Outputs CLS token and register features."""
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        assert "reg" in self.model_id, "Model should have registers."
        z = self.model.get_intermediate_layers(images, return_prefix_tokens=True)[0][1]
        z = einops.rearrange(z, '(b t) ... -> b t ...', b=b)
        if pool_registers == "concat":
            # Concatenate CLS and register tokens
            z = einops.rearrange(z, 'b t p d -> b t (p d)')
        elif pool_registers == "mean":
            # Average the register tokens
            z = z.mean(dim=2)
        else:
            raise ValueError(f"Unknown pool_registers: {pool_registers}")
        return z


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    return transforms.Normalize(mean=mean, std=std)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def compute_dino_embeddings(
    video_paths, preprocess, model, batch_size=32, n_frames=16, verbose=True,
):
    """
    Compute DINO features for a list of videos with batching support.
    
    Args:
        video_paths (list): List of paths to video files
        preprocess: Preprocessing function for frames
        model: DINO model for feature extraction
        batch_size (int): Number of videos to process in each batch (default: 32)
        n_frames (int): Number of frames to sample from each video (default: 16)
        verbose (bool): Whether to show progress bar (default: True)
        
    Returns:
        torch.Tensor: Features for all videos, shape (n_videos, n_frames, feature_dim)
    """
    device = next(model.parameters()).device
    all_features = []
    all_inputs = []
    
    # Create batches of video paths
    n_videos = len(video_paths)
    starts = np.arange(0, n_videos, batch_size)
    
    if verbose:
        iterator = tqdm_iterator(starts, desc=f"Processing {n_videos} videos:")
    else:
        iterator = starts
        
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_videos)
        batch_paths = video_paths[start_idx:end_idx]
        
        # Process each video in the batch
        batch_features = []
        for path in batch_paths:
            # Load and preprocess frames
            frames = load_frames_linspace(path, n=n_frames)
            frames = torch.stack([preprocess(f) for f in frames])
            all_inputs.append(frames)
            batch_features.append(frames)
            
        # Stack frames for the batch
        batch_frames = torch.stack(batch_features)
        
        # Rearrange for model input: (B, T, C, H, W) -> (B, C, T, H, W)
        batch_frames = einops.rearrange(batch_frames, "b t c h w -> b c t h w")
        
        # Move to device and compute features
        with torch.no_grad():
            batch_frames = batch_frames.to(device)
            features = model(batch_frames).cpu()
            
        all_features.append(features)
        
        # Clear GPU memory
        del batch_frames, features
        torch.cuda.empty_cache()
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    
    return all_inputs, all_features


def compute_dino_features_for_single_video(video_path, preprocess, model, n_frames=-1, device=None):
    frames = load_frames_linspace(video_path, n=n_frames)
    x = torch.stack([preprocess(f) for f in frames])
    x = einops.rearrange(x, "t c h w -> 1 c t h w")
    # Move to device and compute features
    with torch.no_grad():
        z = model(x.to(device)).cpu()[0]
    return x, z


if __name__ == "__main__":

    # Test backbone
    print("Loading DINOv2ForVideo model.")
    backbone = DINOv2ForVideo(model_id='vit_base_patch14_reg4_dinov2.lvd142m')
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Number of parameters: {n_params/1e6:.2f}M")
    print("-" * 100)
    
    print("Testing DINOv2ForVideo with random input.")
    x = torch.randn(2, 3, 16, 240, 427)
    z = backbone(x)
    print("Input: ", x.shape)
    print("Output (CLS): ", z.shape)
    print("-" * 100)
    
    # Test on a real video
    print("Testing DINOv2ForVideo on a single video.")
    video_path = "./assets/folding_paper.mp4"
    preprocess = make_classification_eval_transform()
    x, z = compute_dino_features_for_single_video(video_path, preprocess, backbone, n_frames=32)
    print("Input: ", x.shape)
    print("Output (CLS): ", z.shape)
    print("-" * 100)
    
    # Test feature computation on a batch of videos
    print("Testing feature computation on a batch of videos.")
    video_paths = ["./assets/folding_paper.mp4", "./assets/horsejump-high.mp4"]
    x, z = compute_dino_embeddings(video_paths, preprocess, backbone)
    print("Input: ", x.shape)
    print("Output (CLS): ", z.shape)
    print("-" * 100)
