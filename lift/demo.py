import os

import torch
import einops

from lift.dinov2 import (
    DINOv2ForVideo,
    make_classification_eval_transform,
    compute_dino_features_for_single_video,
    compute_dino_embeddings_for_videos,
)
from lift.model import prepare_model_lift, gather_latents
from lift.model import load_checkpoint as load_lift_checkpoint
import shared.utils as su
from lift.viz_utils import show_trajectory_with_reconstruction


def load_lift_module(
    latent_dim=384,
    # LiFT with DINOv2
    feature_dim=384,
    max_len=5000,
    ckpt_root="/work/piyush/experiments/TimeBound.v1/time-antonyms/",
    ckpt_name="ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
):
    
    args = {
        "model": "lift",
        "encoder": "DualCLSTransformer",
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "max_len": max_len,
    }
    args = su.misc.AttrDict(args)
    litmodule = prepare_model_lift(args)
    
    litmodule = load_lift_checkpoint(
        litmodule,
        ckpt_root=ckpt_root,
        ckpt_name=ckpt_name,
    )
    litmodule = litmodule.eval()
    return litmodule


def compute_lift_embeddings(base_embeds, model, t=16, unflatten=False, device=None):
    if unflatten:
        base_embeds = einops.rearrange(base_embeds, "b (t d) -> b t d", t=t)
    lift_output = gather_latents(
        model, base_embeds.to(device), reconstruct=True,
    )
    lift_output = {k: v.cpu() for k, v in lift_output.items()}
    return lift_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=str, default="/work/piyush/experiments/TimeBound.v1/time-antonyms/")
    parser.add_argument("--ckpt_name", type=str, default="ggwirp95/checkpoints/epoch=458-step=834003.ckpt")
    args = parser.parse_args()
    ckpt_root = args.ckpt_root
    ckpt_name = args.ckpt_name

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("-" * 100)
    
    su.log.print_update("Loading backbone: DINOv2ForVideo model.", color='yellow')
    backbone = DINOv2ForVideo(model_id='vit_small_patch14_reg4_dinov2.lvd142m')
    backbone = backbone.to(device)
    preprocess = make_classification_eval_transform()
    su.misc.num_params(backbone)
    su.log.print_update(".", color='yellow')
    
    # Load LiFT
    lift_module = load_lift_module(ckpt_root=ckpt_root, ckpt_name=ckpt_name).to(device)


    # Sample video
    su.log.print_update("Demonstrating LiFT on a single video.", color='green')
    video_path = "./assets/76198.webm"
    print(f"Processing video: {video_path}")
    # video_path = "./sample_data/folding_paper.mp4"
    assert os.path.exists(video_path), \
        f"Video file {video_path} not found"
    frames, _, X = compute_dino_features_for_single_video(
        video_path, preprocess, backbone, return_frames=True, device=device, n_frames=16,
    )
    print("Per-frame DINO features: ", X.shape)

    # Compute LiFT features
    lift_output = compute_lift_embeddings(X.unsqueeze(0), lift_module, device=device)
    print("LiFT features: ")
    print(lift_output.keys())
    for k, v in lift_output.items():
        print(k, v.shape)

    # Visualize DINO trajectory and LiFT reconstruction
    image = show_trajectory_with_reconstruction(
        video_path=video_path,
        x=X,
        x_hat=lift_output["reconstructed"].squeeze(0),
        class_name="moving something down",
        method="tsne",
        joint_dimred=True,
        return_img=True,
    )
    image.save("lift_output.png")
    print("Saved visualization to lift_output.png. \n")


    # Demonstrate how to compute LiFT embeddings for a batch of videos
    su.log.print_update("Demonstrating LiFT on a batch of videos.", color='magenta')
    video_paths = [
        "./assets/76198.webm",
        "./assets/folding_paper.mp4",
        "./assets/horsejump-high.mp4",
    ]
    _, X = compute_dino_embeddings_for_videos(
        video_paths, preprocess, backbone, batch_size=3,
    )
    print("Per-frame DINO features: ", X.shape)
    lift_output = compute_lift_embeddings(X, lift_module, device=device)
    Z = lift_output["concat"]
    print("LiFT embeddings: ", Z.shape)
    print(lift_output.keys())
    for k, v in lift_output.items():
        print(k, v.shape)
    print("-" * 100)
