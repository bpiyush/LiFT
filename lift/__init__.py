"""LiFT: Linearized Feature Trajectories for Time-Aware Video Representation Learning.

This package provides the LiFT model for learning time-aware video representations.
LiFT transforms non-linear DINOv2 trajectories into compact video embeddings.

Example usage:
    from lift import LiFT
    
    # Load pretrained model
    model = LiFT.from_pretrained(ckpt_path="path/to/checkpoint.ckpt")
    
    # Or use the convenience functions
    from lift.demo import load_lift_module, compute_lift_embeddings
"""

__version__ = "0.1.0"
__author__ = "Piyush Bagad"

from lift.model import (
    LinearisedFeatureTrajectories,
    LiFTLightningModule,
    load_checkpoint,
    gather_latents,
    prepare_model_lift,
)

from lift.dinov2 import (
    DINOv2ForVideo,
    make_classification_eval_transform,
    compute_dino_features_for_single_video,
    compute_dino_embeddings_for_videos,
)

from lift.demo import (
    load_lift_module,
    compute_lift_embeddings,
)

# Convenience alias
LiFT = LinearisedFeatureTrajectories

__all__ = [
    # Core model
    "LiFT",
    "LinearisedFeatureTrajectories",
    "LiFTLightningModule",
    "load_checkpoint",
    "gather_latents",
    "prepare_model_lift",
    # DINOv2 backbone
    "DINOv2ForVideo",
    "make_classification_eval_transform",
    "compute_dino_features_for_single_video",
    "compute_dino_embeddings_for_videos",
    # Demo/convenience functions
    "load_lift_module",
    "compute_lift_embeddings",
]

