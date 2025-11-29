"""Helper functions for visualizing trajectories and reconstructions."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import shared.utils as su


def show_trajectory(
    frames, x, class_name, kappa=None,
    method="pca", colorbar=False, fig=None, axes=None, scatter=False, cmap="Reds_r",
    return_as_pil=False,
):

    # Plot tsne
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 5.2), height_ratios=[0.23, 0.77])
    
    ax = axes[0]
    ax.axis("off")
    show_frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 8, dtype=int)]
    ax.imshow(su.visualize.concat_images_with_border(show_frames))
    ax.set_aspect('auto')
    if kappa is not None:
        ax.set_title(class_name + f"({kappa:.2f})", fontsize=10)
    else:
        ax.set_title(class_name, fontsize=10)
    
    ax = axes[1]
    su.visualize.show_temporal_tsne(
        x,
        tsne_kwargs=dict(method=method, perplexity=min(25., len(x) - 1.)),
        ax=ax,
        fig=fig,
        show=False,
        scatter=scatter,
        cmap=cmap,
        colorbar=colorbar,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    if return_as_pil:
        # Need to return the PIL image
        fig.canvas.draw()
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(),
        )
        plt.close(fig)
        return pil_image
    
    if fig is None or axes is None:
        plt.show()
    else:
        return fig, axes


def show_selfsimilarity(x, return_as_pil=False):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    ax.imshow(x @ x.T)
    ax.axis("off")
    if return_as_pil:
        fig.canvas.draw()
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(),
        )
        plt.close(fig)
        return pil_image
    plt.show()


def show_trajectory_with_reconstruction(
        # frames,
        video_path,
        x,
        x_hat,
        class_name,
        kappa=None, method="tsne", joint_dimred=False, return_img=False,
    ):

    # Load frames
    frames = su.video.load_frames_linspace(video_path, n=len(x_hat))

    # Apply dimred to x and x_hat together
    if joint_dimred:
        x = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
        x_hat = torch.from_numpy(x_hat) if not isinstance(x_hat, torch.Tensor) else x_hat
        x = x - x.mean(0)
        x_hat = x_hat - x_hat.mean(0)
        x_combined = torch.cat([x, x_hat], axis=0)
        x_combined = su.visualize.reduce_dim(x_combined, method=method)
        x = x_combined[:len(x)]
        x_hat = x_combined[len(x):]

    # Plot tsne
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.1), height_ratios=[0.25, 0.75])

    ax = axes[0]
    ax.axis("off")
    show_frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 8, dtype=int)]
    ax.imshow(su.visualize.concat_images_with_border(show_frames, border_width=20))
    ax.set_aspect('auto')
    if kappa is not None:
        ax.set_title(class_name + f"({kappa:.2f})", fontsize=10)
    else:
        ax.set_title(class_name, fontsize=10)
    
    ax = axes[1]
    # Plot original
    su.visualize.show_temporal_tsne(
        x,
        tsne_kwargs=dict(method=method),
        ax=ax,
        fig=fig,
        show=False,
        scatter=False,
        cmap="Reds_r",
        # cmap="gray",
        plot_label="Original",
        colorbar=False,
    )
    # Plot reconstruction
    su.visualize.show_temporal_tsne(
        x_hat,
        tsne_kwargs=dict(method=method),
        ax=ax,
        fig=fig,
        show=False,
        scatter=False,
        colorbar=False,
        cmap="Blues_r",
        plot_label="Reconstruction",
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    if not return_img:
        plt.show()
    else:
        # Convert to PIL Image and return
        from PIL import Image
        fig.canvas.draw()
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(),
        )
        return pil_image
