"""Linearised Feature Trajectories (LiFT)."""
import os
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L

from lift.layers import (
    TransformerEncoderLayer,
    TransformerEncoder,
)
from lift.feat_utils import (
    get_linear_probe_accuracy,
)
import shared.utils as su


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input
    """
    def __init__(self, d_model, max_len=16):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * \
                (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        t = x.size(1)
        if t == self.max_len:
            return x + self.pe
        elif t > self.max_len:
            # Need to interpolate
            pe = self.pe
            pe = F.interpolate(pe, size=(t,), mode='linear', align_corners=False)
            return x + pe
        else:
            # Need to truncate
            pe = self.pe[:, :t]
            return x + pe


class DualCLSTransformer(nn.Module):
    """
    Transformer encoder with two CLS tokens: one for static and one for dynamic information
    """
    def __init__(self, feature_dim, hidden_dim, nhead=8, num_layers=4, dropout=0.1, max_len=16):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        
        # CLS tokens (static and dynamic)
        self.cls_static = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_dynamic = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        
        # Output projections for latent codes
        self.static_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for outputs
        self.static_norm = nn.LayerNorm(hidden_dim)
        self.dynamic_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, need_weights=False):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static latent code [batch_size, hidden_dim]
            z_d: Dynamic latent code [batch_size, hidden_dim]
        """
        # TODO: add sinusoidal position encoding to represent time
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Expand CLS tokens to batch size
        cls_s = self.cls_static.expand(batch_size, -1, -1)
        cls_d = self.cls_dynamic.expand(batch_size, -1, -1)
        
        # Concatenate CLS tokens with input sequence
        # [batch_size, 2+seq_len, hidden_dim]
        x_with_cls = torch.cat([cls_s, cls_d, x], dim=1)
        
        # Pass through transformer
        if need_weights:
            output, attn_maps = self.transformer(x_with_cls, need_weights=True)
        else:
            output = self.transformer(x_with_cls, need_weights=False)
        
        # Extract and process the CLS token outputs
        z_s = self.static_norm(self.static_proj(output[:, 0]))
        z_d = self.dynamic_norm(self.dynamic_proj(output[:, 1]))
        
        if not need_weights:
            return z_s, z_d
        else:
            return z_s, z_d, attn_maps


class DualCLSTransformerWithoutProjection(nn.Module):
    """
    Transformer encoder with two CLS tokens: one for static and one for dynamic information
    """
    def __init__(self, feature_dim, hidden_dim=None, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # Positional encoding
        hidden_dim = feature_dim
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # CLS tokens (static and dynamic)
        self.cls_static = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_dynamic = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Layer normalization for outputs
        self.static_norm = nn.LayerNorm(hidden_dim)
        self.dynamic_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, need_weights=False):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static latent code [batch_size, hidden_dim]
            z_d: Dynamic latent code [batch_size, hidden_dim]
        """
        # TODO: add sinusoidal position encoding to represent time
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        x = self.pos_encoder(x)

        # Expand CLS tokens to batch size
        cls_s = self.cls_static.expand(batch_size, -1, -1)
        cls_d = self.cls_dynamic.expand(batch_size, -1, -1)

        # Concatenate CLS tokens with input sequence
        # [batch_size, 2+seq_len, hidden_dim]
        x_with_cls = torch.cat([cls_s, cls_d, x], dim=1)

        # Pass through transformer
        if need_weights:
            output, attn_maps = self.transformer(x_with_cls, need_weights=True)
        else:
            output = self.transformer(x_with_cls, need_weights=False)
        
        # Extract and process the CLS token outputs
        z_s = self.static_norm(output[:, 0])
        z_d = self.dynamic_norm(output[:, 1])

        if not need_weights:
            return z_s, z_d
        else:
            return z_s, z_d, attn_maps


class Decoder(nn.Module):
    """
    Simple decoder that reconstructs features from the latent representation
    """
    def __init__(self, latent_dim, feature_dim, hidden_dim=512):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent vector at time t [batch_size, latent_dim]
        Returns:
            x_hat: Reconstructed feature [batch_size, feature_dim]
        """
        return self.model(z)


class LinearisedFeatureTrajectories(nn.Module):
    """
    Complete model for video linearization
    """
    def __init__(
        self,
        encoder="DualCLSTransformer",
        feature_dim=384,
        latent_dim=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=16,
    ):
        super().__init__()
        
        self.encoder = eval(encoder)(
            feature_dim=feature_dim,
            hidden_dim=latent_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.feature_dim = feature_dim
        self.embed_dim = latent_dim * 2

        self.decoder = Decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            hidden_dim=latent_dim * 2
        )
    
    def encode(self, x, need_weights=False, cat=False):
        """
        Encode sequence into static and dynamic components
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
        """
        z_s, z_d = self.encoder(x, need_weights=need_weights)
        if cat:
            z = torch.cat([z_s, z_d], dim=-1)
            return z
        else:
            return z_s, z_d
    
    def compute_latents(self, x):
        """
        Compute static and dynamic latents
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
        Returns:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
        """
        z_s, z_d = self.encode(x, need_weights=False)

        # Concatenate static and dynamic components
        z = torch.cat([z_s, z_d], dim=-1)

        return z_s, z_d, z
    
    def interpolate(self, z_s, z_d, t, T=1.0):
        """
        Interpolate in latent space using the linear model
        
        Args:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
            t: Normalized time value or tensor of time values [batch_size]
            T: Total sequence length for normalization (default=1.0)
        Returns:
            z_t: Interpolated latent [batch_size, latent_dim]
        """
        # Handle scalar t
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z_s.device)
        
        # Handle scalar t with batched inputs
        if t.dim() == 0 and z_s.dim() > 1:
            t = t.expand(z_s.size(0))
        
        # Normalize t and reshape for broadcasting
        t_norm = (t / T).view(-1, 1)
        
        # Linear interpolation
        z_t = z_s + t_norm * z_d
        
        return z_t
    
    def decode(self, z_t):
        """
        Decode latent vector to feature
        
        Args:
            z_t: Latent vector at time t [batch_size, latent_dim]
        Returns:
            x_hat: Reconstructed feature [batch_size, feature_dim]
        """
        return self.decoder(z_t)
    
    def forward(self, x, times=None, need_weights=False):
        """
        Full forward pass
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
            times: Optional specific time points to reconstruct.
                   If None, reconstructs all input frames.
            need_weights: Whether to return attention weights
        Returns:
            x_hat: Reconstructed features
            z_s: Static latent code
            z_d: Dynamic latent code
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Encode sequence
        if need_weights:
            z_s, z_d, attn_maps = self.encode(x, need_weights=True)
        else:
            z_s, z_d = self.encode(x, need_weights=False)
        
        # Determine which time points to reconstruct
        if times is None:
            times = torch.arange(seq_len, device=x.device).float()
            times = times.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
            times_flat = times.flatten()  # [batch_size * seq_len]
            
            # Repeat z_s and z_d for each time point
            z_s_rep = z_s.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, z_s.size(-1))
            z_d_rep = z_d.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, z_d.size(-1))
            
            # Interpolate and decode
            z_t = self.interpolate(z_s_rep, z_d_rep, times_flat, T=seq_len-1)
            x_hat = self.decode(z_t)
            
            # Reshape output
            x_hat = x_hat.view(batch_size, seq_len, feature_dim)
        else:
            # Handle custom time points
            if not torch.is_tensor(times):
                times = torch.tensor(times, device=x.device).float()
            
            # Ensure times has the right shape
            if times.dim() == 1:
                times = times.unsqueeze(0).expand(batch_size, -1)
            
            times_flat = times.flatten()
            num_times = times.size(1)
            
            # Repeat z_s and z_d for each time point
            z_s_rep = z_s.unsqueeze(1).expand(-1, num_times, -1).reshape(-1, z_s.size(-1))
            z_d_rep = z_d.unsqueeze(1).expand(-1, num_times, -1).reshape(-1, z_d.size(-1))
            
            # Interpolate and decode
            z_t = self.interpolate(z_s_rep, z_d_rep, times_flat, T=seq_len-1)
            x_hat = self.decode(z_t)
            
            # Reshape output
            x_hat = x_hat.view(batch_size, num_times, feature_dim)
        
        if need_weights:
            return x_hat, z_s, z_d, attn_maps
        else:
            return x_hat, z_s, z_d


class SimplifiedLoss(nn.Module):
    """
    Simplified loss function with just reconstruction and orthogonality losses
    """
    def __init__(self, recon_weight=1.0, ortho_weight=0.1, fotd_weight=0.):
        super().__init__()
        self.recon_weight = recon_weight
        self.ortho_weight = ortho_weight
        self.fotd_weight = fotd_weight
        print("Loss weights:")
        print(f"  Reconstruction: {self.recon_weight}")
        print(f"  Orthogonality: {self.ortho_weight}")
        print(f"  First-order temporal difference: {self.fotd_weight}")
        
    def reconstruction_loss(self, x_hat, x):
        """Mean squared error reconstruction loss"""
        return F.mse_loss(x_hat, x)
    
    def first_order_temporal_difference_loss(self, x_hat, x):
        """
        Applies a first-order temporal difference loss.

            delta_X = X[1:, :] - X[:-1, :] (Shape: (T-1) x D)
            delta_X_hat = X_hat[1:, :] - X_hat[:-1, :] (Shape: (T-1) x D)
            loss = MSE(delta_X_hat, delta_X)
        """
        delta_x = x[1:, :] - x[:-1, :]
        delta_x_hat = x_hat[1:, :] - x_hat[:-1, :]
        return F.mse_loss(delta_x_hat, delta_x)
    
    def orthogonality_loss(self, z_s, z_d):
        """Normalized dot product between static and dynamic components"""
        z_s_norm = F.normalize(z_s, dim=1)
        z_d_norm = F.normalize(z_d, dim=1)
        cos_sim = torch.abs(torch.sum(z_s_norm * z_d_norm, dim=1)).mean()
        return cos_sim
    
    def forward(self, x, x_hat, z_s, z_d):
        """
        Calculate the combined loss
        
        Args:
            x: Original features [batch_size, seq_len, feature_dim]
            x_hat: Reconstructed features [batch_size, seq_len, feature_dim]
            z_s: Static latent code [batch_size, latent_dim]
            z_d: Dynamic latent code [batch_size, latent_dim]
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        recon_loss = self.reconstruction_loss(x_hat, x)
        ortho_loss = self.orthogonality_loss(z_s, z_d)
        fotd_loss = self.first_order_temporal_difference_loss(x_hat, x)
        
        # Combine losses
        total_loss = self.recon_weight * recon_loss + \
            self.ortho_weight * ortho_loss + \
            self.fotd_weight * fotd_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'ortho': ortho_loss.item(),
            'fotd': fotd_loss.item(),
        }
        
        return total_loss, loss_dict


class LiFTLightningModule(L.LightningModule):
    def __init__(
            self,
            model,
            opt_name="adam",
            lr=1e-4,
            sched_name="plateau",
            loss_weights=dict(
                recon_weight=1.0,
                ortho_weight=0.1,
                fotd_weight=0.,
            ),
            show_traj=True,
            no_wandb=False,
            loss="SimplifiedLoss",
        ):
        super().__init__()

        self.model = model
        self.opt_name = opt_name
        self.sched_name = sched_name
        self.lr = lr
        self.criterion = eval(loss)(
            **loss_weights,
        )
        self.show_traj = show_traj
        self.no_wandb = no_wandb
    
        # To store outputs
        # NOTE: no need to store train outputs since the 
        # train and eval for linear probe is all in the 
        # validation set itself.
        # self.train_step_outputs = defaultdict(list)
        self.valid_step_outputs = defaultdict(list)

    def configure_optimizers(self):

        # Tried other optimizers like SGD, but Adam works best
        if self.opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=1e-5,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_name}")

        if self.sched_name != "none":
        
            # Learning rate scheduler
            # Tried other schedulers like CosineAnnealingLR, but
            # ReduceLROnPlateau works best
            if self.sched_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                )
            else:
                raise ValueError(f"Unknown scheduler: {self.sched_name}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/loss",
            }

        elif self.sched_name == "none":
            return optimizer

    def process_batch(self, batch, return_data=False):

        # Get inputs
        x = batch['features']

        # Forward pass
        x_hat, z_s, z_d = self.model(x)

        # Calculate loss
        loss, loss_dict = self.criterion(x, x_hat, z_s, z_d)

        # Return outputs
        output = dict(
            latents=torch.cat([z_s, z_d], dim=-1),
            loss=loss,
            chiral_triplet_ids=batch["chiral_triplet_id"],
            chiral_labels=batch["chiral_label"],
            linprobe_split=batch["linprobe_split"],
            **loss_dict,
        )

        # If return_data is True, return the data as well
        if return_data:
            output["original"] = x
            output["reconstructed"] = x_hat

        return output

    def training_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)
        loss = outputs['loss']

        # Log loss
        self.log(f"train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.process_batch(batch, return_data=(batch_idx == 0))
        loss = outputs['loss']

        # Log loss
        self.log(f"valid/loss", outputs['loss'], prog_bar=True)

        # Log visualisation of trajectories for few videos
        # only for the first batch
        if (batch_idx == 0) and self.show_traj:
            from adapt4change.playground.expt_utils import (
                show_feature_trajectories,
            )
            n_show = 4
            images = [
                show_feature_trajectories(
                    video_path=batch['video_path'][i],
                    x=outputs["original"][i].detach().cpu().numpy(),
                    x_hat=outputs["reconstructed"][i].detach().cpu().numpy(),
                ) for i in range(n_show)
            ]
            canvas = su.visualize.concat_images_with_border(images)

            # Log image
            if not self.no_wandb:
                import wandb
                self.logger.experiment.log(
                    {
                        "valid/feature_trajectories": wandb.Image(canvas),
                    },
            )
                
            del outputs["original"]
            del outputs["reconstructed"]

        # Add to validation outputs
        del outputs["loss"]
        del outputs['total']
        del outputs["recon"]
        del outputs["ortho"]
        del outputs["fotd"]
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                outputs[k] = v.detach().cpu().numpy()
            self.valid_step_outputs[k].append(outputs[k])

        return loss

    def on_validation_epoch_end(self):

        # Concatenate all outputs
        valid_step_outputs_all = {
            k: np.concatenate(v, axis=0) for k, v in self.valid_step_outputs.items()
        }

        # Get train and eval indices
        split_info = np.array(valid_step_outputs_all['linprobe_split'])
        train_indices = np.where(split_info == "train")[0]
        valid_indices = np.where(split_info == "validation")[0]

        train_step_outputs_all = {
            k: v[train_indices] for k, v in valid_step_outputs_all.items()
        }
        valid_step_outputs_all = {
            k: v[valid_indices] for k, v in valid_step_outputs_all.items()
        }

        # Run linear probe on each triplet subset
        triplets_train = train_step_outputs_all["chiral_triplet_ids"]
        triplets_valid = valid_step_outputs_all["chiral_triplet_ids"]
        triplets_common = np.intersect1d(
            np.unique(triplets_train), np.unique(triplets_valid)
        )
        val_accs = []
        for tid in triplets_common:
            idx_train = np.where(triplets_train == tid)[0]
            idx_valid = np.where(triplets_valid == tid)[0]
            Z_train = train_step_outputs_all["latents"][idx_train]
            Z_valid = valid_step_outputs_all["latents"][idx_valid]
            Y_train = train_step_outputs_all["chiral_labels"][idx_train]
            Y_valid = valid_step_outputs_all["chiral_labels"][idx_valid]
            val_acc = get_linear_probe_accuracy(
                Z_train, Y_train, Z_valid, Y_valid, verbose=False,
            )
            val_accs.append(val_acc)
        val_acc = np.mean(val_accs)
        self.log(
            "valid/linear_probe_acc", val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Free up memory
        del self.valid_step_outputs
        self.valid_step_outputs = defaultdict(list)


def load_checkpoint(
        litmodule,
        ckpt_name="ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
        ckpt_root="/work/piyush/experiments/TimeBound.v1/time-antonyms/",
    ):
        """
        Load a checkpoint into a LightningModule.
        """
        ckpt_path = os.path.join(ckpt_root, ckpt_name)
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}."
        print("::: Loading checkpoint from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        msg = litmodule.load_state_dict(ckpt["state_dict"])
        print(msg)
        return litmodule


def load_pretrained_model(
        ckpt_name="ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
        latent_dim=384,
        feature_dim=384,
        hidden_dim=128,
        load_pretrained=True,
        max_len=16,
        device=None,
    ):
    """
    Get video embedding from a sequence compressor model.

    Args:
        ckpt_name: str
            (sub) Path to the checkpoint file.
        latent_dim: int
            Dimension of the latent space.
    """

    args = dict(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=1e-3,
    )
    args = su.misc.AttrDict(args)

    model = LinearisedFeatureTrajectories(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=getattr(args, "max_len", max_len),
    )
    su.misc.num_trainable_params(model)

    # Initialise Lightning Module
    litmodule = LiFTLightningModule(
        model=model, lr=args.lr, no_wandb=True,
    )

    # Load pre-trained weights
    if load_pretrained:
        ckpt_root = "/work/piyush/experiments/TimeBound.v1/time-antonyms/"
        ckpt_path = os.path.join(ckpt_root, ckpt_name)
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}."
        print("::: Loading checkpoint from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        msg = litmodule.load_state_dict(ckpt["state_dict"])
        print(msg)

    # Port to GPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    litmodule = litmodule.to(device).eval()

    return litmodule


def gather_latents(litmodule, X, batch_size=1024, reconstruct=False, verbose=True, device=None, to_cpu=True):
    """Helper to run feature computation for litmodule."""
    if device is None:
        device = next(litmodule.model.parameters()).device
    else:
        litmodule = litmodule.to(device).eval()
    start_indices = np.arange(0, len(X), batch_size)
    if verbose:
        iterator = su.log.tqdm_iterator(start_indices, desc="Gathering features")
    else:
        iterator = start_indices
    latents = {
        "static": [],
        "dynamic": [],
        "concat": [],
    }
    if reconstruct:
        latents["reconstructed"] = []
    for si in iterator:
        ei = min(si + batch_size, len(X))
        with torch.no_grad():

            # Reconstruction as well as latents
            x_hat, zs, zd = litmodule.model(X[si:ei].to(device))
            z = torch.cat([zs, zd], dim=-1)

            # # Only latents
            # zs, zd, z = litmodule.model.compute_latents(X[si:ei].to(device))

            if to_cpu:
                zs = zs.cpu()
                zd = zd.cpu()
                z = z.cpu()
                x_hat = x_hat.cpu()

        latents["static"].append(zs)
        latents["dynamic"].append(zd)
        latents["concat"].append(z)
        if reconstruct:
            latents["reconstructed"].append(x_hat)
    latents = {k: torch.cat(v) for k, v in latents.items()}
    return latents


def prepare_model_lift(args):
    su.log.print_update("Creating model: LinearisedFeatureTrajectories (LiFT)", color="cyan")
    print("NOTE: Using max_len from args", getattr(args, "max_len", 16))
    model = LinearisedFeatureTrajectories(
        encoder=args.encoder,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=getattr(args, "max_len", 16),
    )
    su.misc.num_trainable_params(model)
    litmodule = LiFTLightningModule(
        model=model,
        lr=getattr(args, "lr", 1e-4),
        sched_name=getattr(args, "sched_name", "plateau"),
        opt_name=getattr(args, "opt_name", "adam"),
        loss_weights=dict(
            recon_weight=getattr(args, "recon_weight", 1.0),
            ortho_weight=getattr(args, "ortho_weight", 0.1),
            fotd_weight=getattr(args, "fotd_weight", 0.),
        ),
        no_wandb=getattr(args, "no_wandb", True),
        loss=getattr(args, "loss", "SimplifiedLoss"),
    )
    su.log.print_update("", color="cyan")
    return litmodule
