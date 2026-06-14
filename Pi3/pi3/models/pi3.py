import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.refined_point_head import RefinedLinearPts3d
from .layers.conv_point_head import ConvLinearPts3d, SimpleConvPts3d
from .layers.camera_head import CameraHead
from .layers.autoregressive_transformer import AutoregressiveTokenTransformer
from .dinov2.hub.backbones import dinov2_vitl14_reg


class LFG(nn.Module):
    """
    Single-view LFG model for current-frame geometry and future-frame prediction.
    """
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            encoder_name='dinov2',
            n_future_frames=3,
            ar_n_heads=16,
            ar_n_layers=8,
            ar_dropout=0.1,
            use_segmentation_head=False,  # Enable segmentation head
            segmentation_num_classes=6,  # Number of segmentation classes
            use_motion_head=True,  # Enable motion head
            use_flow_head=False,  # Enable optical flow head
            point_head_type='linear',  # Options: 'linear', 'refined', 'conv', 'simple_conv'
            point_head_config=None,  # Config for the point head
            pretrained_encoder=False,  # Checkpoints normally include encoder weights.
        ):
        super().__init__()
        # point_head_type = 'simple_conv'
        self.use_segmentation_head = use_segmentation_head
        self.use_motion_head = use_motion_head
        self.use_flow_head = use_flow_head
        self.segmentation_num_classes = segmentation_num_classes
        self.point_head_type = point_head_type
        self.point_head_config = point_head_config or {}

        # ----------------------
        #        Encoder
        # ----------------------
        if encoder_name == 'dinov2':
            self.encoder = dinov2_vitl14_reg(pretrained=pretrained_encoder)
            self.patch_size = 14
            del self.encoder.mask_token
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not implemented")

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        
        # VGGT style alternating attention heads
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        # Autoregressive Transformer
        # ----------------------
        self.n_future_frames = n_future_frames
        self.autoregressive_transformer = AutoregressiveTokenTransformer(
            d_model=2 * self.dec_embed_dim,  # Concatenated features from decode
            n_heads=ar_n_heads,
            n_layers=ar_n_layers,
            d_ff=3 * self.dec_embed_dim,
            dropout=ar_dropout,
            n_future_frames=n_future_frames,
            max_seq_len=15  # Can handle up to 15 frames total
        )

        # ----------------------
        #  Task-specific Decoders
        # ----------------------
        # Point decoder
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        # Initialize point head based on type
        if self.point_head_type == 'linear':
            self.point_head = LinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3
            )
        elif self.point_head_type == 'refined':
            self.point_head = RefinedLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3,
                **self.point_head_config
            )
        elif self.point_head_type == 'conv':
            self.point_head = ConvLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3,
                **self.point_head_config
            )
        elif self.point_head_type == 'simple_conv':
            self.point_head = SimpleConvPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3,
                **self.point_head_config
            )
        else:
            raise ValueError(f"Unknown point head type: {self.point_head_type}")

        # Confidence decoder
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size, 
            dec_embed_dim=1024, 
            output_dim=1
        )

        # Camera pose decoder
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # Segmentation decoder and head (optional)
        if self.use_segmentation_head:
            self.segmentation_decoder = deepcopy(self.point_decoder)
            self.segmentation_head = LinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=self.segmentation_num_classes,
            )
        
        # Motion decoder and head (optional)
        if self.use_motion_head:
            self.motion_decoder = deepcopy(self.point_decoder)
            self.motion_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=1,  # Binary motion mask (0=static, 1=moving)
            )

        # Flow decoder and head (optional)
        if self.use_flow_head:
            self.flow_decoder = deepcopy(self.point_decoder)
            self.flow_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=2,  # 2D optical flow (dx, dy)
            )

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        """Same as original Pi3 decode function"""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, n_future_frames_override=None):
        """
        Forward pass for LFG.

        Args:
            imgs: Input images [B, N, C, H, W]
            n_future_frames_override: Optional override for n_future_frames.
                                     Set to 0 for current-frames-only mode (no AR work)
        """
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Encode images
        imgs = imgs.reshape(B*N, _, H, W)
        if hasattr(self.encoder, 'forward_features'):
            hidden = self.encoder.forward_features(imgs)
        else:
            hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        dino_features = hidden
        # Decode and aggregate spatial-temporal features
        hidden, pos = self.decode(hidden, N, H, W)
        pi3_features = hidden

        # Generate future tokens autoregressively
        all_hidden, all_pos = self.autoregressive_transformer(hidden, N, pos, n_future_frames_override=n_future_frames_override)
        autonomy_features = all_hidden  # [B*(N+M), S, D] - all features including future frames from AR transformer

        # Update frame count to include future frames
        n_future = n_future_frames_override if n_future_frames_override is not None else self.n_future_frames
        total_frames = N + n_future

        # Process all tokens (current + future) through task decoders
        point_hidden = self.point_decoder(all_hidden, xpos=all_pos)
        conf_hidden = self.conf_decoder(all_hidden, xpos=all_pos)
        camera_hidden = self.camera_decoder(all_hidden, xpos=all_pos)

        # lets get point, conf, cam features for all frames (current + future)
        point_features = point_hidden
        conf_features = conf_hidden
        camera_features = camera_hidden
        
        if self.use_segmentation_head:
            segmentation_hidden = self.segmentation_decoder(all_hidden, xpos=all_pos)
        
        if self.use_motion_head:
            motion_hidden = self.motion_decoder(all_hidden, xpos=all_pos)

        if self.use_flow_head:
            flow_hidden = self.flow_decoder(all_hidden, xpos=all_pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Points - [B*total_frames, H, W, 3]
            point_hidden = point_hidden.float()
            # point_hidden = modality_hidden.float()

            local_points_flat = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W))
            local_points_raw = local_points_flat.reshape(B, total_frames, H, W, -1)
            
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # Confidence - [B*total_frames, H, W, 1]
            # conf_hidden = modality_hidden.float()
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W))
            conf = conf_flat.reshape(B, total_frames, H, W, -1)

            # Camera poses - [B*total_frames, 4, 4]
            # camera_hidden = modality_hidden.float()
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w)
            camera_poses = camera_poses_flat.reshape(B, total_frames, 4, 4)

            # Segmentation - [B*total_frames, H, W, 6]
            if self.use_segmentation_head:
                segmentation_hidden = segmentation_hidden.float()
                segmentation_flat = self.segmentation_head([segmentation_hidden[:, self.patch_start_idx:]], (H, W))
                segmentation = segmentation_flat.reshape(B, total_frames, H, W, -1)
            else:
                segmentation = None
            
            # Motion - [B*total_frames, H, W, 1]
            if self.use_motion_head:
                motion_hidden = motion_hidden.float()
                motion_flat = self.motion_head([motion_hidden[:, self.patch_start_idx:]], (H, W))
                motion = motion_flat.reshape(B, total_frames, H, W, -1)  # Binary motion masks
            else:
                motion = None

            # Flow - [B*total_frames, H, W, 2]
            if self.use_flow_head:
                flow_hidden = flow_hidden.float()
                flow_flat = self.flow_head([flow_hidden[:, self.patch_start_idx:]], (H, W))
                flow = flow_flat.reshape(B, total_frames, H, W, -1)  # 2D optical flow (dx, dy)
            else:
                flow = None

            # Unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        result = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            n_current_frames=N,
            n_future_frames=n_future,  # Actual n_future used (may be overridden)
            dino_features=dino_features,  # [B*N, S, D] - DINOv2 encoder features for potential supervision
            pi3_features=pi3_features,    # [B*N, S, D]
            autonomy_features=autonomy_features,  # [B*(N+M), S, D] - all features including future frames from AR transformer,
            point_features=point_features,  # [B*(N+M), S, D]
            conf_features=conf_features,    # [B*(N+M), S, D
            camera_features=camera_features   # [B*(N+M), S, D]
        )
        
        if self.use_segmentation_head:
            result['segmentation'] = segmentation
        
        if self.use_motion_head:
            result['motion'] = motion
        
        if self.use_flow_head:
            result['flow'] = flow
        
        # Always include decoder features for potential supervision
        result['all_decoder_features'] = all_hidden  # Current + future decoder features [B*(N+M), S, D]
        result['all_positional_encoding'] = all_pos  # [B*(N+M), S, pos_dim]
            
        return result
