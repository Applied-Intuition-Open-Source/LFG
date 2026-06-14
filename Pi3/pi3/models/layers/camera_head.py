import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


class TemporalCameraMLP(nn.Module):
    def __init__(self, N, M, D):
        super().__init__()
        self.N = N  # current frames
        self.M = M  # future frames  
        self.D = D  # feature dimension
        
        # Current frame processing
        self.current_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        
        # Future frame prediction using temporal modeling
        self.temporal_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(), 
            nn.Linear(D, D * M)
        )
        self.future_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

    def forward(self, x, batch_size, num_current_frames):
        # x shape: [B*N, D] where N is current frames
        BN, D = x.shape
        B, N = batch_size, num_current_frames
        M = self.M
        
        # Current frame processing
        current_feat = self.current_mlp(x)  # [B*N, D]
        
        # Future frame prediction using global context within each batch
        # Reshape [B*N, D] -> [B, N, D] for proper pooling within batches
        x_batched = x.view(B, N, D)
        
        # Pool information within each batch (mean across N current frames)
        global_context = x_batched.mean(dim=1)  # [B, D] - per-batch global context
        
        # Generate M future frame predictions for each batch
        temporal_feat = self.temporal_proj(global_context)  # [B, D*M]
        temporal_feat = temporal_feat.view(B, M, D)  # [B, M, D]
        temporal_feat = temporal_feat.view(B * M, D)  # [B*M, D]
        future_feat = self.future_mlp(temporal_feat)  # [B*M, D]
        
        # Concatenate: [B*N, D] + [B*M, D] = [B*(N+M), D]
        all_feat = torch.cat([current_feat, future_feat], dim=0)
        
        return all_feat


class FutureCameraHead(nn.Module):
    def __init__(self, dim=512, N=3, M=3):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.N = N  # current frames
        self.M = M  # future frames

        # Use improved temporal modeling instead of simple expansion
        self.temporal_mlp = TemporalCameraMLP(self.N, self.M, output_dim)
        
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w, batch_size, num_current_frames):
        BN, hw, c = feat.shape  # [B*N, hw, c]
        
        # Apply residual convolutions
        for i in range(2):
            feat = self.res_conv[i](feat)

        # Global average pooling
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())
        feat = feat.view(feat.size(0), -1)  # [B*N, dim]

        # Temporal modeling for current + future frames
        feat = self.temporal_mlp(feat, batch_size, num_current_frames)  # [B*(N+M), dim] 
        new_BN = feat.shape[0]  # B*(N+M)

        # Final MLPs and pose prediction
        feat = self.more_mlps(feat)  # [B*(N+M), dim]
        with torch.amp.autocast(device_type='cuda', enabled=False): 
            out_t = self.fc_t(feat.float())  # [B*(N+M), 3]
            out_r = self.fc_rot(feat.float())  # [B*(N+M), 9]
            pose = self.convert_pose_to_4x4(new_BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r



class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):

        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r