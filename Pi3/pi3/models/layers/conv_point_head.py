import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLinearPts3d(nn.Module):
    """
    Convolutional point head for Pi3 model.
    
    Instead of a single linear projection, this uses convolutional layers
    to generate 3D points with better spatial coherence.
    """
    
    def __init__(
        self, 
        patch_size, 
        dec_embed_dim, 
        output_dim=3,
        hidden_dim=512,
        num_conv_layers=3,
        kernel_size=3,
        use_batch_norm=True,
        activation='gelu'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Initial projection to get to image space
        self.initial_proj = nn.Linear(dec_embed_dim, hidden_dim * (patch_size // 2)**2)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First upsample
        self.conv_layers.append(nn.ConvTranspose2d(
            hidden_dim, 
            hidden_dim // 2, 
            kernel_size=4, 
            stride=2, 
            padding=1
        ))
        
        current_dim = hidden_dim // 2
        
        # Additional conv layers
        for i in range(num_conv_layers - 1):
            out_dim = current_dim // 2 if i < num_conv_layers - 2 else output_dim
            
            conv = nn.Conv2d(
                current_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.conv_layers.append(conv)
            current_dim = out_dim
        
        # Normalization and activation
        if use_batch_norm:
            self.norms = nn.ModuleList([
                nn.BatchNorm2d(hidden_dim // 2 if i == 0 else 
                              (hidden_dim // (2**(i+1)) if i < num_conv_layers - 1 else output_dim))
                for i in range(num_conv_layers)
            ])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_conv_layers)])
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        # Initial projection
        x = self.initial_proj(tokens)  # B, S, hidden_dim * (patch_size//2)**2
        
        # Reshape to image-like tensor
        ps_half = self.patch_size // 2
        x = x.transpose(-1, -2).reshape(B, -1, H // self.patch_size * ps_half, W // self.patch_size * ps_half)
        
        # Apply convolutional layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norms)):
            x = conv(x)
            x = norm(x)
            
            # Apply activation except for the last layer
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
        
        # Return in same format as LinearPts3d: B,H,W,output_dim
        return x.permute(0, 2, 3, 1)


class SimpleConvPts3d(nn.Module):
    """
    Simplified convolutional point head with direct upsampling.
    """
    
    def __init__(
        self,
        patch_size,
        dec_embed_dim,
        output_dim=3,
        hidden_dims=[256, 128, 64],
        kernel_size=3
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Project to spatial dimensions
        self.proj = nn.Linear(dec_embed_dim, hidden_dims[0] * patch_size**2)
        
        # Build conv layers
        layers = []
        in_dim = hidden_dims[0]
        
        for out_dim in hidden_dims[1:]:
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(num_groups=min(32, out_dim), num_channels=out_dim),
                nn.GELU()
            ])
            in_dim = out_dim
        
        # Final conv to output dimension
        layers.append(nn.Conv2d(in_dim, output_dim, kernel_size=1))
        
        self.conv_net = nn.Sequential(*layers)
    
    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        # Convert to float32 if needed (BFloat16 not supported by some ops)
        original_dtype = tokens.dtype
        if tokens.dtype == torch.bfloat16:
            tokens = tokens.float()
        
        # Project and reshape
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).reshape(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,hidden_dims[0],H,W
        
        # Apply conv network
        out = self.conv_net(feat)
        
        # Convert back to original dtype if needed
        if original_dtype == torch.bfloat16:
            out = out.to(original_dtype)
        
        # Return in same format as LinearPts3d: B,H,W,output_dim
        return out.permute(0, 2, 3, 1)