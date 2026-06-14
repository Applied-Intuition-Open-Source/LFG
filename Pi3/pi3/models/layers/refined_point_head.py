import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinedLinearPts3d(nn.Module):
    """
    Enhanced point head with refinement layers for Pi3 model.
    
    This module extends the basic LinearPts3d by adding refinement layers that:
    1. Process the initial point predictions
    2. Apply spatial and feature refinement
    3. Improve local consistency and accuracy
    """
    
    def __init__(
        self, 
        patch_size, 
        dec_embed_dim, 
        output_dim=3,
        num_refine_layers=2,
        refine_hidden_dim=256,
        use_layer_norm=True,
        dropout=0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Initial projection (same as LinearPts3d)
        self.proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        
        # Projection to refinement dimension
        self.input_proj = nn.Conv2d(output_dim, refine_hidden_dim, kernel_size=1)
        
        # Refinement layers
        self.refine_layers = nn.ModuleList()
        
        for i in range(num_refine_layers):
            layer = nn.ModuleDict({
                'conv1': nn.Conv2d(
                    refine_hidden_dim, 
                    refine_hidden_dim, 
                    kernel_size=3, 
                    padding=1
                ),
                'conv2': nn.Conv2d(
                    refine_hidden_dim, 
                    refine_hidden_dim, 
                    kernel_size=3, 
                    padding=1
                ),
                'norm1': nn.LayerNorm([refine_hidden_dim]) if use_layer_norm else nn.Identity(),
                'norm2': nn.LayerNorm([refine_hidden_dim]) if use_layer_norm else nn.Identity(),
                'dropout': nn.Dropout2d(dropout),
                'activation': nn.GELU()
            })
            self.refine_layers.append(layer)
        
        # Final projection back to output dimension
        self.final_proj = nn.Conv2d(refine_hidden_dim, output_dim, kernel_size=1)
    
    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        # Initial 3D point prediction
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).reshape(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,output_dim,H,W
        
        # Project to refinement dimension
        x = self.input_proj(feat)
        
        # Apply refinement layers
        for i, layer in enumerate(self.refine_layers):
            # First conv block
            x = layer['conv1'](x)
            x = x.permute(0, 2, 3, 1)  # B,H,W,C for LayerNorm
            x = layer['norm1'](x)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = layer['activation'](x)
            x = layer['dropout'](x)
            
            # Second conv block
            x = layer['conv2'](x)
            x = x.permute(0, 2, 3, 1)  # B,H,W,C for LayerNorm
            x = layer['norm2'](x)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = layer['activation'](x)
        
        # Final projection
        refined_feat = self.final_proj(x)
        
        # Return in same format as LinearPts3d: B,H,W,output_dim
        return refined_feat.permute(0, 2, 3, 1)