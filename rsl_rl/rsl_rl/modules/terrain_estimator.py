# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks.memory import Memory


class TerrainEstimator(nn.Module):
    """
    Terrain estimator that predicts raycast distances from depth images and proprioceptive data.
    
    Architecture:
    1. CNN encoder for depth images
    2. GRU memory for temporal processing
    3. MLP decoder to predict raycast distances
    """
    
    def __init__(
        self,
        depth_image_shape: tuple,  # (height, width)
        proprio_dim: int,  # proprioceptive input dimension (base vel + ang vel)
        num_raycast_outputs: int,  # number of raycast distance outputs
        encoder_output_dim: int = 64,
        memory_hidden_size: int = 256,
        memory_num_layers: int = 1,
        memory_type: str = "gru",
        decoder_hidden_dims: list = [128, 64],
        activation: str = "elu",
        **kwargs
    ):
        super().__init__()
        
        self.depth_image_shape = depth_image_shape
        self.proprio_dim = proprio_dim
        self.num_raycast_outputs = num_raycast_outputs
        self.encoder_output_dim = encoder_output_dim
        
        # Set activation function
        if activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ELU()
        
        # CNN encoder for depth images
        self.depth_encoder = self._build_cnn_encoder(depth_image_shape, encoder_output_dim)
        
        # Combine depth features with proprioceptive data
        combined_input_dim = encoder_output_dim + proprio_dim
        self.combination_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, encoder_output_dim),
            self.activation
        )
        
        # GRU memory for temporal processing
        self.memory = Memory(
            input_size=encoder_output_dim,
            type=memory_type,
            num_layers=memory_num_layers,
            hidden_size=memory_hidden_size
        )
        
        # MLP decoder to predict raycast distances
        self.decoder = self._build_decoder(memory_hidden_size, decoder_hidden_dims, num_raycast_outputs)
        
        print(f"TerrainEstimator initialized:")
        print(f"  Depth image shape: {depth_image_shape}")
        print(f"  Proprio dim: {proprio_dim}")
        print(f"  Raycast outputs: {num_raycast_outputs}")
        print(f"  Memory type: {memory_type}, hidden size: {memory_hidden_size}")
    
    def _build_cnn_encoder(self, image_shape, output_dim):
        """Build CNN encoder for depth images."""
        height, width = image_shape
        
        # CNN layers for depth image processing
        encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # Reduce spatial size
            self.activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further reduce
            self.activation,
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            
            # Global average pooling to handle variable input sizes
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            
            # Final MLP
            nn.Linear(64 * 4 * 4, 128),
            self.activation,
            nn.Linear(128, output_dim),
            self.activation
        )
        
        return encoder
    
    def _build_decoder(self, input_dim, hidden_dims, output_dim):
        """Build MLP decoder for raycast distance prediction."""
        layers = []
        
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation
            ])
            current_dim = hidden_dim
        
        # Output layer (no activation to allow full range of distances)
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, depth_images, proprio_data, masks=None, hidden_states=None):
        """
        Forward pass of terrain estimator.
        
        Args:
            depth_images: Depth images tensor
                - Single step: [batch_size, height, width]
                - Sequence: [seq_len, batch_size, height, width]
            proprio_data: Proprioceptive data (base vel + ang vel)
                - Single step: [batch_size, proprio_dim]
                - Sequence: [seq_len, batch_size, proprio_dim]
            masks: Optional masks for sequence handling [seq_len, batch_size]
            hidden_states: Optional initial hidden states
            
        Returns:
            Predicted raycast distances
                - Single step: [batch_size, num_raycast_outputs]
                - Sequence: [seq_len, batch_size, num_raycast_outputs]
        """
        batch_mode = masks is not None
        
        # Process depth images through CNN encoder
        if batch_mode:
            # Sequence mode: process each frame separately
            seq_len, batch_size = depth_images.shape[0], depth_images.shape[1]
            
            depth_features = []
            for i in range(seq_len):
                # Add channel dimension and encode
                frame = depth_images[i].unsqueeze(1)  # [batch_size, 1, height, width]
                frame_features = self.depth_encoder(frame)  # [batch_size, encoder_output_dim]
                depth_features.append(frame_features)
            
            depth_features = torch.stack(depth_features, dim=0)  # [seq_len, batch_size, encoder_output_dim]
        else:
            # Single step mode
            if len(depth_images.shape) == 3:
                # Add channel dimension
                depth_images = depth_images.unsqueeze(1)  # [batch_size, 1, height, width]
            depth_features = self.depth_encoder(depth_images)  # [batch_size, encoder_output_dim]
        
        # Combine depth features with proprioceptive data
        if batch_mode:
            combined_features = []
            for i in range(seq_len):
                combined = torch.cat([depth_features[i], proprio_data[i]], dim=-1)
                combined = self.combination_mlp(combined)
                combined_features.append(combined)
            
            combined_input = torch.stack(combined_features, dim=0)  # [seq_len, batch_size, encoder_output_dim]
        else:
            combined = torch.cat([depth_features, proprio_data], dim=-1)
            combined_input = self.combination_mlp(combined)  # [batch_size, encoder_output_dim]
        
        # Process through memory (GRU)
        memory_output = self.memory(combined_input, masks, hidden_states)
        
        # Decode to raycast distances
        if batch_mode:
            raycast_predictions = []
            for i in range(seq_len):
                pred = self.decoder(memory_output[i])
                raycast_predictions.append(pred)
            
            raycast_predictions = torch.stack(raycast_predictions, dim=0)  # [seq_len, batch_size, num_raycast_outputs]
        else:
            if len(memory_output.shape) == 3:  # Remove sequence dimension added by memory
                memory_output = memory_output.squeeze(0)
            raycast_predictions = self.decoder(memory_output)  # [batch_size, num_raycast_outputs]
        
        return raycast_predictions
    
    def act_inference(self, depth_images, proprio_data):
        """Inference mode forward pass."""
        return self.forward(depth_images, proprio_data, masks=None, hidden_states=None)
    
    def reset(self, dones=None, hidden_states=None):
        """Reset memory hidden states."""
        self.memory.reset(dones, hidden_states)
    
    def detach_hidden_states(self, dones=None):
        """Detach hidden states to prevent backprop through sequences."""
        self.memory.detach_hidden_states(dones)
    
    def get_hidden_states(self):
        """Get current hidden states."""
        return self.memory.hidden_states
    
    def set_hidden_states(self, hidden_states):
        """Set hidden states."""
        self.memory.hidden_states = hidden_states

