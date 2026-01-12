import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_recurrent import Memory
class DepthMLPEnc(nn.Module):
    """
    Simple MLP-based depth encoder without recurrent components.
    More stable for PPO training.
    """
    def __init__(self, base_backbone, num_observations, output_dim=32) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # Simple MLP combining depth features with proprioception
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + num_observations, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, output_dim),
            last_activation
        )

    def forward(self, depth_image, proprioception):
        """
        Forward pass through MLP depth encoder.
        
        Args:
            depth_image: Depth image tensor
            proprioception: Proprioceptive observations
            
        Returns:
            Depth latent features
        """
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        return depth_latent

    def detach_hidden_states(self):
        """Compatibility method - no hidden states in MLP."""
        pass

class DepthHistMLPEnc(nn.Module):
    """
    MLP-based depth encoder that processes depth history through simple concatenation and MLPs.
    Processes multiple frames without recurrent connections.
    """
    def __init__(self, base_backbone, num_observations, output_dim=32, num_frames=2) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.num_frames = num_frames
        
        # Process each frame through the backbone
        # Then combine all frames + proprioception
        combined_input_dim = 32 * num_frames + num_observations
        
        self.combination_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, output_dim),
            last_activation
        )

    def forward(self, depth_images, proprioception):
        """
        Forward pass through history MLP depth encoder.
        
        Args:
            depth_images: Depth image tensor [batch_size, num_frames, height, width] or [batch_size, height, width]
            proprioception: Proprioceptive observations
            
        Returns:
            Depth latent features
        """
        # Handle both single frame and multi-frame inputs
        if len(depth_images.shape) == 3:
            # Single frame case: [batch_size, height, width]
            depth_features = self.base_backbone(depth_images)  # [batch_size, 32]
        else:
            # Multi-frame case: [batch_size, num_frames, height, width]
            batch_size = depth_images.shape[0]
            num_frames = depth_images.shape[1]
            
            # Process each frame through backbone
            depth_features_list = []
            for i in range(num_frames):
                frame_features = self.base_backbone(depth_images[:, i])  # [batch_size, 32]
                depth_features_list.append(frame_features)
            
            # Concatenate all frame features
            depth_features = torch.cat(depth_features_list, dim=-1)  # [batch_size, 32 * num_frames]
        
        # Combine depth features with proprioception
        combined_input = torch.cat((depth_features, proprioception), dim=-1)
        depth_latent = self.combination_mlp(combined_input)
        
        return depth_latent

    def detach_hidden_states(self):
        """Compatibility method - no hidden states in MLP."""
        pass

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, num_observations) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # Pre-processing network for depth and proprioception
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + num_observations, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        
        # Use Memory module from actor_critic_recurrent.py
        # This handles hidden state management consistently with the PPO framework
        self.memory = Memory(input_size=32, type='gru', num_layers=1, hidden_size=512)
        
        # Post-processing network after recurrent layer
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32),
                                last_activation
                            )
        
        # Remove direct GRU and hidden_states as Memory handles this
        print("Using Memory-based RecurrentDepthBackbone with GRU")

    def forward(self, depth_image, proprioception, masks=None, hidden_states=None):
        """
        Forward pass through recurrent depth encoder.
        
        Args:
            depth_image: Depth image tensor [batch_size, height, width] or [seq_len, batch_size, height, width]
            proprioception: Proprioceptive observations [batch_size, proprio_dim] or [seq_len, batch_size, proprio_dim]
            masks: Optional masks for sequence handling [seq_len, batch_size]
            hidden_states: Optional initial hidden states
            
        Returns:
            Depth latent features [batch_size, output_dim] or [seq_len, batch_size, output_dim]
        """
        # Check if we're in sequence mode (for PPO update) or single step (for rollout)
        batch_mode = masks is not None
        
        # Handle depth image shapes
        if depth_image is not None:
            if batch_mode:
                # In batch mode, reshape to process each step separately
                seq_len, batch_size = depth_image.shape[0], depth_image.shape[1]
                
                # Process each frame through the image backbone
                depth_feats = []
                for i in range(seq_len):
                    # Process each frame separately
                    frame_feats = self.base_backbone(depth_image[i])
                    depth_feats.append(frame_feats)
                
                # Stack features back into sequence
                depth_features = torch.stack(depth_feats, dim=0)  # [seq_len, batch_size, 32]
            else:
                # Single step mode (rollout)
                depth_features = self.base_backbone(depth_image)  # [batch_size, 32]
        else:
            # Handle the case with no depth image
            if batch_mode:
                depth_features = torch.zeros(proprioception.shape[0], proprioception.shape[1], 32, 
                                           device=proprioception.device)
            else:
                depth_features = torch.zeros(proprioception.shape[0], 32, device=proprioception.device)
        
        # Combine depth features with proprioception
        if batch_mode:
            # Combine for each step in sequence
            combined_features = []
            for i in range(seq_len):
                combined = torch.cat([depth_features[i], proprioception[i]], dim=-1)
                combined = self.combination_mlp(combined)
                combined_features.append(combined)
            
            depth_latent = torch.stack(combined_features, dim=0)  # [seq_len, batch_size, 32]
        else:
            # Single step mode
            combined = torch.cat([depth_features, proprioception], dim=-1)
            depth_latent = self.combination_mlp(combined)  # [batch_size, 32]
        
        # Process through Memory module
        # Memory handles hidden state management based on whether we're in batch mode
        depth_latent = self.memory(depth_latent, masks, hidden_states)
        
        # Apply output MLP
        if batch_mode:
            # Apply to each step in the sequence
            out_features = []
            for i in range(seq_len):
                out_features.append(self.output_mlp(depth_latent[i]))
            depth_latent = torch.stack(out_features, dim=0)
        else:
            depth_latent = self.output_mlp(depth_latent.squeeze(0))
        
        return depth_latent

    def detach_hidden_states(self):
        """Detach hidden states to prevent backprop through sequences."""
        if hasattr(self.memory, 'hidden_states') and self.memory.hidden_states is not None:
            for h in self.memory.hidden_states:
                h.detach_()
    
    def get_hidden_states(self):
        """Return current hidden states (for storage in rollout buffer)."""
        if hasattr(self.memory, 'hidden_states'):
            return self.memory.hidden_states
        return None
    
    def reset(self, dones=None):
        """Reset hidden states when episodes end."""
        if hasattr(self.memory, 'reset'):
            self.memory.reset(dones)

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [64, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation
            
        self.scandots_output_dim = scandots_output_dim

    def forward(self, images: torch.Tensor):
        """
        Process depth images through CNN backbone.
        
        Args:
            images: Depth image tensor with shape:
                  - [batch_size, height, width] (single image, non-batch mode)
                  - [seq_len, batch_size, height, width] (sequence of images, batch mode)
                  
        Returns:
            Processed image features with shape:
                  - [batch_size, scandots_output_dim] (non-batch mode)
                  - [seq_len, batch_size, scandots_output_dim] (batch mode)
        """
        # Handle different input formats
        original_shape = images.shape
        
        if len(original_shape) == 3:
            # [batch_size, height, width] -> [batch_size, 1, height, width]
            images = images.unsqueeze(1)
            batch_size = original_shape[0]
            is_sequence = False
        elif len(original_shape) == 4:
            # [seq_len, batch_size, height, width] -> reshape for processing
            seq_len, batch_size = original_shape[0], original_shape[1]
            # Reshape to [seq_len*batch_size, 1, height, width]
            images = images.reshape(-1, 1, original_shape[2], original_shape[3])
            is_sequence = True
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
            
        # Process through CNN
        features = self.image_compression(images)
        features = self.output_activation(features)
        
        # Reshape output based on input format
        if is_sequence:
            # Reshape back to [seq_len, batch_size, scandots_output_dim]
            features = features.reshape(seq_len, batch_size, self.scandots_output_dim)

        return features