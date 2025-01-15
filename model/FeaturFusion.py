import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, input_dim, time_steps):
        """
        Feature Fusion module supporting multiple time steps.
        :param input_dim: Dimension of input features (same for intrinsic, transferred, and temporal).
        :param time_steps: Number of time steps to consider for temporal features.
        """
        super(FeatureFusion, self).__init__()

        # Transform each type of feature
        self.intrinsic_fc = nn.Linear(input_dim, input_dim)
        self.transferred_fc = nn.Linear(input_dim, input_dim)
        self.temporal_fc = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(time_steps)])

        # Gating mechanism: combines features at all time steps
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3 * time_steps, input_dim),  # Combine all features across time steps
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # Final transformation layer
        self.output_fc = nn.Linear(input_dim, input_dim)

    def fuse_temporal_features(self, temporal_features):
        """
        Fuse temporal features across multiple time steps.
        :param temporal_features: Temporal features for all time steps
                                  (num_new_nodes x time_steps x input_dim).
        :return: Combined temporal features (num_new_nodes x input_dim).
        """
        fused_temporal_features = []
        for t in range(temporal_features.size(1)):  # Iterate over time steps
            temporal_step = temporal_features[:, t, :]  # Shape: (num_new_nodes x input_dim)
            transformed_step = self.temporal_fc[t](temporal_step)  # Transform for current time step
            fused_temporal_features.append(transformed_step)

        # Sum over all time steps to produce a single temporal representation
        return torch.stack(fused_temporal_features, dim=1).mean(dim=1)  # Shape: (num_new_nodes x input_dim)

    def forward(self, intrinsic_features, transferred_features, temporal_features):
        """
        Fuse features with time step support.
        :param intrinsic_features: Initial features of new nodes (num_new_nodes x input_dim).
        :param transferred_features: Transferred features from existing nodes (num_new_nodes x input_dim).
        :param temporal_features: Temporal features for multiple time steps
                                  (num_new_nodes x time_steps x input_dim).
        :return: Fused features for new nodes (num_new_nodes x input_dim).
        """
        # Step 1: Transform intrinsic and transferred features
        intrinsic_transformed = self.intrinsic_fc(intrinsic_features)  # (num_new_nodes x input_dim)
        transferred_transformed = self.transferred_fc(transferred_features)  # (num_new_nodes x input_dim)

        # Step 2: Fuse temporal features across time steps
        fused_temporal = self.fuse_temporal_features(temporal_features)  # (num_new_nodes x input_dim)

        # Step 3: Concatenate features across time steps
        combined_features = torch.cat(
            [intrinsic_transformed, transferred_transformed, fused_temporal], dim=-1
        )  # (num_new_nodes x input_dim * 3)

        # Step 4: Compute gating values
        gating_values = self.gate(combined_features)  # (num_new_nodes x input_dim)

        # Step 5: Apply gating mechanism to fuse intrinsic and transferred features
        gated_fusion = gating_values * intrinsic_transformed + (1 - gating_values) * transferred_transformed

        # Step 6: Add fused temporal features
        fused_features = gated_fusion + fused_temporal

        # Step 7: Apply final transformation
        output_features = self.output_fc(fused_features)  # (num_new_nodes x input_dim)

        return output_features
