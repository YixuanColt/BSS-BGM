import torch
import torch.nn as nn
import numpy as np


class SpatialConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        """
        Spatial Convolution module for processing spatial features.
        :param input_dim: Dimension of input features (e.g., POIs, road network).
        :param hidden_dim: Dimension of hidden features.
        :param num_layers: Number of convolution layers.
        """
        super(SpatialConv, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=1))
            input_dim = hidden_dim

    def forward(self, adjacency_matrix, node_features):
        """
        Forward pass of spatial convolution.
        :param adjacency_matrix: Adjacency matrix of the graph (num_nodes x num_nodes).
        :param node_features: Node features matrix (num_nodes x input_dim).
        """
        h = node_features.unsqueeze(0)  # Add batch dimension
        for layer in self.layers:
            h = layer(torch.matmul(adjacency_matrix, h))
            h = nn.functional.relu(h)
        return h.squeeze(0)  # Remove batch dimension


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        """
        Temporal Transformer module for processing temporal features.
        :param input_dim: Dimension of input features (e.g., hourly demand, weather data).
        :param hidden_dim: Dimension of Transformer hidden state.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of Transformer layers.
        :param dropout: Dropout probability.
        """
        super(TemporalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, temporal_features):
        """
        Forward pass of temporal transformer.
        :param temporal_features: Temporal feature sequence (seq_len x num_nodes x input_dim).
        """
        embedded = self.embedding(temporal_features)
        transformed = self.transformer(embedded)
        return self.fc(transformed)


class FeatureEncoder:
    def __init__(self, spatial_input_dim, temporal_input_dim, spatial_hidden_dim, temporal_hidden_dim):
        """
        Initialize the feature encoder with spatial and temporal modules.
        """
        self.spatial_conv = SpatialConv(spatial_input_dim, spatial_hidden_dim)
        self.temporal_transformer = TemporalTransformer(temporal_input_dim, temporal_hidden_dim)

    def process_spatial_features(self, adjacency_matrix, node_features):
        """
        Extract spatial features using SpatialConv.
        """
        spatial_features = self.spatial_conv(adjacency_matrix, node_features)
        return spatial_features

    def process_temporal_features(self, temporal_features):
        """
        Extract temporal features using TemporalTransformer.
        """
        temporal_encoded = self.temporal_transformer(temporal_features)
        return temporal_encoded

    def encode_features(self, adjacency_matrix, node_features, temporal_features):
        """
        Encode both spatial and temporal features.
        """
        spatial_encoded = self.process_spatial_features(adjacency_matrix, node_features)
        temporal_encoded = self.process_temporal_features(temporal_features)
        return spatial_encoded, temporal_encoded
