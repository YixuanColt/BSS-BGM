import torch
import torch.nn as nn
import numpy as np


class DynamicGraph(nn.Module):
    def __init__(self, num_nodes, spatial_decay=0.1, temporal_decay=0.1):
        """
        Dynamic graph modeling module.
        :param num_nodes: Initial number of nodes in the graph.
        :param spatial_decay: Decay factor for spatial distances.
        :param temporal_decay: Decay factor for temporal similarities.
        """
        super(DynamicGraph, self).__init__()
        self.num_nodes = num_nodes
        self.spatial_decay = spatial_decay
        self.temporal_decay = temporal_decay

        # Initialize adjacency matrix as a zero matrix
        self.adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    def compute_adjacency(self, spatial_distances, temporal_similarities):
        """
        Compute the adjacency matrix dynamically using spatial and temporal data.
        :param spatial_distances: Matrix of spatial distances between nodes (num_nodes x num_nodes).
        :param temporal_similarities: Matrix of temporal similarities between nodes (num_nodes x num_nodes).
        """
        # Calculate spatial and temporal weights
        spatial_weights = torch.exp(-self.spatial_decay * spatial_distances)
        temporal_weights = self.temporal_decay * temporal_similarities

        # Combine weights to form the adjacency matrix
        self.adjacency_matrix = spatial_weights + temporal_weights
        print("Adjacency matrix updated.")

    def add_new_node(self, new_node_features, spatial_distances, temporal_similarities):
        """
        Add a new node to the graph and update the adjacency matrix.
        :param new_node_features: Features of the new node(s).
        :param spatial_distances: Spatial distances between new and existing nodes.
        :param temporal_similarities: Temporal similarities between new and existing nodes.
        """
        # Calculate new size of the adjacency matrix
        new_size = self.num_nodes + len(new_node_features)

        # Initialize new adjacency matrix
        new_adjacency = torch.zeros((new_size, new_size), dtype=torch.float32)

        # Copy existing adjacency matrix
        new_adjacency[:self.num_nodes, :self.num_nodes] = self.adjacency_matrix

        # Update adjacency matrix with new nodes
        for i, _ in enumerate(new_node_features):
            new_idx = self.num_nodes + i
            new_adjacency[new_idx, :self.num_nodes] = spatial_distances[i]
            new_adjacency[:self.num_nodes, new_idx] = spatial_distances[i]

        # Update adjacency matrix and number of nodes
        self.adjacency_matrix = new_adjacency
        self.num_nodes = new_size
        print(f"New nodes added. Total nodes: {self.num_nodes}.")

    def forward(self, node_features, temporal_features):
        """
        Forward pass for combining spatial and temporal features.
        :param node_features: Spatial features of nodes (num_nodes x feature_dim).
        :param temporal_features: Temporal features of nodes (num_nodes x feature_dim).
        """
        # Combine spatial and temporal features
        combined_features = torch.matmul(self.adjacency_matrix, node_features) + temporal_features
        return combined_features


# Utility function for creating a spatial distance matrix
def compute_spatial_distances(node_positions):
    """
    Compute spatial distance matrix from node positions.
    :param node_positions: Positions of nodes in 2D space (num_nodes x 2).
    """
    num_nodes = len(node_positions)
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distances[i, j] = np.linalg.norm(node_positions[i] - node_positions[j])
    return torch.tensor(distances, dtype=torch.float32)


# Utility function for creating a temporal similarity matrix
def compute_temporal_similarities(temporal_data):
    """
    Compute temporal similarity matrix from temporal data.
    :param temporal_data: Temporal data for nodes (num_nodes x time_steps).
    """
    num_nodes = temporal_data.shape[0]
    similarities = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            similarities[i, j] = np.dot(temporal_data[i], temporal_data[j]) / (
                np.linalg.norm(temporal_data[i]) * np.linalg.norm(temporal_data[j]) + 1e-8
            )
    return torch.tensor(similarities, dtype=torch.float32)
