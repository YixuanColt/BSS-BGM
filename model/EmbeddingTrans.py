import torch
import torch.nn as nn


class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Embedding Transformation module to project features of top-3 similar nodes to new nodes.
        :param input_dim: Dimension of input features (features of existing nodes).
        :param hidden_dim: Dimension of transformed features (output dimension for new nodes).
        """
        super(EmbeddingTransformer, self).__init__()
        self.top_k = 3  # Fixed to select the top-3 similar nodes
        self.fc = nn.Linear(input_dim, hidden_dim)  # Linear layer for feature transformation
        self.orthogonal_mapping = nn.Linear(hidden_dim, hidden_dim)  # Orthogonal mapping

    def compute_similarity(self, new_node_features, existing_node_features):
        """
        Compute cosine similarity between new nodes and existing nodes.
        :param new_node_features: Features of new nodes (num_new_nodes x feature_dim).
        :param existing_node_features: Features of existing nodes (num_existing_nodes x feature_dim).
        :return: Cosine similarity matrix (num_new_nodes x num_existing_nodes).
        """
        # Normalize features to compute cosine similarity
        new_norm = torch.norm(new_node_features, p=2, dim=1, keepdim=True)  # Shape: (num_new_nodes x 1)
        existing_norm = torch.norm(existing_node_features, p=2, dim=1, keepdim=True)  # Shape: (num_existing_nodes x 1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(new_node_features, existing_node_features.T) / (
            new_norm * existing_norm.T + 1e-8
        )  # Shape: (num_new_nodes x num_existing_nodes)

        return similarity_matrix

    def select_top_k(self, similarity_matrix):
        """
        Select the indices and similarities of the top-3 most similar nodes for each new node.
        :param similarity_matrix: Cosine similarity matrix (num_new_nodes x num_existing_nodes).
        :return: Indices and values of top-3 similar nodes for each new node.
        """
        # Get the top-3 most similar nodes
        top_k_values, top_k_indices = torch.topk(similarity_matrix, self.top_k, dim=1)  # Shape: (num_new_nodes x top_k)

        return top_k_indices, top_k_values

    def aggregate_features(self, top_k_indices, top_k_values, existing_node_features):
        """
        Aggregate features of the top-3 similar nodes for each new node using a weighted average.
        :param top_k_indices: Indices of the top-3 similar nodes (num_new_nodes x top_k).
        :param top_k_values: Similarity values of the top-3 similar nodes (num_new_nodes x top_k).
        :param existing_node_features: Features of existing nodes (num_existing_nodes x feature_dim).
        :return: Aggregated features for new nodes (num_new_nodes x feature_dim).
        """
        aggregated_features = []
        for i, indices in enumerate(top_k_indices):
            # Extract features of top-3 nodes
            top_k_features = existing_node_features[indices]  # Shape: (top_k x feature_dim)

            # Compute weighted average of the top-3 features
            weights = top_k_values[i].unsqueeze(1)  # Shape: (top_k x 1)
            weighted_features = top_k_features * weights  # Shape: (top_k x feature_dim)
            aggregated_feature = weighted_features.sum(dim=0) / weights.sum()  # Shape: (feature_dim)

            aggregated_features.append(aggregated_feature)

        return torch.stack(aggregated_features)  # Shape: (num_new_nodes x feature_dim)

    def forward(self, new_node_features, existing_node_features):
        """
        Perform embedding transformation for new nodes based on existing nodes.
        :param new_node_features: Features of new nodes (num_new_nodes x feature_dim).
        :param existing_node_features: Features of existing nodes (num_existing_nodes x feature_dim).
        :return: Transformed features for new nodes (num_new_nodes x hidden_dim).
        """
        # Step 1: Compute similarity matrix
        similarity_matrix = self.compute_similarity(new_node_features, existing_node_features)

        # Step 2: Select top-3 similar nodes for each new node
        top_k_indices, top_k_values = self.select_top_k(similarity_matrix)

        # Step 3: Aggregate features of top-3 similar nodes
        aggregated_features = self.aggregate_features(top_k_indices, top_k_values, existing_node_features)

        # Step 4: Apply linear transformation
        transformed_features = self.fc(aggregated_features)

        # Step 5: Perform orthogonal mapping
        projected_features = self.orthogonal_mapping(transformed_features)

        return projected_features  # Shape: (num_new_nodes x hidden_dim)
