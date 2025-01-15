import pandas as pd
from torch.utils.data import Dataset


class DemandDataset(Dataset):
    def __init__(self, intrinsic_features, existing_features, temporal_features, spatial_distances, temporal_similarities, targets):
        """
        Dataset for demand prediction.

        :param intrinsic_features: Features of new nodes (num_samples x feature_dim).
        :param existing_features: Features of existing nodes (num_existing_nodes x feature_dim).
        :param temporal_features: Temporal features for new nodes (num_samples x time_steps x feature_dim).
        :param spatial_distances: Spatial distances between new and existing nodes (num_samples x num_existing_nodes).
        :param temporal_similarities: Temporal similarities between new and existing nodes (num_samples x num_existing_nodes).
        :param targets: Target demand values (num_samples x 1).
        """
        self.intrinsic_features = intrinsic_features
        self.existing_features = existing_features
        self.temporal_features = temporal_features
        self.spatial_distances = spatial_distances
        self.temporal_similarities = temporal_similarities
        self.targets = targets

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample based on the index.
        :param idx: Index of the sample to retrieve.
        :return: Dictionary containing input features and target.
        """
        return {
            'intrinsic_features': self.intrinsic_features[idx],
            'existing_features': self.existing_features,
            'temporal_features': self.temporal_features[idx],
            'spatial_distances': self.spatial_distances[idx],
            'temporal_similarities': self.temporal_similarities[idx],
            'targets': self.targets[idx]
        }


def load_data_from_csv(file_path):
    """
    Utility function to load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: Pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} successfully.")
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def preprocess_features(df, feature_columns):
    """
    Preprocess features from a DataFrame.
    :param df: Input DataFrame.
    :param feature_columns: List of column names to extract as features.
    :return: Processed features as a NumPy array.
    """
    features = df[feature_columns].values
    return features


def preprocess_targets(df, target_column):
    """
    Preprocess target values from a DataFrame.
    :param df: Input DataFrame.
    :param target_column: Name of the target column.
    :return: Processed targets as a NumPy array.
    """
    targets = df[target_column].values
    return targets
