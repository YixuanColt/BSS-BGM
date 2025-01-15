import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.DemandPredictionModel import DemandPredictionModel
import numpy as np


# Example Dataset Class
class DemandDataset(Dataset):
    def __init__(self, intrinsic_features, existing_features, temporal_features, spatial_distances, temporal_similarities, targets):
        """
        Dataset for demand prediction.
        """
        self.intrinsic_features = intrinsic_features
        self.existing_features = existing_features
        self.temporal_features = temporal_features
        self.spatial_distances = spatial_distances
        self.temporal_similarities = temporal_similarities
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'intrinsic_features': self.intrinsic_features[idx],
            'temporal_features': self.temporal_features[idx],
            'spatial_distances': self.spatial_distances[idx],
            'temporal_similarities': self.temporal_similarities[idx],
            'targets': self.targets[idx]
        }


# Loss Function
def compute_loss(predictions, targets):
    """
    Loss function combining Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    """
    mse_loss = nn.MSELoss()(predictions, targets)
    mae_loss = nn.L1Loss()(predictions, targets)
    return mse_loss + 0.5 * mae_loss  # Weighted combination


# Training Loop
def train_model(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            # Move data to device
            intrinsic_features = batch['intrinsic_features']
            temporal_features = batch['temporal_features']
            spatial_distances = batch['spatial_distances']
            temporal_similarities = batch['temporal_similarities']
            targets = batch['targets']

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(intrinsic_features, batch['existing_features'], temporal_features,
                                spatial_distances, temporal_similarities)

            # Compute loss
            loss = compute_loss(predictions, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch['intrinsic_features'], batch['existing_features'], batch['temporal_features'],
                                batch['spatial_distances'], batch['temporal_similarities'])
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")


# Example Data Preparation
num_new_nodes = 10
num_existing_nodes = 5
input_dim = 16
time_steps = 4

intrinsic_features = torch.rand((num_new_nodes, input_dim))
existing_features = torch.rand((num_existing_nodes, input_dim))
temporal_features = torch.rand((num_new_nodes, time_steps, input_dim))
spatial_distances = torch.rand((num_new_nodes, num_existing_nodes))
temporal_similarities = torch.rand((num_new_nodes, num_existing_nodes))
targets = torch.rand((num_new_nodes, 1))

# Create Dataset and DataLoader
dataset = DemandDataset(intrinsic_features, existing_features, temporal_features,
                        spatial_distances, temporal_similarities, targets)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize Model, Optimizer, and Train
model = DemandPredictionModel(num_nodes=num_existing_nodes, input_dim=input_dim, hidden_dim=32,
                               time_steps=time_steps, spatial_decay=0.1, temporal_decay=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, optimizer, num_epochs=20)

# Evaluate the model
evaluate_model(model, train_loader)
