import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataloader import DemandDataset, load_data_from_csv, preprocess_features, preprocess_targets
from lib.logger import ExperimentLogger
from lib.metrics import torch_rmse, torch_mae, torch_evaluate_metrics
from lib.utils import set_seed
from model.DemandPredictionModel import DemandPredictionModel
import numpy as np


# Configuration
config = {
    "seed": 42,
    "learning_rate": 0.001,
    "batch_size": 8,
    "num_epochs": 20,
    "input_dim": 16,
    "hidden_dim": 32,
    "time_steps": 4,
    "spatial_decay": 0.1,
    "temporal_decay": 0.1,
    "log_dir": "./logs"
}

# Set random seed for reproducibility
set_seed(config["seed"])

# Initialize logger
logger = ExperimentLogger(config["log_dir"])
logger.log_config(config)

# Load and preprocess data
print("Loading data...")
intrinsic_features = np.random.rand(50, config["input_dim"])  # Replace with actual features
existing_features = np.random.rand(5, config["input_dim"])  # Replace with actual features
temporal_features = np.random.rand(50, config["time_steps"], config["input_dim"])  # Replace with actual temporal features
spatial_distances = np.random.rand(50, 5)  # Replace with actual spatial distances
temporal_similarities = np.random.rand(50, 5)  # Replace with actual temporal similarities
targets = np.random.rand(50, 1)  # Replace with actual demand values

# Create Dataset and DataLoader
dataset = DemandDataset(
    intrinsic_features=torch.tensor(intrinsic_features, dtype=torch.float32),
    existing_features=torch.tensor(existing_features, dtype=torch.float32),
    temporal_features=torch.tensor(temporal_features, dtype=torch.float32),
    spatial_distances=torch.tensor(spatial_distances, dtype=torch.float32),
    temporal_similarities=torch.tensor(temporal_similarities, dtype=torch.float32),
    targets=torch.tensor(targets, dtype=torch.float32)
)
train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model
model = DemandPredictionModel(
    num_nodes=existing_features.shape[0],
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    time_steps=config["time_steps"],
    spatial_decay=config["spatial_decay"],
    temporal_decay=config["temporal_decay"]
)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Define loss function
def compute_loss(predictions, targets):
    mse_loss = torch.mean((predictions - targets) ** 2)  # MSE
    mae_loss = torch.mean(torch.abs(predictions - targets))  # MAE
    return mse_loss + 0.5 * mae_loss  # Weighted loss

# Training loop
print("Starting training...")
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    for batch in train_loader:
        intrinsic_features = batch["intrinsic_features"]
        existing_features = batch["existing_features"]
        temporal_features = batch["temporal_features"]
        spatial_distances = batch["spatial_distances"]
        temporal_similarities = batch["temporal_similarities"]
        targets = batch["targets"]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(
            intrinsic_features,
            existing_features,
            temporal_features,
            spatial_distances,
            temporal_similarities
        )

        # Compute loss
        loss = compute_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log metrics
    logger.log_metrics(epoch + 1, {"Loss": total_loss / len(train_loader)})
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
print("Evaluating model...")
model.eval()
all_predictions, all_targets = [], []

with torch.no_grad():
    for batch in train_loader:
        predictions = model(
            batch["intrinsic_features"],
            batch["existing_features"],
            batch["temporal_features"],
            batch["spatial_distances"],
            batch["temporal_similarities"]
        )
        all_predictions.append(predictions.numpy())
        all_targets.append(batch["targets"].numpy())

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Compute evaluation metrics
metrics = torch_evaluate_metrics(torch.tensor(all_predictions), torch.tensor(all_targets))
print(f"RMSE: {metrics['RMSE'].item():.4f}, MAE: {metrics['MAE'].item():.4f}")

# Log final metrics
logger.log_metrics("Final Evaluation", {"RMSE": metrics["RMSE"].item(), "MAE": metrics["MAE"].item()})
