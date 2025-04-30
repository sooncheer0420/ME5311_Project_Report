import os
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR



# Set random seed for reproducibility

torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load data
ds_slp = xr.open_dataset('slp_low_res.nc')  # Sea Level Pressure
ds_t2m = xr.open_dataset('t2m_low_res.nc')  # 2m Temperature

# Get data values
da_msl = ds_slp['msl']  # 'msl' is the variable name for sea level pressure
x_msl = da_msl.values

da_t2m = ds_t2m['t2m']  # 't2m' is the variable name for 2m temperature
x_t2m = da_t2m.values

print("x_msl.shape =", x_msl.shape)
print("x_t2m.shape =", x_t2m.shape)

# Extract subset for testing/training (days 10000-11000)
# x_msl_subset = x_msl[10000:15000, :, :]
# x_t2m_subset = x_t2m[10000:15000, :, :]
x_msl_subset = x_msl
x_t2m_subset = x_t2m-273.15


print("x_msl_subset.shape:", x_msl_subset.shape)
print("x_t2m_subset.shape:", x_t2m_subset.shape)


customed_training_epoch=100
customed_dropout_rate=0.0
customed_learning_rate=0.001


####################################################################################################################
####################################################################################################################
####################################################################################################################

# MultiTaskModel

# Use both pressure and temperature as inputs
# Predict both future pressure and temperature simultaneously
# Share knowledge between tasks while maintaining task-specific components


class WeatherMultiTaskModel(nn.Module):
    def __init__(self, input_channels=2, shared_cnn_features=64, lstm_hidden=128, 
                 output_height=51, output_width=81, dropout_rate=customed_dropout_rate):
        super(WeatherMultiTaskModel, self).__init__()
        
        self.output_height = output_height
        self.output_width = output_width
        self.dropout_rate = dropout_rate
        
        # Shared CNN encoder
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(64, shared_cnn_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(shared_cnn_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate CNN output size
        pool_strides = [2, 2, 2]
        reduction_factor = 1
        for stride in pool_strides:
            reduction_factor *= stride
        
        cnn_output_height = output_height // reduction_factor
        cnn_output_width = output_width // reduction_factor
        self.cnn_flattened_size = shared_cnn_features * cnn_output_height * cnn_output_width
        
        # Shared LSTM layer
        self.shared_lstm = nn.LSTM(
            input_size=self.cnn_flattened_size, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Task-specific layers
        
        # For pressure prediction
        self.pressure_dropout = nn.Dropout(dropout_rate)
        self.pressure_fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.pressure_fc2 = nn.Linear(lstm_hidden // 2, output_height * output_width)
        
        # For temperature prediction
        self.temp_dropout = nn.Dropout(dropout_rate)
        self.temp_fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.temp_fc2 = nn.Linear(lstm_hidden // 2, output_height * output_width)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        - x: Input tensor with shape [batch_size, seq_len, channels, height, width]
              where channels=2 (pressure, temperature)
        
        Returns:
        - pressure_output: Predicted pressure map
        - temp_output: Predicted temperature map
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process each time step through the shared CNN
        cnn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # Shape: [batch_size, channels, height, width]
            cnn_out = self.shared_cnn(x_t)
            cnn_outputs.append(cnn_out)
        
        # Stack time steps and reshape for LSTM
        cnn_sequence = torch.stack(cnn_outputs, dim=1)
        lstm_input = cnn_sequence.reshape(batch_size, seq_len, -1)
        
        # Process through shared LSTM
        lstm_out, _ = self.shared_lstm(lstm_input)
        
        final_hidden = lstm_out[:, -1, :]  # Get the final time step output
        
        # Task-specific decoders
        
        # Pressure prediction path
        pressure_hidden = self.pressure_dropout(final_hidden)
        pressure_hidden = F.relu(self.pressure_fc1(pressure_hidden))
        pressure_output = self.pressure_fc2(pressure_hidden)
        pressure_output = pressure_output.reshape(batch_size, self.output_height, self.output_width)
        
        # Temperature prediction path
        temp_hidden = self.temp_dropout(final_hidden)
        temp_hidden = F.relu(self.temp_fc1(temp_hidden))
        temp_output = self.temp_fc2(temp_hidden)
        temp_output = temp_output.reshape(batch_size, self.output_height, self.output_width)
        
        return pressure_output, temp_output


# a new dataset class that can handle both variables
class WeatherMultiVariableDataset(Dataset):
    """Dataset for multi-variable weather forecasting tasks."""
    
    def __init__(self, X_pressure, X_temp, y_pressure, y_temp, use_float16=False):
        """
        Initialize dataset with both pressure and temperature data.
        
        Parameters:
        - X_pressure: Input pressure sequences
        - X_temp: Input temperature sequences
        - y_pressure: Target pressure values
        - y_temp: Target temperature values
        - use_float16: Whether to use half precision (float16)
        """
        dtype = torch.float16 if use_float16 else torch.float32
        
        # Ensure all input arrays have the correct shape [samples, timesteps, height, width]
        self.X_pressure = torch.tensor(X_pressure, dtype=dtype)
        self.X_temp = torch.tensor(X_temp, dtype=dtype)
        self.y_pressure = torch.tensor(y_pressure, dtype=dtype)
        self.y_temp = torch.tensor(y_temp, dtype=dtype)
        
        # Sanity check
        assert self.X_pressure.shape[0] == self.X_temp.shape[0], "Number of samples must match"
        assert self.X_pressure.shape[1] == self.X_temp.shape[1], "Time window sizes must match"
    
    def __len__(self):
        return len(self.X_pressure)
    
    def __getitem__(self, idx):
        # Stack pressure and temperature channels
        inputs = torch.stack([
            self.X_pressure[idx],  # [time_steps, height, width]
            self.X_temp[idx]       # [time_steps, height, width]
        ], dim=1)  # Result: [time_steps, channels=2, height, width]
        
        # Transpose to get [channels, time_steps, height, width]
        inputs = inputs.permute(1, 0, 2, 3)
        
        return {
            'input': inputs,
            'target_pressure': self.y_pressure[idx],
            'target_temp': self.y_temp[idx]
        }


def create_time_series_dataset(data, window_size=7, horizon=1):
    """
    Create sliding windows from time series data.
    
    Parameters:
    - data: numpy array of shape [time_steps, height, width]
    - window_size: number of time steps to include in each input window
    - horizon: how many steps ahead to predict
    
    Returns:
    - X: input sequences [samples, time_steps, height, width]
    - y: target values [samples, height, width]
    """
    X = []
    y = []
    
    # Create sliding windows
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+horizon-1])
    
    return np.array(X), np.array(y)


# loss function for multi-task learning that balances the losses

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights={'pressure': 0.5, 'temperature': 0.5}):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pressure_pred, temp_pred, pressure_target, temp_target):
        pressure_loss = self.mse_loss(pressure_pred, pressure_target)
        temp_loss = self.mse_loss(temp_pred, temp_target)
        
        # Weighted sum of losses
        total_loss = (self.task_weights['pressure'] * pressure_loss + 
                      self.task_weights['temperature'] * temp_loss)
        
        return total_loss, pressure_loss, temp_loss


# train the multi-task model
def train_multitask_model(model, train_dataloader, test_dataloader, num_epochs=100, 
                         learning_rate=0.001, task_weights={'pressure': 0.5, 'temperature': 0.5},
                         scheduler_type='plateau', patience=5):
    """
    Train the multi-task weather forecasting model.
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = MultiTaskLoss(task_weights=task_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Set up learning rate scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, 
                                     patience=patience, verbose=True)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    training_losses = {'total': [], 'pressure': [], 'temperature': []}
    validation_losses = {'total': [], 'pressure': [], 'temperature': []}
    learning_rates = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_total_loss = 0.0
        epoch_pressure_loss = 0.0
        epoch_temp_loss = 0.0
        
        for batch in train_dataloader:
            # Move data to device
            inputs = batch['input'].to(device)  # [batch, channels, time_steps, height, width]
            # Permute to [batch, time_steps, channels, height, width]
            inputs = inputs.permute(0, 2, 1, 3, 4)
            
            pressure_targets = batch['target_pressure'].to(device)
            temp_targets = batch['target_temp'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pressure_pred, temp_pred = model(inputs)
            
            # Calculate loss
            total_loss, pressure_loss, temp_loss = criterion(
                pressure_pred, temp_pred, pressure_targets, temp_targets
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_pressure_loss += pressure_loss.item()
            epoch_temp_loss += temp_loss.item()
        
        # Calculate average training losses for the epoch
        avg_train_total_loss = epoch_total_loss / len(train_dataloader)
        avg_train_pressure_loss = epoch_pressure_loss / len(train_dataloader)
        avg_train_temp_loss = epoch_temp_loss / len(train_dataloader)
        
        training_losses['total'].append(avg_train_total_loss)
        training_losses['pressure'].append(avg_train_pressure_loss)
        training_losses['temperature'].append(avg_train_temp_loss)
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_pressure_loss = 0.0
        val_temp_loss = 0.0
        
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['input'].to(device)
                inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch, time_steps, channels, height, width]
                
                pressure_targets = batch['target_pressure'].to(device)
                temp_targets = batch['target_temp'].to(device)
                
                # Forward pass
                pressure_pred, temp_pred = model(inputs)
                
                # Calculate loss
                total_loss, pressure_loss, temp_loss = criterion(
                    pressure_pred, temp_pred, pressure_targets, temp_targets
                )
                
                # Accumulate losses
                val_total_loss += total_loss.item()
                val_pressure_loss += pressure_loss.item()
                val_temp_loss += temp_loss.item()
        
        # Calculate average validation losses for the epoch
        avg_val_total_loss = val_total_loss / len(test_dataloader)
        avg_val_pressure_loss = val_pressure_loss / len(test_dataloader)
        avg_val_temp_loss = val_temp_loss / len(test_dataloader)
        
        validation_losses['total'].append(avg_val_total_loss)
        validation_losses['pressure'].append(avg_val_pressure_loss)
        validation_losses['temperature'].append(avg_val_temp_loss)
        
        # Save current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Update learning rate
        if scheduler_type == 'plateau':
            scheduler.step(avg_val_total_loss)
        else:
            scheduler.step()
        
        # Save best model
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! (Validation Loss: {best_val_loss:.4f})")
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: [Total: {avg_train_total_loss:.4f}, P: {avg_train_pressure_loss:.4f}, T: {avg_train_temp_loss:.4f}], "
              f"Val Loss: [Total: {avg_val_total_loss:.4f}, P: {avg_val_pressure_loss:.4f}, T: {avg_val_temp_loss:.4f}], "
              f"LR: {current_lr:.6f}")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model, training_losses, validation_losses, learning_rates


####################################################################################################################
####################################################################################################################
####################################################################################################################

############ implement ############

# Normalize pressure data
mean_slp = np.mean(x_msl_subset)
std_slp = np.std(x_msl_subset)
x_msl_normalized = (x_msl_subset - mean_slp) / std_slp

# Normalize temperature data
mean_t2m = np.mean(x_t2m_subset)
std_t2m = np.std(x_t2m_subset)
x_t2m_normalized = (x_t2m_subset - mean_t2m) / std_t2m



# Create time series datasets for both variables
X_pressure, y_pressure = create_time_series_dataset(x_msl_normalized, window_size=7, horizon=1)
X_temp, y_temp = create_time_series_dataset(x_t2m_normalized, window_size=7, horizon=1)

# Create the multi-variable dataset
full_dataset = WeatherMultiVariableDataset(
    X_pressure=X_pressure,
    X_temp=X_temp,
    y_pressure=y_pressure,
    y_temp=y_temp,
    use_float16=False
)

# Calculate sizes for train and test sets (80% train, 20% test)
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Initialize the multi-task model
input_height, input_width = x_msl_subset.shape[1], x_msl_subset.shape[2]  # 51, 81
multi_task_model = WeatherMultiTaskModel(
    input_channels=2,  # pressure and temperature
    shared_cnn_features=64,
    lstm_hidden=128,
    output_height=input_height,
    output_width=input_width
)

# Train the model
trained_model, training_losses, validation_losses, learning_rates = train_multitask_model(
    model=multi_task_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    num_epochs=customed_training_epoch,
    learning_rate=customed_learning_rate,
    task_weights={'pressure': 0.5, 'temperature': 0.5},  # Equal weight to both tasks
    scheduler_type='plateau',
    patience=5
)

# Save the trained model
torch.save(trained_model.state_dict(), 'CNN-LSTM_model.pth')

####################################################################################################################



def visualize_multitask_prediction(model, dataloader, num_samples=3):
    """
    Visualize model predictions for both pressure and temperature.
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    inputs = batch['input'].to(device)
    inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch, time_steps, channels, height, width]
    
    pressure_targets = batch['target_pressure'].to(device)
    temp_targets = batch['target_temp'].to(device)
    
    # Generate predictions
    with torch.no_grad():
        pressure_preds, temp_preds = model(inputs)
    
    # Move tensors to CPU for plotting
    inputs = inputs.cpu().numpy()
    pressure_targets = pressure_targets.cpu().numpy()
    temp_targets = temp_targets.cpu().numpy()
    pressure_preds = pressure_preds.cpu().numpy()
    temp_preds = temp_preds.cpu().numpy()
    
    # Visualize multiple samples
    for i in range(min(num_samples, inputs.shape[0])):
        # Pressure visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot last frame of input sequence (pressure)
        im0 = axes[0, 0].imshow(inputs[i, -1, 0])
        axes[0, 0].set_title('Last Input Frame (Pressure)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Plot ground truth (pressure)
        im1 = axes[0, 1].imshow(pressure_targets[i])
        axes[0, 1].set_title('Ground Truth (Pressure)')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Plot prediction (pressure)
        im2 = axes[0, 2].imshow(pressure_preds[i])
        axes[0, 2].set_title('Prediction (Pressure)')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Plot last frame of input sequence (temperature)
        im3 = axes[1, 0].imshow(inputs[i, -1, 1])
        axes[1, 0].set_title('Last Input Frame (Temperature)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot ground truth (temperature)
        im4 = axes[1, 1].imshow(temp_targets[i])
        axes[1, 1].set_title('Ground Truth (Temperature)')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Plot prediction (temperature)
        im5 = axes[1, 2].imshow(temp_preds[i])
        axes[1, 2].set_title('Prediction (Temperature)')
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'multitask_prediction_sample_{i}.png')
        plt.show()



def evaluate_model_comprehensive(model, dataloader, mean_pressure=None, std_pressure=None, 
                                mean_temp=None, std_temp=None):
    """
    Comprehensive evaluation of the multi-task weather forecasting model.
    
    Parameters:
    - model: Trained PyTorch model
    - dataloader: DataLoader containing test data
    - mean_pressure, std_pressure: Statistics for denormalizing pressure data
    - mean_temp, std_temp: Statistics for denormalizing temperature data
    
    Returns:
    - Dictionary of evaluation metrics for both tasks
    """
    model.eval()
    
    # Lists to store predictions and targets
    pressure_preds_list = []
    pressure_targets_list = []
    temp_preds_list = []
    temp_targets_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch, time_steps, channels, height, width]
            
            pressure_targets = batch['target_pressure'].to(device)
            temp_targets = batch['target_temp'].to(device)
            
            # Forward pass
            pressure_preds, temp_preds = model(inputs)
            
            # Move to CPU and convert to numpy for metric calculation
            pressure_preds_list.append(pressure_preds.cpu().numpy())
            pressure_targets_list.append(pressure_targets.cpu().numpy())
            temp_preds_list.append(temp_preds.cpu().numpy())
            temp_targets_list.append(temp_targets.cpu().numpy())
    
    # Concatenate all batches
    pressure_preds = np.concatenate(pressure_preds_list, axis=0)
    pressure_targets = np.concatenate(pressure_targets_list, axis=0)
    temp_preds = np.concatenate(temp_preds_list, axis=0)
    temp_targets = np.concatenate(temp_targets_list, axis=0)
    
    # Denormalize if mean and std are provided
    if mean_pressure is not None and std_pressure is not None:
        pressure_preds = pressure_preds * std_pressure + mean_pressure
        pressure_targets = pressure_targets * std_pressure + mean_pressure
    
    if mean_temp is not None and std_temp is not None:
        temp_preds = temp_preds * std_temp + mean_temp
        temp_targets = temp_targets * std_temp + mean_temp
    
    # Reshape for metric calculation (flatten spatial dimensions)
    p_preds_flat = pressure_preds.reshape(-1)
    p_targets_flat = pressure_targets.reshape(-1)
    t_preds_flat = temp_preds.reshape(-1)
    t_targets_flat = temp_targets.reshape(-1)
    
    # Calculate metrics for pressure
    pressure_mae = mean_absolute_error(p_targets_flat, p_preds_flat)
    pressure_rmse = np.sqrt(mean_squared_error(p_targets_flat, p_preds_flat))
    pressure_r2 = r2_score(p_targets_flat, p_preds_flat)
    
    
    # Calculate metrics for temperature
    temp_mae = mean_absolute_error(t_targets_flat, t_preds_flat)
    temp_rmse = np.sqrt(mean_squared_error(t_targets_flat, t_preds_flat))
    temp_r2 = r2_score(t_targets_flat, t_preds_flat)
    

    
    # Combined metrics (average of both tasks)
    combined_mae = (pressure_mae + temp_mae) / 2
    combined_rmse = (pressure_rmse + temp_rmse) / 2
    combined_r2 = (pressure_r2 + temp_r2) / 2
    
    # Calculate spatial correlation coefficients
    pressure_spatial_corrs = []
    temp_spatial_corrs = []
    
    for i in range(pressure_preds.shape[0]):
        # Correlation for each sample across spatial dimensions
        p_corr = np.corrcoef(pressure_targets[i].flatten(), pressure_preds[i].flatten())[0, 1]
        t_corr = np.corrcoef(temp_targets[i].flatten(), temp_preds[i].flatten())[0, 1]
        
        pressure_spatial_corrs.append(p_corr)
        temp_spatial_corrs.append(t_corr)
    
    pressure_spatial_corr = np.mean(pressure_spatial_corrs)
    temp_spatial_corr = np.mean(temp_spatial_corrs)
    combined_spatial_corr = (pressure_spatial_corr + temp_spatial_corr) / 2
    
    # Return all metrics in a dictionary
    return {
        'pressure_mae': pressure_mae,
        'pressure_rmse': pressure_rmse,
        'pressure_r2': pressure_r2,
        'pressure_spatial_corr': pressure_spatial_corr,
        
        'temp_mae': temp_mae,
        'temp_rmse': temp_rmse,
        'temp_r2': temp_r2,
        'temp_spatial_corr': temp_spatial_corr,
        
        'combined_mae': combined_mae,
        'combined_rmse': combined_rmse,
        'combined_r2': combined_r2,
        'combined_spatial_corr': combined_spatial_corr
    }


# Evaluate model with comprehensive metrics
metrics = evaluate_model_comprehensive(
    trained_model, 
    test_dataloader,
    mean_pressure=mean_slp,  # Pass the original mean used for normalization
    std_pressure=std_slp,    # Pass the original std used for normalization
    mean_temp=mean_t2m,      # Pass the original mean used for temperature normalization
    std_temp=std_t2m         # Pass the original std used for temperature normalization
)

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print("-" * 40)
print("Pressure Metrics:")
for metric_name in ['pressure_mae', 'pressure_rmse', 'pressure_r2', 'pressure_spatial_corr']:
    print(f"  {metric_name}: {metrics[metric_name]:.4f}")

print("\nTemperature Metrics:")
for metric_name in ['temp_mae', 'temp_rmse', 'temp_r2', 'temp_spatial_corr']:
    print(f"  {metric_name}: {metrics[metric_name]:.4f}")

print("\nCombined Metrics:")
for metric_name in ['combined_mae', 'combined_rmse', 'combined_r2',  'combined_spatial_corr']:
    print(f"  {metric_name}: {metrics[metric_name]:.4f}")

# Plot training and validation loss curves for both tasks
plt.figure(figsize=(15, 10))

# Plot 1: Total Loss
plt.subplot(2, 2, 1)
plt.plot(training_losses['total'], label='Training Loss')
plt.plot(validation_losses['total'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss (Combined Tasks)')
plt.legend()
plt.grid(True)

# Plot 2: Pressure Loss
plt.subplot(2, 2, 2)
plt.plot(training_losses['pressure'], label='Training Loss')
plt.plot(validation_losses['pressure'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Pressure Loss')
plt.title('Pressure Task Loss')
plt.legend()
plt.grid(True)

# Plot 3: Temperature Loss
plt.subplot(2, 2, 3)
plt.plot(training_losses['temperature'], label='Training Loss')
plt.plot(validation_losses['temperature'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Temperature Loss')
plt.title('Temperature Task Loss')
plt.legend()
plt.grid(True)

# Plot 4: Learning Rate
plt.subplot(2, 2, 4)
plt.plot(learning_rates, 'g-')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.savefig('multitask_training_curves.png')
plt.show()
print("Training curves saved as 'multitask_training_curves.png'")

# Save the trained model
torch.save(trained_model.state_dict(), 'weather_multitask_model.pth')
print("Model saved successfully.")

# Visualize predictions
visualize_multitask_prediction(trained_model, test_dataloader, num_samples=3)