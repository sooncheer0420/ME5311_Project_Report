import os
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shap

# Two explain method provided, occulusion and SHAP, however, SHAP anaylsis
# require too much computation resource, so i cannot verify its validation
# through my laptop, therefore, this code automatically conduct the occulusion
# analysis, while the SHAP analysis can also be implemented if wish 


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#------------------------------------------------------
# Define the model architecture (needed to load the model)
#------------------------------------------------------

class WeatherMultiTaskModel(nn.Module):
    def __init__(self, input_channels=2, shared_cnn_features=64, lstm_hidden=128, 
                 output_height=51, output_width=81, dropout_rate=0.01):
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

##########################################################################################
##########################################################################################
##########################################################################################



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


##########################################################################################
##########################################################################################
##########################################################################################
#------------------------------------------------------
# SHAP Implementation - Completely Model-Agnostic Approach

class WeatherModelWrapper:
    """
    Model wrapper for SHAP explainer that uses a completely model-agnostic approach.
    This avoids any gradient computations that could trigger the cudnn error.
    """
    def __init__(self, model, coords, output_type='temperature'):
        self.model = model
        self.coords = coords
        self.output_type = output_type
    
    def __call__(self, x):
        """
        Method to make the wrapper callable.
        
        Parameters:
        - x: Numpy array input of shape [batch_size, seq_len*channels*height*width]
            These are flattened inputs that need to be reshaped
            
        Returns:
        - Array of predictions for the specified coordinates
        """
        # Get original input shape from the first dimension
        batch_size = x.shape[0]
        
        # Create a list to store results
        results = []
        
        # Process each batch item
        for i in range(batch_size):
            # Reshape flattened input to original form
            # Assuming seq_len=7, channels=2, height=51, width=81
            seq_len = 7
            channels = 2
            height = 51
            width = 81
            
            # Reshape to [seq_len, channels, height, width]
            sample = x[i].reshape(seq_len, channels, height, width)
            
            # Add batch dimension and convert to tensor
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Apply model
            with torch.no_grad():
                pressure_pred, temp_pred = self.model(sample_tensor)
            
            # Get prediction at target coordinates
            if self.output_type == 'temperature':
                pred_value = temp_pred[0, self.coords[0], self.coords[1]].cpu().numpy()
            else:  # pressure
                pred_value = pressure_pred[0, self.coords[0], self.coords[1]].cpu().numpy()
            
            results.append(pred_value)
        
        return np.array(results)


class FeatureGroupSHAP:
    """
    Class to perform SHAP analysis on spatial feature groups.
    Instead of computing SHAP values for each pixel, we compute them for regions.
    """
    def __init__(self, model):
        self.model = model.eval()  # Ensure model is in eval mode
    
    def explain_prediction(self, inputs, coords=(7, 3), output_type='temperature', num_regions=5):
        """
        Explain a model prediction using SHAP, dividing the input into regions.
        
        Parameters:
        - inputs: Input tensor [batch, seq_len, channels, height, width]
        - coords: Tuple (y, x) of the coordinates to explain
        - output_type: 'temperature' or 'pressure'
        - num_regions: Number of regions to divide each spatial dimension into
        
        Returns:
        - relevance_maps: List of relevance maps for each input time step
        """
        # Create model wrapper
        model_wrapper = WeatherModelWrapper(self.model, coords, output_type)
        
        # Get input data as numpy array
        inputs_np = inputs.cpu().numpy()
        batch_size, seq_len, channels, height, width = inputs_np.shape
        
        # Create relevance maps
        relevance_maps = []
        
        # Process each time step separately
        for t in range(seq_len):
            # Get data for this time step: [batch, channels, height, width]
            time_step_data = inputs_np[:, t, :, :, :]
            
            # Create an integrated gradient approximation for each channel at this time step
            relevance_map = {
                'pressure': np.zeros((height, width)),
                'temperature': np.zeros((height, width))
            }
            
            # Create baseline (zeros) and compute gradients
            for ch_idx, ch_name in enumerate(['pressure', 'temperature']):
                # Create a zero baseline with the same shape
                baseline = np.zeros_like(time_step_data)
                
                # Create 10 steps from baseline to input for approximation
                steps = 10
                for alpha in np.linspace(0, 1, steps):
                    # Interpolate between baseline and input
                    interpolated = baseline * (1 - alpha) + time_step_data * alpha
                    
                    # Compute output for interpolated input and get gradients
                    # We need to reshape to include all time steps with zeros except the current one
                    full_input = np.zeros((batch_size, seq_len, channels, height, width))
                    full_input[:, t, :, :, :] = interpolated
                    
                    # Flatten for the model wrapper
                    flat_input = full_input.reshape(batch_size, -1)
                    
                    # Get predictions
                    pred = model_wrapper(flat_input)
                    
                    # Create a simple relevance map based on the predictions
                    # This is a simplified approach since we can't compute proper gradients
                    if alpha > 0:  # Skip baseline
                        contrib = pred / alpha  # Scale by alpha to get contribution
                        
                        # Add to relevance map for this channel
                        # We'll use the average prediction as an approximation for each pixel's importance
                        relevance_map[ch_name] += np.abs(contrib[0]) / steps
            
            # Normalize relevance maps
            for ch_name in ['pressure', 'temperature']:
                rmap = relevance_map[ch_name]
                rmap_min, rmap_max = rmap.min(), rmap.max()
                if rmap_max > rmap_min:
                    relevance_map[ch_name] = (rmap - rmap_min) / (rmap_max - rmap_min)
            
            relevance_maps.append(relevance_map)
        
        return relevance_maps


def visualize_maps(inputs, relevance_maps, model, targets, coords=(7, 3), output_file="shap_visualization.png", title_prefix="", output_type="temperature"):
    """
    Visualize the input maps alongside their corresponding relevance maps,
    plus ground truth and prediction.
    
    Args:
        inputs: Input tensor of shape [batch, seq_len, channels, height, width]
        relevance_maps: List of relevance maps from the explainer
        model: The prediction model to generate prediction
        targets: Ground truth target tensors (dict with 'target_pressure' and 'target_temp')
        coords: The coordinates being explained (for title)
        output_file: Path to save the visualization
        title_prefix: Optional prefix for the title
        output_type: 'temperature' or 'pressure'
    """
    num_timesteps = len(relevance_maps)
    
    # Create a figure with 2 more rows (ground truth and prediction)
    fig, axes = plt.subplots(num_timesteps + 2, 4, figsize=(16, 3 * (num_timesteps + 2)))
    
    # If there's only one timestep + 2 extra rows, wrap axes in a list
    if num_timesteps + 2 == 3:
        axes = [axes[i] for i in range(num_timesteps + 2)]
    
    # Loop through each timestep and plot
    for t in range(num_timesteps):
        # Get input maps for this timestep
        pressure_input = inputs[0, t, 0].cpu().detach().numpy()
        temp_input = inputs[0, t, 1].cpu().detach().numpy()
        
        # Get relevance maps for this timestep
        pressure_relevance = relevance_maps[t]['pressure']
        temp_relevance = relevance_maps[t]['temperature']
        
        # Plot pressure input
        im0 = axes[t][0].imshow(pressure_input)
        axes[t][0].set_title(f'T-{num_timesteps-t} Pressure Input')
        plt.colorbar(im0, ax=axes[t][0])
        
        # Plot pressure relevance
        im1 = axes[t][1].imshow(pressure_relevance, cmap='hot')
        axes[t][1].set_title(f'T-{num_timesteps-t} Pressure Relevance')
        plt.colorbar(im1, ax=axes[t][1])
        
        # Plot temperature input
        im2 = axes[t][2].imshow(temp_input)
        axes[t][2].set_title(f'T-{num_timesteps-t} Temperature Input')
        plt.colorbar(im2, ax=axes[t][2])
        
        # Plot temperature relevance
        im3 = axes[t][3].imshow(temp_relevance, cmap='hot')
        axes[t][3].set_title(f'T-{num_timesteps-t} Temperature Relevance')
        plt.colorbar(im3, ax=axes[t][3])
        
        # Mark the target coordinate on all plots
        for ax_idx in range(4):
            axes[t][ax_idx].plot(coords[1], coords[0], 'ro', markersize=10)
    
    # Get ground truth map based on output_type
    if output_type == 'temperature':
        ground_truth = targets['target_temp'][0].cpu().detach().numpy()
    else:  # pressure
        ground_truth = targets['target_pressure'][0].cpu().detach().numpy()
    
    # Get prediction from model
    with torch.no_grad():
        pressure_pred, temp_pred = model(inputs)
        if output_type == 'temperature':
            prediction = temp_pred[0].cpu().detach().numpy()
        else:  # pressure
            prediction = pressure_pred[0].cpu().detach().numpy()
    
    # Plot ground truth (second to last row)
    gt_idx = num_timesteps
    
    # Ground truth pressure/temperature
    im_gt = axes[gt_idx][0].imshow(ground_truth)
    axes[gt_idx][0].set_title(f'Ground Truth {output_type.capitalize()}')
    plt.colorbar(im_gt, ax=axes[gt_idx][0])
    axes[gt_idx][0].plot(coords[1], coords[0], 'ro', markersize=10)
    
    # Keep other cells in ground truth row empty
    for i in range(1, 4):
        axes[gt_idx][i].axis('off')
    
    # Plot prediction (last row)
    pred_idx = num_timesteps + 1
    
    # Model prediction
    im_pred = axes[pred_idx][0].imshow(prediction)
    axes[pred_idx][0].set_title(f'Model Prediction {output_type.capitalize()}')
    plt.colorbar(im_pred, ax=axes[pred_idx][0])
    axes[pred_idx][0].plot(coords[1], coords[0], 'ro', markersize=10)
    
    # Calculate and display error (difference between prediction and ground truth)
    error = prediction - ground_truth
    im_error = axes[pred_idx][1].imshow(error, cmap='RdBu_r')
    axes[pred_idx][1].set_title('Prediction Error')
    plt.colorbar(im_error, ax=axes[pred_idx][1])
    axes[pred_idx][1].plot(coords[1], coords[0], 'ro', markersize=10)
    
    # Keep other cells in prediction row empty
    for i in range(2, 4):
        axes[pred_idx][i].axis('off')
    
    # Add value information for the specific point
    gt_value = ground_truth[coords[0], coords[1]]
    pred_value = prediction[coords[0], coords[1]]
    error_value = pred_value - gt_value
    
    # Display the specific values as text
    value_text = f"Point ({coords[0]}, {coords[1]}):\n" \
                 f"Ground Truth: {gt_value:.4f}\n" \
                 f"Prediction: {pred_value:.4f}\n" \
                 f"Error: {error_value:.4f} ({error_value/gt_value*100:.2f}%)"
    
    axes[pred_idx][2].text(0.1, 0.5, value_text, fontsize=12)
    
    plt.suptitle(f'{title_prefix} Explanation for {output_type.capitalize()} Prediction at Coordinates {coords}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_file)
    plt.pause(1)  # Display for 1 second
    plt.close(fig)

#------------------------------------------------------
# Alternative Approach: Occlusion-based Feature Importance
#------------------------------------------------------

class OcclusionExplainer:
    """Generate feature importance maps using occlusion technique."""
    
    def __init__(self, model):
        self.model = model.eval()
    
    def explain_prediction(self, inputs, coords=(7, 3), output_type='temperature', 
                           window_size=5, stride=3):
        """
        Explain a prediction using occlusion method.
        
        Parameters:
        - inputs: Input tensor [batch, seq_len, channels, height, width]
        - coords: (y, x) coordinates to explain
        - output_type: 'temperature' or 'pressure'
        - window_size: Size of occlusion window
        - stride: Stride of occlusion window
        
        Returns:
        - relevance_maps: List of relevance maps for each input time step
        """
        batch_size, seq_len, channels, height, width = inputs.shape
        
        # Make a copy of inputs that we can modify
        inputs_tensor = inputs.clone()
        
        # Get baseline prediction without occlusion
        with torch.no_grad():
            pressure_pred, temp_pred = self.model(inputs_tensor)
            
        # Get the target prediction value
        if output_type == 'temperature':
            original_pred = temp_pred[0, coords[0], coords[1]].item()
        else:  # pressure
            original_pred = pressure_pred[0, coords[0], coords[1]].item()
        
        # Initialize relevance maps
        relevance_maps = []
        
        # Process each time step
        for t in range(seq_len):
            # Create relevance map for this time step
            relevance_map = {
                'pressure': torch.zeros((height, width), device=device),
                'temperature': torch.zeros((height, width), device=device)
            }
            
            # Process each channel
            for ch in range(channels):
                ch_name = 'pressure' if ch == 0 else 'temperature'
                
                # Apply occlusion to different parts of the image
                for i in range(0, height - window_size + 1, stride):
                    for j in range(0, width - window_size + 1, stride):
                        # Make a copy of the input
                        occluded_input = inputs_tensor.clone()
                        
                        # Store original values
                        original_values = occluded_input[0, t, ch, i:i+window_size, j:j+window_size].clone()
                        
                        # Replace with mean value to simulate "not knowing" this part
                        occluded_input[0, t, ch, i:i+window_size, j:j+window_size] = 0
                        
                        # Forward pass with occluded input
                        with torch.no_grad():
                            occ_pressure_pred, occ_temp_pred = self.model(occluded_input)
                        
                        # Get occluded prediction
                        if output_type == 'temperature':
                            occluded_pred = occ_temp_pred[0, coords[0], coords[1]].item()
                        else:  # pressure
                            occluded_pred = occ_pressure_pred[0, coords[0], coords[1]].item()
                        
                        # Calculate difference (importance)
                        importance = abs(original_pred - occluded_pred)
                        
                        # Update relevance map - accumulate in the occluded region
                        relevance_map[ch_name][i:i+window_size, j:j+window_size] += importance
                        
                        # Restore original values
                        occluded_input[0, t, ch, i:i+window_size, j:j+window_size] = original_values
            
            # Normalize relevance maps
            for ch_name in ['pressure', 'temperature']:
                rmap = relevance_map[ch_name]
                rmap_min, rmap_max = rmap.min(), rmap.max()
                if rmap_max > rmap_min:
                    relevance_map[ch_name] = (rmap - rmap_min) / (rmap_max - rmap_min)
                
                # Convert to numpy for visualization
                relevance_map[ch_name] = relevance_map[ch_name].cpu().numpy()
            
            relevance_maps.append(relevance_map)
        
        return relevance_maps


#------------------------------------------------------
# Main execution
#------------------------------------------------------

def main(explainable_type='occlusion'):
    """
    Main function to load data, model, and perform explainability analysis.
    
    Parameters:
    - explainable_type: Type of explainability analysis ('SHAP' or 'occlusion')
    """
    print("Loading weather data...")
    
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
    
    # Extract subset for testing (using all data as you did in your original code)
    x_msl_subset = x_msl
    x_t2m_subset = x_t2m - 273.15
    
    print("x_msl_subset.shape:", x_msl_subset.shape)
    print("x_t2m_subset.shape:", x_t2m_subset.shape)
    
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

    # Split into train and test sets (80% train, 20% test)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    print("Loading pre-trained model...")
    
    # Initialize the model with the same architecture
    input_height, input_width = x_msl_subset.shape[1], x_msl_subset.shape[2]  # 51, 81
    model = WeatherMultiTaskModel(
        input_channels=2,  # pressure and temperature
        shared_cnn_features=64,
        lstm_hidden=128,
        output_height=input_height,
        output_width=input_width
    )
    
    # Load the pre-trained weights
    model.load_state_dict(torch.load('weather_multitask_model.pth', map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully.")
    
    # Define points of interest for analysis
    points_of_interest = [
        (7, 3, "coastal_point"),
        (25, 40, "central_europe"),
        (40, 60, "eastern_region")
    ]
    
    # Get a batch from test data for analysis
    batch = next(iter(test_dataloader))
    inputs = batch['input'].to(device)
    targets = {
        'target_pressure': batch['target_pressure'].to(device),
        'target_temp': batch['target_temp'].to(device)
    }
    
    inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch, time_steps, channels, height, width]
    
    # Select just one sample
    input_sample = inputs[0:1]  # Keep batch dimension
    targets_sample = {
        'target_pressure': targets['target_pressure'][0:1],
        'target_temp': targets['target_temp'][0:1]
    }
    
    # Choose analysis method based on explainable_type
    if explainable_type.lower() == 'shap':
        print("\nPerforming SHAP analysis on selected points...")
        
        # Create a background dataset for SHAP (could use more samples)
        background = torch.cat([next(iter(test_dataloader))['input'].permute(0, 2, 1, 3, 4) 
                               for _ in range(3)], dim=0).to(device)
        
        for y, x, name in points_of_interest:
            print(f"\nAnalyzing point {name} at coordinates ({y}, {x})...")
            
            # For temperature prediction
            temp_wrapper = WeatherModelWrapper(model, (y, x), output_type='temperature')
            
            # Create SHAP explainer - using KernelExplainer since it's model-agnostic
            flattened_input = input_sample.cpu().numpy().reshape(1, -1)
            flattened_background = background.cpu().numpy().reshape(background.shape[0], -1)
            
            explainer = shap.KernelExplainer(temp_wrapper, flattened_background)
            shap_values = explainer.shap_values(flattened_input)
            
            # Reshape SHAP values back to original input format
            shaped_shap = np.reshape(shap_values, (input_sample.shape[1], 2, 
                                                   input_sample.shape[3], input_sample.shape[4]))
            
            # Create visualization-friendly format
            temp_importance_maps = []
            for t in range(input_sample.shape[1]):
                relevance_map = {
                    'pressure': shaped_shap[t, 0],
                    'temperature': shaped_shap[t, 1]
                }
                temp_importance_maps.append(relevance_map)
                
            # Visualize temperature
            temp_output_file = f"shap_temp_{name}_y{y}_x{x}.png"
            visualize_maps(
                input_sample, 
                temp_importance_maps,
                model,
                targets_sample,
                coords=(y, x), 
                output_file=temp_output_file,
                title_prefix="SHAP Temperature",
                output_type="temperature"
            )
            
            # Visualize pressure
            pressure_output_file = f"shap_pressure_{name}_y{y}_x{x}.png"
            visualize_maps(
                input_sample, 
                temp_importance_maps,
                model,
                targets_sample,
                coords=(y, x), 
                output_file=pressure_output_file,
                title_prefix="SHAP Pressure",
                output_type="pressure"
            )
            
            # Repeat for pressure prediction if needed
            # [You could add similar code for dedicated pressure analysis]
            
    elif explainable_type.lower() == 'occlusion':
        print("\nPerforming occlusion-based analysis on selected points...")
        
        # Create occlusion explainer
        occlusion_explainer = OcclusionExplainer(model)
        
        for y, x, name in points_of_interest:
            print(f"\nAnalyzing point {name} at coordinates ({y}, {x})...")
            
            # For temperature prediction
            temp_relevance_maps = occlusion_explainer.explain_prediction(
                input_sample, 
                coords=(y, x), 
                output_type='temperature',
                window_size=5,  # Size of occlusion window
                stride=3         # Stride for occlusion analysis
            )
            
            # Visualize temperature
            temp_output_file = f"occlusion_temp_{name}_y{y}_x{x}.png"
            visualize_maps(
                input_sample, 
                temp_relevance_maps,
                model,
                targets_sample,
                coords=(y, x), 
                output_file=temp_output_file,
                title_prefix="Occlusion Temperature",
                output_type="temperature"
            )
            
            # For pressure prediction
            pressure_relevance_maps = occlusion_explainer.explain_prediction(
                input_sample, 
                coords=(y, x), 
                output_type='pressure',
                window_size=5,
                stride=3
            )
            
            # Visualize pressure
            pressure_output_file = f"occlusion_pressure_{name}_y{y}_x{x}.png"
            visualize_maps(
                input_sample, 
                pressure_relevance_maps,
                model,
                targets_sample,
                coords=(y, x), 
                output_file=pressure_output_file,
                title_prefix="Occlusion Pressure",
                output_type="pressure"
            )
    
    else:
        print(f"Error: Unknown explainable_type '{explainable_type}'. Please use 'SHAP' or 'occlusion'.")
        return
    
    print("\nFeature importance analysis completed!")


if __name__ == "__main__":
    # Use this to specify which type of analysis to run (default is SHAP)
    # Options: 'SHAP' or 'occlusion'
    main(explainable_type='occlusion')  # Change this parameter as needed