#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for Deep Kalman Filter (DKF) Models

This file contains various utility functions for data preparation, model training, 
evaluation, and visualization. It includes functions for generating synthetic data, 
saving configurations, plotting results, and more.

Author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""

import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.integrate import solve_ivp

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functools import partial

import os
import yaml

from Utils.model_fcts import model_build, model_train


#--------------------------------------------------------------------------

def save_config_to_yaml(args, training_config, models_config, output_dir, generate_latex=True):
    """
    Save the configuration (args, training_config, and models_config) to a YAML file.
    Optionally, generate LaTeX tables for the configuration.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        training_config (dict): Training configuration.
        models_config (list): List of model configurations.
        output_dir (str): Directory to save the YAML file.
        generate_latex (bool): Whether to generate LaTeX tables. Default is True.
    """
    # Filter out null values from args
    filtered_args = {k: v for k, v in vars(args).items() if v is not None}

    # Combine all configurations into a single dictionary
    config = {
        "args": filtered_args,
        "training_config": training_config,
        "models_config": models_config
    }
    
    # Save the configuration to a YAML file
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Optionally generate LaTeX tables
    if generate_latex:
        latex_file_path = generate_latex_table_from_yaml(config_path)
        print(f"LaTeX tables generated and saved to: {latex_file_path}")


def generate_latex_table_from_yaml(yaml_file_path,
                                   output_file_name="config_tables.tex"):
    """
    Reads a YAML file containing configuration and generates a LaTeX table for
    each section.
    Saves the LaTeX code to a file in the same directory as the YAML file.
    
    Args:
        yaml_file_path (str): Path to the YAML file.
        output_file_name (str): Name of the output LaTeX file. Default is
        "config_tables.tex".
    
    Returns:
        str: Path to the generated LaTeX file.
    """
    # Load the YAML file
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize the LaTeX output
    latex_output = ""
    
    # Define a helper function to generate a LaTeX table for a given section
    def generate_table(section_name, section_data):
        """
        Generates a LaTeX table for a given section of the configuration.
        
        Args:
            section_name (str): Name of the section (e.g., "models").
            section_data (dict or list): Data for the section. If it's a list,
            it will generate a single table for all items.
        
        Returns:
            str: LaTeX code for the table(s).
        """
        
        # If section_data is a list, generate a single table for all models
        if isinstance(section_data, list):
            # Create a header for the table
            table_header = f"""
                            \\begin{{table}}[h!]
                            \\centering
                            \\caption{{{section_name.capitalize()} Configuration}}
                            \\label{{tab:{section_name}_config}}
                            \\begin{{tabular}}{{|l|{'l|' * len(section_data)}}}
                            \\hline
                            \\textbf{{Parameter}} & \\textbf{{Model 1}} & 
                            \\textbf{{Model 2}} & \\textbf{{Model 3}} & 
                            \\textbf{{Model 4}} \\\\ \\hline
                            """
            
            # Initialize a dictionary to store all parameters across models
            all_params = {}
            for i, model_config in enumerate(section_data):
                for param, value in model_config.items():
                    if param not in all_params:
                        all_params[param] = [""] * len(section_data)
                    all_params[param][i] = value
            
            # Generate rows for each parameter
            table_rows = ""
            for param, values in all_params.items():
                # Convert values to strings and handle special cases
                row_values = []
                for value in values:
                    if isinstance(value, (list, tuple)):
                        value = ", ".join(map(str, value))
                    elif isinstance(value, bool):
                        value = "Yes" if value else "No"
                    elif value is None:
                        value = "None"
                    else:
                        value = str(value)
                    
                    # Escape underscores for LaTeX
                    value = value.replace("_", "\\_")
                    row_values.append(value)
                
                # Add row to the table
                p = param.replace('_', '\\_')
                table_rows += f"{p} & {' & '.join(row_values)} \\\\ \\hline\n"
            
            table_footer = """
                            \\end{tabular}
                            \\end{table}
                            """
            
            return table_header + table_rows + table_footer
        
        # If section_data is a dictionary, generate a single table
        table_header = f"""
                        \\begin{{table}}[h!]
                        \\centering
                        \\caption{{{section_name.capitalize()} Configuration}}
                        \\label{{tab:{section_name}_config}}
                        \\begin{{tabular}}{{|l|l|}}
                        \\hline
                        \\textbf{{Parameter}} & \\textbf{{Value}} \\\\ \\hline
                        """
        
        table_rows = ""
        for param, value in section_data.items():
            # Convert values to strings and handle special cases
            if isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
            elif value is None:
                value = "None"
            else:
                value = str(value)
            
            # Escape underscores for LaTeX
            param = param.replace("_", "\\_")
            value = value.replace("_", "\\_")
            
            # Add row to the table
            table_rows += f"{param} & {value} \\\\ \\hline\n"
        
        table_footer = """
                        \\end{tabular}
                        \\end{table}
                        """
        
        return table_header + table_rows + table_footer
    
    # Generate tables for each section
    for section_name, section_data in config.items():
        latex_output += generate_table(section_name, section_data)
    
    # Save the LaTeX code to a file in the same directory as the YAML file
    output_file_path = os.path.join(os.path.dirname(yaml_file_path),
                                    output_file_name)
    with open(output_file_path, "w") as f:
        f.write(latex_output)
    
    return output_file_path

#--------------------------------------------------------------------------

def prepare_plot_saver(output_dir):
    """Return a partially prepared function for saving plots."""
    def save_plot(plot_func, *args, output_dir, plot_name=None, **kwargs):
        """Save the plot to a file in the specified directory."""
        #print(f"save_plot. args:{args}. kwargs: {kwargs}")

        # Use the function name as the default plot name
        if plot_name is None:
            plot_file_name = f"{plot_func.__name__}.png"
        else:
            # Append the plot_name to the function name for uniqueness
            plot_file_name = f"{plot_name}.png"
        
        plot_file_path = os.path.join(output_dir, plot_file_name)
        plot_func(*args, **kwargs)
        plt.savefig(plot_file_path)
        plt.close()
        
    return partial(save_plot, output_dir=output_dir)
    
#--------------------------------------------------------------------------

    
#--------------------------------------------------------------------------

def get_models_names(models):
    replacements = {
        "AR_DKF_1": "AR_DKF - Linear Combiner",
        "AR_DKF_2": "AR_DKF - MLP Combiner",
        "FAR_DKF_1": "FAR_DKF - Linear Combiner",
        "FAR_DKF_2": "FAR_DKF - MLP Combiner",
        "BAR_DKF_1": "BAR_DKF - Linear Combiner",
        "BAR_DKF_2": "BAR_DKF - MLP Combiner"
    }
    models_names = []
    for item in models:
        # Replace each occurrence in the current string
        for old_str, new_str in replacements.items():
            item = item.replace(old_str, new_str)
        models_names.append(item)

    return models_names

#--------------------------------------------------------------------------

def get_model_pth(dataset_name, model_name, directory='Checkpoints'):
    """
    Generates a file path for a model checkpoint based on the dataset name
    and model name.

    Parameters:
    - dataset_name (str): The name of the dataset associated with the model.
    - model_name (str): The name of the model.
    - directory (str): The directory where the model checkpoint will be saved.
                       Default is 'Checkpoints'.

    Returns:
    - str: The file path for the model checkpoint in the format 
           '{directory}/{dataset_name}_{model_name}.pth'.
    """
    def clean_string(input_string):
        """
        Cleans a string by converting all characters to lowercase and
        replacing spaces with underscores.

        Parameters:
        - input_string (str): The input string to clean.

        Returns:
        - str: The cleaned string.
        """
        # Convert to lowercase
        cleaned_string = input_string.lower()
        
        # Replace spaces with underscores
        cleaned_string = cleaned_string.replace(" ", "_")
        
        return cleaned_string
    
    model_name = clean_string(model_name)
    dataset_name = clean_string(dataset_name)
    
    # Get the absolute path to the directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    checkpoints_dir = os.path.join(base_dir, "..", directory)
    
    # Ensure the directory exists
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    # Generate the full path to the model file
    model_pth = os.path.join(checkpoints_dir, f"{dataset_name}_{model_name}.pth")
    
    return model_pth

#--------------------------------------------------------------------------

def load_training_config(config_section, custom_config=None):
    """
    Load the training configuration from a YAML file and merge it with a custom
    configuration dictionary.

    Args:
        config_section (str): The section of the YAML file to load
        (e.g., "PREDICTION_TRAINING").
        custom_config (dict, optional): A dictionary containing custom
                                        configuration parameters.
                                        If provided, it will override the
                                        values from the YAML file.

    Returns:
        dict: The merged training configuration.

    Raises:
        FileNotFoundError: If the YAML configuration file is not found.
        KeyError: If the specified section does not exist in the YAML file.
    """
    # Use an absolute path to the training_config.yaml file
    config_file = os.path.join(os.path.dirname(__file__), "..", "Config",
                               "training_conf.yaml")

    try:
        # Load the YAML configuration file
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Check if the specified section exists in the YAML file
        if config_section not in yaml_config:
            raise KeyError("The specified configuration section "
                           f"'{config_section}' does not exist in the YAML file.")

        # Get the default configuration for the specified section
        default_config = yaml_config[config_section]

        # Merge the default configuration with the custom configuration (if provided)
        if custom_config is not None:
            default_config.update(custom_config)

        return default_config

    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML configuration file '{config_file}' "
                                "was not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML configuration file: {e}")

#--------------------------------------------------------------------------

def normalize_data(data_tensor, normalization_type="standardize",
                   per_feature=False, per_sequence=False, range_=None):
    """
    Normalize a tensor of shape
    [number of sequences, sequence length, number of features].

    Args:
        data_tensor (torch.Tensor): Input tensor of shape
                                    [num_sequences, seq_length, num_features].
        normalization_type (str): Type of normalization to apply. Options:
            - "minmax": Min-Max scaling.
            - "standardize": Standardization (Z-Score normalization).
        per_feature (bool): If True, normalize each feature independently.
        per_sequence (bool): If True, normalize each sequence independently.
        range_ (tuple): Range for Min-Max scaling. Options:
            - (0, 1): Scale to [0, 1].
            - (-1, 1): Scale to [-1, 1].

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if normalization_type == "minmax":
        # Min-Max Scaling
        if per_feature:
            # Compute min and max for each feature
            min_vals = data_tensor.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]
            max_vals = data_tensor.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        elif per_sequence:
            # Compute min and max for each sequence
            min_vals = data_tensor.min(dim=1, keepdim=True)[0]
            max_vals = data_tensor.max(dim=1, keepdim=True)[0]
        else:
            # Compute min and max across the entire tensor
            min_vals = data_tensor.min()
            max_vals = data_tensor.max()

        # Perform Min-Max scaling
        normalized_data = (data_tensor - min_vals) / (max_vals - min_vals)

        # Scale to the specified range if provided
        if range_ is not None:
            normalized_data = normalized_data * (range_[1] - range_[0]) + range_[0]

    elif normalization_type == "standardize":
        # Standardization (Z-Score Normalization)
        if per_feature:
            # Compute mean and std for each feature
            mean = data_tensor.mean(dim=(0, 1), keepdim=True)
            std = data_tensor.std(dim=(0, 1), keepdim=True)
        elif per_sequence:
            # Compute mean and std for each sequence
            mean = data_tensor.mean(dim=1, keepdim=True)
            std = data_tensor.std(dim=1, keepdim=True)
        else:
            # Compute mean and std across the entire tensor
            mean = data_tensor.mean()
            std = data_tensor.std()

        # Perform standardization
        normalized_data = (data_tensor - mean) / std

    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}. "
                         "Use 'minmax' or 'standardize'.")

    return normalized_data

#--------------------------------------------------------------------------

def generate_sine_wave(frequency, amplitude=1.0, phase=0.0, duration=1.0,
                       sampling_rate=1000):
    """
    Generates a sine wave with the specified parameters and returns it as a
    2D PyTorch tensor.

    Args:
        frequency (float): Frequency of the sine wave in Hz.
        amplitude (float): Amplitude of the sine wave (default: 1.0).
        phase (float): Phase of the sine wave in radians (default: 0.0).
        duration (float): Duration of the sine wave in seconds (default: 1.0).
        sampling_rate (int): Number of samples per second (default: 1000).

    Returns:
        torch.Tensor: A 2D PyTorch tensor of shape (num_samples, 1) containing
                      the sine wave.
        numpy.ndarray: A numpy array containing the time points.

    Example:
        >>> wave, time = generate_sine_wave(frequency=5, amplitude=2.0,
                                            duration=2.0, sampling_rate=1000)
        >>> print(wave.shape)  # Output: torch.Size([2000, 1])
        >>> print(time.shape)  # Output: (2000,)
    """
    # Calculate the number of samples
    num_samples = int(sampling_rate * duration)
    
    # Generate the time points
    time = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generate the sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    
    # Reshape the sine wave to a 2D tensor with shape (num_samples, 1)
    sine_wave_tensor = torch.tensor(sine_wave, dtype=torch.float32).unsqueeze(1)
    
    return sine_wave_tensor, time

# Plot the sine wave with different colors for training, validation, and testing
def plot_sine_wave(data, l_train, l_val, title="Sine Wave"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Split the data into training, validation, and testing segments
    train_data = data[:l_train]
    val_data = data[l_train:l_train + l_val]
    test_data = data[l_train + l_val:]

    # Plot each segment with a different color
    ax.plot(train_data, lw=1, color='blue', label='Training Data')
    ax.plot(val_data, lw=1, color='green', label='Validation Data')
    ax.plot(test_data, lw=1, color='red', label='Testing Data')

    # Set labels and title
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # Add a legend
    ax.legend()

    plt.show()
#--------------------------------------------------------------------------

# Define the Lorenz system
def lorenz_system(t, state, sigma=10, beta=8/3, rho=28):
    """
    Defines the Lorenz system of differential equations.

    Parameters:
    - t (float): Time parameter (not used in this system but required by solve_ivp).
    - state (list or array): The current state of the system [x, y, z].
    - sigma (float): The sigma parameter of the Lorenz system. Default is 10.
    - beta (float): The beta parameter of the Lorenz system. Default is 8/3.
    - rho (float): The rho parameter of the Lorenz system. Default is 28.

    Returns:
    - list: The derivatives [dx/dt, dy/dt, dz/dt] of the Lorenz system.
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Generate Lorenz data
def generate_lorenz_data(seq_len, initial_point=[2, 1, 1]):
    """
    Generates data for the Lorenz system using numerical integration.

    Parameters:
    - seq_len (int): The number of time steps to generate.
    - initial_point (list or array): The initial state of the system 
      [x0, y0, z0]. Default is [2, 1, 1].

    Returns:
    - torch.Tensor: A tensor of shape (seq_len, 3) containing the 
      generated Lorenz data.
    """
    dt = 0.002
    t_span = (0, seq_len * dt)
    sol = solve_ivp(lorenz_system, t_span, initial_point,
                    t_eval=np.linspace(0, seq_len * dt, seq_len))
    data = sol.y.T  # Shape: (seq_len, 3)
    return torch.tensor(data, dtype=torch.float32)


def plot_lorenz_attractor(data, lw=0.6, l_train=-1, l_val=0, 
                          title="Lorenz Attractor", draw_axis=True,
                          draw_grid=True, show_plot=False):
    """
    Plots the Lorenz attractor with different colors for training,
    validation, and testing data.

    Parameters:
    - data (torch.Tensor): The Lorenz data of shape (seq_len, 3).
    - lw (float): Line width for the plot. Default is 0.6.
    - l_train (int): The length of the training data. Default is -1 (all data).
    - l_val (int): The length of the validation data. Default is 0.
    - title (str): The title of the plot. Default is "Lorenz Attractor".
    - draw_axis (bool): Whether to draw the axis. Default is True.
    - draw_grid (bool): Whether to draw the grid. Default is True.

    Returns:
    - None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Split the data into training, validation, and testing segments
    # and plot each segment with a different color
    if l_train != -1:
        train_data = data[:l_train]
        ax.plot(*train_data.T, lw=lw, color='blue', label='Training Data')
        if l_val != 0:
            val_data = data[l_train:l_train + l_val]
            ax.plot(*val_data.T, lw=lw, color='green', label='Validation Data')
        test_data = data[l_train + l_val:]
        ax.plot(*test_data.T, lw=lw, color='red', label='Testing Data')
    else:
        ax.plot(*data.T, lw=lw, color='blue')
        
    # Set labels and title
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    # Set axis limits to fit the data
    set_axis_limits(ax, data)

    # Add a legend
    if l_train != -1:
        ax.legend()

    # Option to disable axis and grid
    if not draw_axis:
        ax.set_axis_off()
    if not draw_grid:
        ax.grid(False)

    if show_plot == True:
        plt.show()


def animate_lorenz_attractor(data, title="Lorenz Attractor Animation",
                             draw_axis=True, draw_grid=True):
    """
    Animates the Lorenz attractor.

    Parameters:
    - data (torch.Tensor): The Lorenz data of shape (seq_len, 3).
    - title (str): The title of the animation.
                   Default is "Lorenz Attractor Animation".
    - draw_axis (bool): Whether to draw the axis. Default is True.
    - draw_grid (bool): Whether to draw the grid. Default is True.

    Returns:
    - HTML: The HTML object to display the GIF in the notebook.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the line object
    line, = ax.plot([], [], [], lw=0.6)

    # Set labels and title
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    # Set axis limits to fit the data
    set_axis_limits(ax, data)

    # Downsample data for faster animation
    ds_data = data[::10]

    # Initialize animation
    def init_animation():
        """
        Initializes the animation by setting the line data to empty.
    
        Returns:
        - tuple: The line object.
        """
        line.set_data([], [])
        line.set_3d_properties([])
        return line,
    
    # Define the update function with the line object as an argument
    def update_animation(frame, line, data):
        """
        Updates the animation frame with the current data.

        Parameters:
        - frame (int): The current frame number.
        - line (matplotlib.lines.Line3D): The line object to update.
        - data (numpy.ndarray): The Lorenz data of shape (seq_len, 3).

        Returns:
        - tuple: The line object.
        """
        line.set_data(data[:frame, 0], data[:frame, 1])
        line.set_3d_properties(data[:frame, 2])
        return line,

    # Create animation
    _ = FuncAnimation(
        fig,
        func=update_animation,
        frames=range(1, len(ds_data)),
        fargs=(line, ds_data),  # Pass the line and data as arguments
        init_func=init_animation,
        blit=True
    )

    # Option to disable axis and grid
    if not draw_axis:
        ax.set_axis_off()
    if not draw_grid:
        ax.grid(False)

    plt.show()
    return fig
    

# Helper function to set axis limits dynamically
def set_axis_limits(ax, data):
    """
    Sets the axis limits dynamically based on the data.

    Parameters:
    - ax (Axes3D): The 3D axis object.
    - data (torch.Tensor): The Lorenz data of shape (seq_len, 3).

    Returns:
    - None
    """
    # Calculate the min and max values for x, y, and z
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    z_min, z_max = data[:, 2].min(), data[:, 2].max()

    # Add some padding to the axis limits
    padding = 0.1  # 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

#--------------------------------------------------------------------------

# Function to plot MSE prediction comparison with standard error
def plot_mse_pred_comparison(models_to_compare, model_results, eval_seq_len,
                             show_plot=False):
    """
    Plot the comparison of MSE prediction for different models, showing all
    points and their associated standard errors using error bars. The x-axis
    is slightly shifted for better visualization of the error bars.

    Parameters:
    -----------
    models_to_compare : list of str
        List of model names to compare.
    model_results : dict
        Dictionary containing MSE results for each model.
    eval_seq_len : int
        Length of the evaluation sequence.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Define a small shift for the x-axis positions to avoid overlapping
    shift = 0.25  # Shift between  models
    
    # Loop through each model
    for idx, model_name in enumerate(models_to_compare):
        # Extract MSE prediction values and compute mean and standard error
        mse_pred = model_results[model_name]['mse_pred']
        mse_pred_mean = np.mean(mse_pred, axis=0)  # Mean over batches
        mse_pred_std_err = np.std(mse_pred, axis=0)  # Standard deviation

        # Time steps (With a shift x-axis for each model)
        x = np.arange(1, eval_seq_len + 1) + idx * shift

        model_name = get_models_names([model_name])[0]
        # Plot the mean MSE prediction with error bars
        plt.errorbar(x, mse_pred_mean, yerr=mse_pred_std_err,
                     label=f'{model_name}',
                     fmt='-o', capsize=2, markersize=3, elinewidth=1,
                     color=f'C{idx}' if idx < 10 else f'C{idx+1}')

    # Labels and title
    plt.xlabel('Time Step')
    plt.ylabel('MSE of the Predictions')
    plt.title('Comparison of MSE of the Predictions Across Models')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Ensure x-axis shows integer numbers
    plt.xticks(np.arange(1, eval_seq_len + 1), fontsize=8)
    # Set x-axis limits to align with the edges of the plot
    plt.xlim(0.1, 101)
    # Rotate x-axis tick labels
    plt.xticks(rotation=45)
    # Adjust the bottom margin
    #plt.subplots_adjust(left=0.06, right=0.99)

    plt.tight_layout()
    
    if show_plot == True:
        plt.show()

#--------------------------------------------------------------------------

# Function to plot MSE comparison for different models
def plot_mse_comparison(models_to_compare, model_results, show_plot=False):
    """
    Plot the comparison of MSE reconstruction and prediction for different
    models.

    Parameters:
    -----------
    models_to_compare : list of str
        List of model names to compare.
    model_results : dict
        Dictionary containing MSE results for each model.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Data for plotting
    models = models_to_compare
    mse_noise = [np.mean(model_results[model]['mse_noise']) for model in models]
    mse_recon = [np.mean(model_results[model]['mse_recon']) for model in models]
    mse_pred = [np.mean(model_results[model]['mse_pred']) for model in models]

    # Bar positions
    x = np.arange(len(models))  # x positions for the models
    width = 0.25  # Width of the bars

    # Plot MSE of the noise
    plt.bar(x - width, mse_noise, width, label='MSE of the Noise',
            color='#d5dbdb', alpha=0.8)
    # Plot MSE of the reconstruction
    plt.bar(x, mse_recon, width, label='MSE of the Reconstruction',
            color='#66c2a5', alpha=0.8)
    # Plot MSE of the prediction
    plt.bar(x + width, mse_pred, width, label='MSE of the Prediction',
            color='#fc8d62', alpha=0.8)

    # Add MSE values as annotations on the bars
    for i, (noise, recon, pred) in enumerate(zip(mse_noise, mse_recon, mse_pred)):
        # Noise bar
        plt.text(x[i] - width, noise, f"{noise:.4f}", ha='center',
                 va='bottom', fontsize=10, color='black')
        # Reconstruction bar
        plt.text(x[i], recon, f"{recon:.4f}", ha='center',
                 va='bottom', fontsize=10, color='black')
        # Prediction bar
        plt.text(x[i] + width, pred, f"{pred:.4f}", ha='center',
                 va='bottom', fontsize=10, color='black')

    # Labels and title
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    if len(models) > 1:
        plt.title('Comparison of MSE for Reconstruction and Prediction Across Models',
                  fontsize=14)
    else:
        plt.title('MSE for Reconstruction and Prediction', fontsize=14)
    
    models_names = get_models_names(models)
    plt.xticks(x, models_names, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    if show_plot == True:
        plt.show()
    
#--------------------------------------------------------------------------

# Function to evaluate the model
def evaluate_model(model_name, dataset_name, eval_data, noise_for_eval,
                   prediction_batch_size, training_seq_len, eval_seq_len,
                   model_config=None, plot_saver=None, show_plot=False):

    _, _, dimensions = eval_data.shape
    
    # Load the trained model
    model, _ = model_build(model_name, model_config)
    model_pth = get_model_pth(dataset_name, model_name)
    model.load_model(model_pth)
    model.eval()

    # Add noise to eval_data
    noisy_eval_data = eval_data.detach().clone()
    noisy_eval_data[:,:training_seq_len,:] = (eval_data[:,:training_seq_len,:]
                                              + noise_for_eval)
    
    stacked_eval_data = torch.stack((eval_data, noisy_eval_data), dim=0)  
    
    # Prepare evaluation data loader
    eval_dataset = TensorDataset(stacked_eval_data)
    eval_data_loader = DataLoader(eval_dataset, batch_size=prediction_batch_size,
                                  shuffle=False, drop_last=False)

    mse_noise_all = []
    mse_recon_all = []
    mse_pred_all = []

    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(eval_data_loader):
            ground_truth = batch[0]
            batch = batch[1]
            ground_truth = ground_truth.permute(1,0,2)
            batch = batch.permute(1,0,2)
            
            predictions = model.predict(batch[:training_seq_len,:,:],
                                        eval_seq_len)
            
            for k in range(dimensions):
                for i in range(batch.shape[1]):
                    mse_noise = ((ground_truth[:training_seq_len, i, k] - 
                                  batch[:training_seq_len, i, k].numpy())**2) 

                    mse_recon = ((ground_truth[:training_seq_len, i, k] - 
                                  predictions[:training_seq_len, i, k].numpy())**2)
                    
                    mse_pred = ((ground_truth[training_seq_len:, i, k] - 
                                 predictions[training_seq_len:, i, k].numpy())**2)
                    
                    mse_noise_all.append(mse_noise)
                    mse_recon_all.append(mse_recon)
                    mse_pred_all.append(mse_pred)

    # Plot results
    for dim in range(dimensions):
        if plot_saver != None:
            plot_saver(plot_evaluation_results, model_name, dataset_name, 
                       eval_data, noisy_eval_data, predictions,
                       training_seq_len, eval_seq_len, dim, show_plot,
                       plot_name=f"{model_name}_dim_{dim}")
        else:
            plot_evaluation_results(model_name, dataset_name, 
                                    eval_data, noisy_eval_data,
                                    predictions, training_seq_len,
                                    eval_seq_len, dim, show_plot)

    return np.array(mse_noise_all), np.array(mse_recon_all), np.array(mse_pred_all)

#--------------------------------------------------------------------------
    
# Function to plot evaluation results
def plot_evaluation_results(model_name, dataset_name, eval_data,
                            noisy_eval_data, predictions,
                            training_seq_len, eval_seq_len, dim=0,
                            show_plot=False):
    """
    Plots the evaluation results for a given model and dataset.

    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        eval_data (torch.Tensor): Ground truth data of shape
        (batch_size, seq_len, dim).
        noisy_eval_data (torch.Tensor): Noisy input data of shape
        (batch_size, seq_len, dim).
        predictions (torch.Tensor): Model predictions of shape
        (seq_len, batch_size, dim).
        training_seq_len (int): Length of the training sequence.
        eval_seq_len (int): Length of the evaluation sequence.
        dim (int): Dimension to plot.
        show_plot (bool): Whether to display the plot.
    """
    model_name = get_models_names([model_name])[0]
    
    _, nb_of_predictions, _ = predictions.shape
    
    # Select random indexes to plot
    indexes = range(0, nb_of_predictions)
    if nb_of_predictions > 3:
        indexes = random.sample(range(0, nb_of_predictions), 4)
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(len(indexes), 1, figsize=(12, 10), sharex=True,
                             sharey=True)
    if len(indexes) == 1:
        axes = [axes]  # Ensure axes is always a list

    for idx, i in enumerate(indexes):
        ax = axes[idx]
        
        # Draw separation with prediction part
        ax.axvline(x=training_seq_len-1, linestyle='-.', color='black',
                   label='Prediction Start')
        
        # Plot ground truth
        ax.plot(eval_data[i, :training_seq_len+eval_seq_len, dim],
                label='Ground Truth', linestyle=':', color='blue')

        # Plot noisy data
        ax.plot(noisy_eval_data[i, :training_seq_len, dim],
                label='Input (to reconstruct)',
                linestyle="--", marker='.', color='orange')
    
        # Plot reconstruction
        ax.plot(predictions[:training_seq_len, i, dim], alpha=0.6,
                linestyle='-', marker='.', color='green',
                label='Reconstruction')
       
        # Join the reconstruction plot to the prediction plot
        xs = range(training_seq_len-1, training_seq_len + 1)
        ys = predictions[training_seq_len-1:training_seq_len+1, i, dim].numpy()
        ax.plot(xs, ys, color='grey', alpha=0.5)
    
        # Plot predictions
        ax.plot(np.arange(training_seq_len, training_seq_len + eval_seq_len),
                predictions[training_seq_len:, i, dim], alpha=0.6,
                color='lime', linestyle='-', marker='.', label='Predictions')
    
        # Set labels and title for each subplot
        ax.set_ylabel('Value')
        ax.set_title(f'Sequence {i}')
        ax.grid(True)

        # Add legend to the first subplot only
        if idx == 0:
            ax.legend(loc='upper left')

    # Set common x-label and title for the entire figure
    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.suptitle(f'Evaluation Results for {model_name} using the {dataset_name} dataset.\n'
                 f'Reconstruction and Predictions in dimension {dim}', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show plot if requested
    if show_plot:
        plt.show()
            
#--------------------------------------------------------------------------

# Function to train the model
def train_model(loss_function, model_name, dataset_name, training_data_loader,
                val_data_loader, training_config, model_config=None,
                plot_saver=None, show_plot=False):
    """
    Train the model using the provided training and validation data loaders.

    Parameters:
    -----------
    model_name : str
        Name of the model to be trained.
    training_data_loader : DataLoader
        DataLoader for the training data.
    val_data_loader : DataLoader
        DataLoader for the validation data.

    Returns:
    --------
    model : torch.nn.Module
        The trained model.
    """
    # Build the model
    model, model_config = model_build(model_name, model_config)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=training_config['learning_rate'])

    # Train the model
    epochs = training_config['num_epochs'] #+ 10
    beta_decay = 0.95
    
    if plot_saver != None:
        plot_saver(model_train, model, optimizer,
                   training_data_loader, val_data_loader, loss_function,'cpu',
                   num_epochs=epochs, beta_decay=beta_decay,
                   show_plot=show_plot,
                   plot_name = f"training_{model_name}")
    else:
        model_train(model, optimizer, training_data_loader, val_data_loader,
                    loss_function,'cpu', num_epochs=epochs,
                    beta_decay=beta_decay, show_plot=show_plot)

    model_pth = get_model_pth(dataset_name, model_name)
    model.save_model(model_pth)
    
    return model, model_config

#--------------------------------------------------------------------------

# Function to prepare data
def prepare_data(x, e, l_train, l_val, training_seq_len, batch_size):
    l = l_train + l_val

    # Prepare training and validation data
    training_data = x[:l_train, :training_seq_len] + e[:l_train, :training_seq_len]
    training_data_tensor = torch.tensor(training_data, dtype=torch.float32)
    training_dataset = TensorDataset(training_data_tensor)
    training_data_loader = DataLoader(training_dataset, batch_size=batch_size,
                                      shuffle=True)

    val_data = x[l_train:l, :training_seq_len] + e[l_train:l, :training_seq_len]
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
    val_dataset = TensorDataset(val_data_tensor)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=True)

    return training_data_loader, val_data_loader



def split_into_sequences(data, seq_length, stride=1):
    """
    Split data into sequences of length `seq_length` with a given stride.
    
    Args:
        data (torch.Tensor): Input data of shape (total_length, features).
        seq_length (int): Length of each sequence.
        stride (int): Step size between sequences.
    
    Returns:
        torch.Tensor: Sequences of shape (num_sequences, seq_length, features).
    """
    num_sequences = (data.size(0) - seq_length) // stride + 1
    sequences = torch.stack([data[i*stride:i*stride+seq_length]
                             for i in range(num_sequences)])
    return sequences

