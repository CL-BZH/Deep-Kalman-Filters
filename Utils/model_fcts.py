#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Functions for Deep Kalman Filter (DKF) Models

This file contains utility functions for building, training, and evaluating
Deep Kalman Filter (DKF) models. 
It includes functions for model initialization, training loops,
and loss visualization.


Author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""


import torch
import yaml
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from Models.dkf_models import DKF, AR_DKF, FAR_DKF, BAR_DKF

def model_build(model_name, config=None):
    """
    Builds and initializes a model based on the provided configuration or the
    default `models_conf.yaml` file.

    Args:
        model_name (str): The name of the model to build (e.g., 'DKF', 'AR_DKF').
        config (dict, optional): A dictionary containing the configuration
                                 parameters for the model.
                                 If None, the function will read the
                                 configuration from `models_conf.yaml`.
                                 If provided, it will override the default
                                 configuration.

    Returns:
        nn.Module: An instance of the model initialized with the specified
                   configuration.

    Raises:
        ValueError: If the provided `model_name` is not supported, the
                    configuration is invalid, or the `models_conf.yaml` file
                    is missing or malformed.
    """
    # Use an absolute path to the models_conf.yaml file
    config_file = os.path.join(os.path.dirname(__file__), "..", "Config", "models_conf.yaml")
    
    # Load the default configuration from models_conf.yaml
    try:
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config is None:
                raise ValueError("models_conf.yaml is empty.")
    except FileNotFoundError:
        raise FileNotFoundError("models_conf.yaml file not found. Please provide a configuration dictionary.")
    except yaml.YAMLError:
        raise ValueError("models_conf.yaml is malformed. Please ensure it is a valid YAML file.")

    # Resolve aliases if model_name is an alias
    if model_name in yaml_config.get("aliases", {}):
        default_config = yaml_config["aliases"][model_name]
    else:
        # Get the default configuration for the model
        default_config = yaml_config["models"].get(model_name)
        model_name = model_name.rsplit('_', 1)[0]  # Remove any suffix like 'DKF_1'
    
    if default_config is None:
        raise ValueError(f"Configuration for model '{model_name}' not found in models_conf.yaml.")

    # Override default configuration with provided config if available
    if config is not None:
        default_config.update(config)

    # Dynamically import the model class from dkf_models.py
    model_class_mapping = {
        "DKF": DKF,
        "AR_DKF": AR_DKF,
        "FAR_DKF": FAR_DKF,
        "BAR_DKF": BAR_DKF
    }

    ModelClass = model_class_mapping.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Validate the configuration
    required_keys = {
        "x_dim", "z_dim", "activation", "q_b_rnn_hidden_size",
        "q_b_rnn_num_layers", "decoder_layers_dim", "dropout_p", "device",
        "learnable_init_z", "combiner_type", "combiner_layers_dim",
        "combiner_alpha_init", "combiner_alpha", "combiner_learnable_alpha",
        "combiner_output_dim", "latent_transition_layers_dim",
        "encoder_layers_dim"
    }

    # Add forward RNN parameters for AR_DKF, FAR_DKF, BAR_DKF
    if model_name in ["AR_DKF", "FAR_DKF", "BAR_DKF"]:
        required_keys.update({
            "pz_f_rnn_hidden_size",  # New name for forward RNN hidden size
            "pz_f_rnn_num_layers",   # New name for forward RNN num layers
            "learnable_init_x"        # New name for learnable_init_x
        })

    # Add additional forward RNN parameters for FAR_DKF
    if model_name == "FAR_DKF":
        required_keys.update({
            "q_f_rnn_hidden_size",  # New name for forward RNN hidden size in FAR_DKF
            "q_f_rnn_num_layers"    # New name for forward RNN num layers in FAR_DKF
        })

    missing_keys = required_keys - set(default_config.keys())
    if missing_keys:
        raise ValueError(f"Missing configuration keys for '{model_name}': {missing_keys}")

    # Initialize the model with the provided configuration
    try:
        model = ModelClass(**default_config)
    except TypeError as e:
        raise ValueError(f"Invalid configuration for '{model_name}': {e}")

    return model, default_config

#=========================================================================

def model_train(model, optimizer, train_data_loader, val_data_loader,
                loss_function, device, num_epochs=50, beta_decay=0.95,
                show_plot=False):
    """
    Train the DKF model and plot the training and validation loss.

    Args:
        model (nn.Module): The DKF model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_data_loader (DataLoader): DataLoader for the training data.
        val_data_loader (DataLoader): DataLoader for the validation data.
        loss_function (function): The loss function to use.
        device (str): The device to use for training (e.g., 'cpu' or 'cuda').
        num_epochs (int): Number of epochs to train.
        beta_decay (float): Decay factor for the beta parameter.

    Returns:
        None: The function prints the training and validation loss for each
        epoch and plots the loss curves.

    Example:
        >>> model_train(model, optimizer, train_data_loader, val_data_loader,
                        loss_function, device, num_epochs=50)
    """
    
    beta = 1
    train_losses = []  # List to store training losses
    val_losses = []    # List to store validation losses

    model.train()
    for epoch in range(num_epochs):
        train_total_loss = 0
        val_total_loss = 0
        
        # Declare the training description f-string
        training_desc = f"Epoch {epoch+1}/{num_epochs} (Training)"
        # Declare the validation description f-string
        validation_desc = f"Epoch {epoch+1}/{num_epochs} (Validation)"
        
        # Training loop
        for train_batch_idx, (train_data,) in tqdm(enumerate(train_data_loader), 
                                               total=len(train_data_loader), 
                                               desc=training_desc):
            train_data = train_data.permute(1, 0, 2).to(device)
            optimizer.zero_grad()
            (x, x_hat, x_hat_logvar, z_mean, z_logvar, z_transition_mean, 
             z_transition_logvar) = model(train_data)
            train_loss = loss_function(x, x_hat, x_hat_logvar,
                                       z_mean, z_logvar,
                                       z_transition_mean, z_transition_logvar,
                                       beta)
            train_loss.backward()
            train_total_loss += train_loss.item()
            optimizer.step()

        # Calculate average training loss for the epoch
        avg_train_loss = train_total_loss / len(train_data_loader.dataset)
        train_losses.append(avg_train_loss)
        #print('----------------------------------------------')
        #print(f'Epoch {epoch+1}.\tTraining Loss:\t\t{avg_train_loss:.3f}')

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_batch_idx, (val_data,) in tqdm(enumerate(val_data_loader), 
                                           total=len(val_data_loader), 
                                           desc=validation_desc):
                val_data = val_data.permute(1, 0, 2).to(device)
                (x, x_hat, x_hat_logvar, z_mean, z_logvar, z_transition_mean, 
                 z_transition_logvar) = model(val_data)
                val_loss = loss_function(x, x_hat, x_hat_logvar,
                                         z_mean, z_logvar,
                                         z_transition_mean, z_transition_logvar,
                                         beta)
                val_total_loss += val_loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = val_total_loss / len(val_data_loader.dataset)
        val_losses.append(avg_val_loss)
        #print(f'\t\tValidation Loss:\t{avg_val_loss:.3f}')

        # Update beta
        beta *= beta_decay

    # Plot the training and validation loss curves
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses,
                 label="Training Loss", color="blue")
    sns.lineplot(x=range(1, num_epochs + 1), y=val_losses,
                 label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss Over Epochs. "
              f"Model class: {model.__class__.__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if show_plot == True:
        plt.show()
    return fig