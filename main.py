#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for Deep Kalman Filter (DKF) Models

This script is the entry point for training and evaluating Deep Kalman Filter (DKF) models.
It handles dataset generation, model training, evaluation, and visualization of results.
The script supports both command-line and Jupyter Notebook execution.

Author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""

import argparse
import os
from datetime import datetime
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

# Import utility functions
from Utils.tools import (
    load_training_config, train_model, evaluate_model,
    split_into_sequences, normalize_data,
    plot_mse_comparison, plot_mse_pred_comparison,
    generate_sine_wave, generate_lorenz_data, 
    plot_lorenz_attractor, animate_lorenz_attractor,
    prepare_plot_saver, save_config_to_yaml
)
from Utils.loss_fcts import loss_function


def main(args=None):
    # If args is None, check if the script is being run in a Jupyter Notebook
    if args is None:
        try:
            # Check if the script is running in a Jupyter Notebook
            get_ipython  # This will raise a NameError if not in a Jupyter Notebook
            is_jupyter = True
        except NameError:
            # If get_ipython is not defined, the script is running in a terminal
            is_jupyter = False

        if is_jupyter:
            # Provide default arguments for Jupyter Notebook
            args = argparse.Namespace(
                dataset="lorenz",
                noise_var=0.006,
                training_config_section="PREDICTION_TRAINING",
                learning_rate=None,
                num_epochs=None,
                train_batch_size=None,
                val_batch_size=None,
                pred_batch_size=None,
                train_seq_len=100,
                test_seq_len=30,
                models=["DKF", "AR_DKF", "FAR_DKF"],
                normalization_type="standardize",
                per_feature=False,
                per_sequence=False,
                normalization_range=None,
                lorenz_points=20000,
                lorenz_initial_point=[2, 1, 1],
                lorenz_range=[8500, 10500],
                loss_function="weighted_mse",
                enable_animation=False,
                generate_latex=False
            )
        else:
            # Set up argument parsing for command-line execution
            parser = argparse.ArgumentParser(
                description="Train and evaluate models with configurable parameters.",
                epilog="Examples:\n"
                       "  python main.py --dataset sine --train_seq_len 50 "
                       "--test_seq_len 20 --noise_var 0.008 --num_epochs 50\n"
                       "  python main.py --dataset lorenz --train_seq_len 100 "
                       "--test_seq_len 33 --train_batch_size 64\n"
                       "  python main.py --help  # Show this help message and exit\n",
                formatter_class=argparse.RawTextHelpFormatter  # Preserve newlines in help text
            )
            
            # Dataset selection
            parser.add_argument("--dataset", type=str,
                                default="lorenz", choices=["sine", "lorenz"],
                                help="Dataset to use for training and evaluation.\n"
                                     "Options:\n"
                                     "  - sine: A simple sine wave dataset.\n"
                                     "  - lorenz: The Lorenz attractor dataset.\n"
                                     "Default: lorenz")
            
            # Noise parameters
            parser.add_argument("--noise_var", type=float, default=0.006,
                                help="Variance of the noise added to the data.\n"
                                     "Default: 0.006")
            
            # Training configuration
            parser.add_argument("--training_config_section", type=str,
                                default="PREDICTION_TRAINING",
                                choices=["SMOOTHER_TRAINING", "PREDICTION_TRAINING"],
                                help="Section of the training configuration to use.\n"
                                     "Options:\n"
                                     "  - SMOOTHER_TRAINING: Use smoother training configuration.\n"
                                     "  - PREDICTION_TRAINING: Use prediction training configuration.\n"
                                     "Default: PREDICTION_TRAINING")
            
            parser.add_argument("--learning_rate", type=float, default=None,
                                help="Learning rate for training.\n"
                                     "Default: None")
            
            parser.add_argument("--num_epochs", type=int, default=None,
                                help="Number of epochs for training.\n"
                                     "Default: None")
            
            parser.add_argument("--train_batch_size", type=int, default=None,
                                help="Batch size for training.\n"
                                     "Default: None")
            
            parser.add_argument("--val_batch_size", type=int, default=None,
                                help="Batch size for validation.\n"
                                     "Default: None")
            
            parser.add_argument("--pred_batch_size", type=int, default=None,
                                help="Batch size for prediction.\n"
                                     "Default: None")
            
            # Data parameters
            parser.add_argument("--train_seq_len", type=int, default=100,
                                help="Sequence length for training.\n"
                                     "Default: 100")
            
            parser.add_argument("--test_seq_len", type=int, default=30,
                                help="Sequence length for testing.\n"
                                     "Default: 30")
            
            # Models to compare
            parser.add_argument("--models", nargs="+",
                                default=["DKF", "AR_DKF", "FAR_DKF"],
                                help="List of models to compare.\n"
                                     "Options: DKF, AR_DKF\n"
                                     "Default: ['DKF', 'AR_DKF', 'FAR_DKF']")
            
            # Normalization parameters
            parser.add_argument("--normalization_type", type=str, default="standardize",
                                choices=["minmax", "standardize"],
                                help="Type of normalization to apply.\n"
                                     "Options:\n"
                                     "  - minmax: Min-Max scaling.\n"
                                     "  - standardize: Standardization (Z-Score normalization).\n"
                                     "Default: standardize")
            
            parser.add_argument("--per_feature", action="store_true",
                                help="If specified, normalize each feature independently.\n"
                                     "Default: False")
            
            parser.add_argument("--per_sequence", action="store_true",
                                help="If specified, normalize each sequence independently.\n"
                                     "Default: False")
            
            parser.add_argument("--normalization_range", type=float, nargs=2, default=None,
                                help="Range for Min-Max scaling (e.g., 0 1 for [0, 1]).\n"
                                     "Ignored if normalization_type is 'standardize'.\n"
                                     "Default: None")
            
            # Lorenz data generation parameters
            parser.add_argument("--lorenz_points", type=int, default=20000,
                                help="Number of points to generate for the Lorenz attractor.\n"
                                     "Default: 11000")
            
            parser.add_argument("--lorenz_initial_point", type=float, nargs=3, default=[2, 1, 1],
                                help="Initial point for the Lorenz attractor (x, y, z).\n"
                                     "Default: [2, 1, 1]")
            
            parser.add_argument("--lorenz_range", type=int, nargs=2, default=[8600, 10500],
                                help="Range of indices to select from the Lorenz attractor data.\n"
                                     "Format: start_index end_index (e.g., 5000 6200).\n"
                                     "Default: [8500, 10500]")
            
            # Loss function selection
            parser.add_argument("--loss_function", type=str, default="weighted_mse",
                                choices=["mse", "weighted_mse"],
                                help="Loss function to use for training.\n"
                                     "Options:\n"
                                     "  - mse: Mean Squared Error.\n"
                                     "  - weighted_mse: Weighted Mean Squared Error.\n"
                                     "Default: weighted_mse")
            
            # Show random examples
            parser.add_argument("--show_eval", action="store_true",
                                help="If specified, show some random examples "
                                     "of the model evaluation.\n"
                                     "Default: False")
            
            # Enable the animation
            parser.add_argument("--enable_animation", action="store_true",
                                help="If specified, enable the Lorenz attractor animation.\n"
                                     "Default: False")
            
            # Generate LaTeX tables
            parser.add_argument("--generate_latex", action="store_true",
                                help="If specified, generate LaTeX tables for the configuration.\n"
                                     "Default: False")
            
            # Parse arguments
            args = parser.parse_args()

    # Create a unique directory under 'Tests'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join("Tests", f"run_{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the plot saver function
    plot_saver = prepare_plot_saver(output_dir)
    
    # Load the training configuration from the YAML file
    training_config = load_training_config(args.training_config_section)

    # Override the training configuration with command-line arguments if provided
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        training_config["num_epochs"] = args.num_epochs
    if args.train_batch_size is not None:
        training_config["train_batch_size"] = args.train_batch_size
    if args.val_batch_size is not None:
        training_config["val_batch_size"] = args.val_batch_size
    if args.pred_batch_size is not None:
        training_config["pred_batch_size"] = args.pred_batch_size

    # Prepare the data normalizer
    normalizer = partial(normalize_data, normalization_type=args.normalization_type,
                         per_feature=args.per_feature, per_sequence=args.per_sequence,
                         range_=args.normalization_range)
    
    dataset_name = ""
    # Generate data based on the selected dataset
    if args.dataset == "sine":
        dataset_name = "Sine wave"
        # Generate sine wave data
        data, _ = generate_sine_wave(frequency=1., duration=30.0, sampling_rate=40)
        x_dim = 1
        l_train = 700
        l_val = 200
        # Plot the sine wave
        #plot_sine_wave(data, l_train, l_val, title="Sine Wave Dataset")
    elif args.dataset == "lorenz":
        dataset_name = "Lorenz attractor"
        # Generate Lorenz attractor data
        lorenz_data = generate_lorenz_data(args.lorenz_points,
                                           initial_point=args.lorenz_initial_point)
        # Plot the Lorenz attractor for the full data
        plot_saver(plot_lorenz_attractor, lorenz_data, show_plot=False,
                   plot_name="full_data")
        # Use a subset of the generated data based on the specified range
        data = lorenz_data[args.lorenz_range[0]:args.lorenz_range[1]]
        data = normalizer(data)
        # Animate the Lorenz attractor for the partial data
        if args.enable_animation:
            animate_lorenz_attractor(data)
        x_dim = 3
        data_len = len(data)
        l_train = int(data_len * (3/5))
        l_val = int(data_len * (1/20))
        # Plot the Lorenz attractor for the partial data
        plot_saver(plot_lorenz_attractor, data, 1.2, l_train, l_val,
                   title="Lorenz Attractor Dataset", show_plot=False,
                   plot_name="partial_data")

    # Special config for the input dimension
    model_config = {"x_dim": x_dim}
        
    # Split data into train, validation, and test sets
    l = l_train + l_val
    train_data = data[:l_train]
    val_data = data[l_train:l]
    test_data = data[l:]

    # Split the data into sequences
    train_sequences = split_into_sequences(train_data,
                                           seq_length=args.train_seq_len,
                                           stride=1)
    val_sequences = split_into_sequences(val_data,
                                         seq_length=args.train_seq_len//2,
                                         stride=1)
    test_sequences = split_into_sequences(test_data,
                                          seq_length=args.train_seq_len + args.test_seq_len,
                                          stride=1)

    # Add noise to sequences
    noise_std = np.sqrt(args.noise_var)
    noise_for_train = torch.randn_like(train_sequences) * noise_std
    noise_for_val = torch.randn_like(val_sequences) * noise_std
    noise_for_test = torch.randn_like(test_sequences[:, :args.train_seq_len, :]) * noise_std

    train_sequences += noise_for_train
    val_sequences += noise_for_val
    
    # Create datasets and data loaders
    training_dataset = TensorDataset(train_sequences)
    training_data_loader = DataLoader(training_dataset,
                                      batch_size=training_config["train_batch_size"],
                                      shuffle=True, drop_last=True)

    val_dataset = TensorDataset(val_sequences)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=training_config["val_batch_size"],
                                 shuffle=True, drop_last=False)

    # Define models to compare
    models_to_compare = args.models
    model_results = {model_name:
                     {'mse_noise': [], 'mse_recon': [], 'mse_pred': []}
                     for model_name in models_to_compare}

    # Create the partially applied loss function
    loss_fn = partial(loss_function, loss_type=args.loss_function)

    # Store the config of all models
    models_config = []
    
    # Train and evaluate models
    for model_name in models_to_compare:
        print("==============================================")
        print(f"Training and evaluating model: {model_name}")

        # Train the model
        _, model_conf = train_model(loss_fn, model_name, dataset_name,
                                    training_data_loader, val_data_loader,
                                    training_config, model_config,
                                    plot_saver=plot_saver,
                                    show_plot=args.show_eval)
        # Store the model config
        models_config.append(model_conf)
        
        # Evaluate the model
        mse_noise, mse_recon, mse_pred = evaluate_model(model_name,
                                                        dataset_name,
                                                        test_sequences,
                                                        noise_for_test,
                                                        training_config["pred_batch_size"],
                                                        args.train_seq_len,
                                                        args.test_seq_len,
                                                        model_config,
                                                        plot_saver=plot_saver,
                                                        show_plot=args.show_eval)

        # Store results
        model_results[model_name]['mse_noise'] = mse_noise
        model_results[model_name]['mse_recon'] = mse_recon
        model_results[model_name]['mse_pred'] = mse_pred

    # Print and plot results
    for model_name in models_to_compare:
        mse_noise = np.mean(model_results[model_name]['mse_noise'])
        mse_recon = np.mean(model_results[model_name]['mse_recon'])
        mse_pred = np.mean(model_results[model_name]['mse_pred'])
        mse_pred_std = np.std(model_results[model_name]['mse_pred'])

        print(f"Model: {model_name}")
        print(f"  MSE of the noise: {mse_noise}")
        print(f"  MSE of the reconstruction: {mse_recon}")
        print(f"  MSE of the prediction: {mse_pred}")
        print(f"  Standard deviation of the MSE of the prediction: {mse_pred_std}")

    # Plot MSE comparison
    show = False
    plot_saver(plot_mse_comparison, models_to_compare, model_results,
               plot_name="mse_comparison", show_plot=show)
    plot_saver(plot_mse_pred_comparison, models_to_compare, model_results,
               args.test_seq_len, plot_name="mse_pred_comparison",
               show_plot=show)
    
    # Save the config of all models
    save_config_to_yaml(args, training_config, models_config,
                        output_dir, generate_latex=args.generate_latex)


if __name__ == "__main__":
    main()