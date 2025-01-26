#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loader for Deep Kalman Filter (DKF) Models

This script provides utility functions for loading configuration settings from
INI files. It is used to load model-specific configurations and training parameters.

Author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""

# config_loader.py

import configparser
import os

def load_config(config_path, config_file, conf_name):
    """
    Load configuration from a '.ini' file.

    Args:
        config_path (str): Path to the directory containing the configuration file.
        config_file (str): Name of the configuration file.
        conf_name (str): Name of the section to load from the configuration file.

    Returns:
        configparser.SectionProxy: The configuration section.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        configparser.NoSectionError: If the specified section does not exist in the configuration file.
    """
    # Construct the full path to the configuration file
    full_config_path = os.path.join(config_path, config_file)

    # Check if the configuration file exists
    if not os.path.isfile(full_config_path):
        raise FileNotFoundError(f"Configuration file '{full_config_path}' not found.")

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(full_config_path)

    # Check if the specified section exists
    if conf_name not in config:
        raise configparser.NoSectionError(f"Section '{conf_name}' not found in the configuration file.")

    # Return the specified section
    return config[conf_name]


def load_model_config(model_name):
    try:
        model_config = load_config('./Config', 'dkf.ini', model_name)
    except (FileNotFoundError, configparser.NoSectionError) as e:
        print(f"Error loading model_configuration: {e}")
        return None  # Return None to indicate failure

    if model_name == 'DKF':
        # Convert config values to appropriate types
        model_config = {
            'x_dim': int(model_config['x_dim']),
            'z_dim': int(model_config['z_dim']),
            'activation': model_config['activation'],
            'bw_rnn_hidden_size': int(model_config['bw_rnn_hidden_size']),
            'bw_rnn_num_layers': int(model_config['bw_rnn_num_layers']),
            'decoder_layers_dim': [int(dim) for dim in model_config['decoder_layers_dim'].split(',')],
            'dropout_p': float(model_config['dropout_p']),
            'device': model_config['device']
        }
        return model_config
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Currently, only 'DKF' is supported.")

