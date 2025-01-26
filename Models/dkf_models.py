#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Kalman Filter (DKF) Models

This file contains the implementation of various Deep Kalman Filter (DKF) models,
including forward, backward, and bidirectional RNNs, as well as combiners,
encoders, decoders, and state transition modules. These models are designed for
sequence modeling and prediction tasks.


@author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DKF_Forward_RNN(nn.Module):
    """
    A forward RNN module that processes input sequences.

    Args:
        input_size (int): The number of expected features in the input to the RNN.
        hidden_size (int): The number of features in the hidden state `h`.
        num_layers (int): Number of recurrent layers.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(DKF_Forward_RNN, self).__init__()
        
        self.fw_rnn = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        """
        Forward pass of the forward RNN.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size).

        Returns:
        --------
            torch.Tensor: Output tensor of shape (seq_len, batch_size, hidden_size).
        """
        fw_rnn_out, _ = self.fw_rnn(x)
        
        return fw_rnn_out


class DKF_Backward_RNN(nn.Module):
    """
    A backward RNN module that processes input sequences in reverse order.

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
        num_layers (int): Number of recurrent layers.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(DKF_Backward_RNN, self).__init__()
        self.bw_rnn = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        """
        Forward pass of the backward RNN.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (seq_len, batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape
                          (seq_len, batch_size, hidden_size).
        """
        x = torch.flip(x, [0])
        bw_rnn_out, _ = self.bw_rnn(x)
        bw_rnn_out = torch.flip(bw_rnn_out, [0])
        
        return bw_rnn_out



class DKF_Bidirectional_RNN(nn.Module):
    """
    A bidirectional LSTM module that processes input sequences in both
    forward and backward order. The outputs can be combined using various
    methods such as concatenation, sum, weighted sum, or a linear layer.

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers in the LSTM.
        combine_type (str): The method to combine forward and backward outputs.
                            Options: "concat", "sum", "weighted_sum", "concat_linear".
        learnable_alpha (bool): Whether the alpha parameter in the weighted sum
                                should be learnable. Default: False.
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 combine_type="concat", learnable_alpha=False):
        super(DKF_Bidirectional_RNN, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)
        
        # Combine type
        self.combine_type = combine_type

        # Learnable alpha for weighted sum
        if combine_type == "weighted_sum":
            if learnable_alpha:
                self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            else:
                self.alpha = 0.5  # Fixed alpha
        else:
            self.alpha = None  # Not used for other combine types

        # Linear layer for concatenation-based combination
        if combine_type == "concat_linear":
            self.combine_linear = nn.Linear(2 * hidden_size, hidden_size)

        # Store hidden size for output size calculation
        self.hidden_size = hidden_size

    @property
    def output_size(self):
        """
        Property to get the output size of the bidirectional LSTM.

        Returns:
            int: The size of the output tensor's feature dimension.
        """
        if self.combine_type == "concat":
            # Concatenation: 2 * hidden_size (forward + backward)
            return 2 * self.hidden_size
        elif self.combine_type in ["sum", "weighted_sum", "concat_linear"]:
            # Sum, Weighted Sum, or Concatenation + Linear: hidden_size
            return self.hidden_size
        else:
            raise ValueError(f"Unknown combine_type: {self.combine_type}")

    def forward(self, x):
        """
        Forward pass of the bidirectional LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).
                          The output_size depends on the combine_type:
                          - "concat": 2 * hidden_size
                          - "sum", "weighted_sum", "concat_linear": hidden_size
        """
        # Pass through the bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        # Split the forward and backward outputs
        fw_rnn_out = lstm_out[:, :, :self.hidden_size]  # Forward output
        bw_rnn_out = lstm_out[:, :, self.hidden_size:]  # Backward output
        
        # Combine forward and backward outputs
        if self.combine_type == "concat":
            # Concatenate the outputs
            bi_rnn_out = torch.cat((fw_rnn_out, bw_rnn_out), dim=2)
        elif self.combine_type == "sum":
            # Sum the outputs
            bi_rnn_out = fw_rnn_out + bw_rnn_out
        elif self.combine_type == "weighted_sum":
            # Weighted sum of the outputs
            if self.alpha is None:
                raise ValueError("Alpha is not defined for weighted_sum. "
                                 "Please check the combine_type.")
            bi_rnn_out = self.alpha * fw_rnn_out + (1 - self.alpha) * bw_rnn_out
        elif self.combine_type == "concat_linear":
            # Concatenate and pass through a linear layer
            combined = torch.cat((fw_rnn_out, bw_rnn_out), dim=2)
            bi_rnn_out = self.combine_linear(combined)
        else:
            raise ValueError(f"Unknown combine_type: {self.combine_type}")
        
        return bi_rnn_out

#------------------------------------------------------------------------------

class DKF_Sampler(nn.Module):
    """
    A module that samples from a normal distribution with a
    given mean and variance applying the reparameterization trick.

    Args:
        device (str): The device to use ('cpu' or 'cuda').
    """
    def __init__(self, device='cpu'):
        super(DKF_Sampler, self).__init__()
        self.device = device

    def forward(self, mean, logvar):
        """
        Forward pass of the Sample module.

        Args:
            mean (torch.Tensor): Mean tensor of shape
                                 (seq_len, batch_size, z_dim).
            logvar (torch.Tensor): Log variance tensor of shape
                                   (seq_len, batch_size, z_dim).

        Returns:
            torch.Tensor: Sampled tensor of shape (seq_len, batch_size, z_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mean + eps * std
    

#------------------------------------------------------------------------------

class DKF_Combiner(ABC, nn.Module):
    """
    Abstract base class for combiners that combine the previous latent state
    `z_{t-1}` with a list of RNN hidden states.

    Args:
        z_dim (int): The dimension of the latent state `z`.
        rnn_dims (int or list): The dimension(s) of the RNN hidden states.
                                If an int, a single RNN dimension is assumed.
                                If a list, it specifies multiple RNN dimensions.
        output_dim (int): The dimension of the output tensor.
        activation (str): The activation function to use ('relu' or 'tanh').
    """
    def __init__(self, z_dim, rnn_dims, output_dim, activation='tanh'):
        super(DKF_Combiner, self).__init__()
        self.z_dim = z_dim
        
        # Ensure rnn_dims is either a single int or a list of ints
        if isinstance(rnn_dims, int):
            self.rnn_dims = [rnn_dims]  # Convert single int to a list
        elif isinstance(rnn_dims, list) and all(isinstance(dim, int) for dim in rnn_dims):
            self.rnn_dims = rnn_dims
        else:
            raise ValueError("rnn_dims must be a single integer or a list of integers.")

        self.output_dim = output_dim

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation type. Choose 'relu' or 'tanh'.")

    @abstractmethod
    def forward(self, z_prev, rnn_hs):
        """
        Abstract method for the forward pass of the combiner.

        Args:
            z_prev (torch.Tensor): Previous latent state (i.e. z at time t-1)
                                   tensor of shape (batch_size, z_dim).
            rnn_hs (list of torch.Tensor): List of RNN hidden state tensors,
                                           each of shape (batch_size, rnn_dim).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, output_dim).
        """
        pass


class DKF_CombinerMLP(DKF_Combiner):
    """
    A module that combines the previous latent state `z_{t-1}` with a list of
    RNN hidden states using a Multi-Layer Perceptron (MLP).

    Args:
        layers_dim (list): List of dimensions for the hidden layers in the MLP.
    """
    def __init__(self, z_dim, rnn_dims, output_dim, activation='tanh',
                 layers_dim=None):
        # Call the base class constructor with the required parameters
        super(DKF_CombinerMLP, self).__init__(z_dim, rnn_dims, output_dim,
                                              activation)
        
        if layers_dim is None:
            raise ValueError("layers_dim must be provided for combiner_type='mlp'")
            
        # Input dimension: z_dim + sum(rnn_dims) (concatenated input)
        input_dim = self.z_dim + sum(self.rnn_dims)
        
        # Define the MLP layers
        layers = []
        for i, dim in enumerate(layers_dim):
            if i == 0:
                # First layer: input_dim -> layers_dim[0]
                layers.append(nn.Linear(input_dim, layers_dim[0]))
            else:
                # Intermediate layers: layers_dim[i-1] -> layers_dim[i]
                layers.append(nn.Linear(layers_dim[i-1], layers_dim[i]))
            layers.append(self.activation)  # Add activation after each layer
        
        # Final layer: layers_dim[-1] -> output_dim
        layers.append(nn.Linear(layers_dim[-1], self.output_dim))
        
        # Combine layers into an MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_prev, rnn_hs):
        """
        Forward pass of the Combiner.

        Args:
            z_prev (torch.Tensor): Previous latent state (i.e. z at time t-1)
                                   tensor of shape (batch_size, z_dim).
            rnn_hs (list of torch.Tensor): List of RNN hidden state tensors,
                                           each of shape (batch_size, rnn_dim).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, output_dim).
        """
        # Concatenate z_prev and all RNN hidden states along the feature dimension
        combined_input = [z_prev] + rnn_hs  # List of tensors to concatenate
        combined_input = torch.cat(combined_input, dim=1)
        
        # Pass through the MLP
        combined_output = self.mlp(combined_input)
        
        return combined_output


class DKF_CombinerLinear(DKF_Combiner):
    """
    A module that combines the previous latent state `z_{t-1}` with a list of
    RNN hidden states using a linear combination.

    Args:
        alpha_init (float or list): The initial value(s) of alpha for the first
                                    forward pass.
                                    If a single float, the same alpha is used
                                    for all RNNs.
                                    If a list, it must have the same length as
                                    rnn_dims.
        alpha (float or list): The value(s) of alpha to use for subsequent
                               forward passes.
                               If a single float, the same alpha is used for
                               all RNNs.
                               If a list, it must have the same length as
                               rnn_dims.
        learnable_alpha (bool or list): Whether to make alpha_init and alpha
                                        learnable.
                                        If a single bool, the same setting is
                                        used for all RNNs.
                                        If a list, it must have the same length
                                        as rnn_dims.
    """
    def __init__(self, z_dim, rnn_dims, output_dim, activation='tanh',
                 alpha_init=0.5, alpha=0.5, learnable_alpha=True):
        # Call the base class constructor with the required parameters
        super(DKF_CombinerLinear, self).__init__(z_dim, rnn_dims, output_dim,
                                                 activation)

        # Linear layers to map z_dim to each RNN dimension
        self.input_layers = nn.ModuleList([
            nn.Linear(self.z_dim, rnn_dim) for rnn_dim in self.rnn_dims
        ])
        
        # Initialize alpha values
        if isinstance(alpha_init, (float, int)):
            # Repeat the same alpha for all RNNs
            alpha_init = [alpha_init] * len(self.rnn_dims)
        if isinstance(alpha, (float, int)):
            # Repeat the same alpha for all RNNs
            alpha = [alpha] * len(self.rnn_dims)
        if isinstance(learnable_alpha, bool):
            # Repeat the same setting for all RNNs
            learnable_alpha = [learnable_alpha] * len(self.rnn_dims)

        # Ensure all lists have the same length as rnn_dims
        if (len(alpha_init) != len(self.rnn_dims) or 
            len(alpha) != len(self.rnn_dims) or 
            len(learnable_alpha) != len(self.rnn_dims)):
            raise ValueError("alpha_init, alpha, and learnable_alpha must have"
                             " the same length as rnn_dims.")

        # Define alpha parameters for each RNN
        self.alpha_init = nn.ParameterList()
        self.alpha_run = nn.ParameterList()
        for i, (init_val, run_val, learnable) in enumerate(zip(alpha_init,
                                                               alpha,
                                                               learnable_alpha)):
            if learnable:
                # Make alpha_init and alpha learnable parameters
                self.alpha_init.append(nn.Parameter(torch.tensor(init_val,
                                                                 dtype=torch.float32)))
                self.alpha_run.append(nn.Parameter(torch.tensor(run_val,
                                                                dtype=torch.float32)))
            else:
                # Use fixed values for alpha_init and alpha
                self.alpha_init.append(torch.tensor(init_val, dtype=torch.float32))
                self.alpha_run.append(torch.tensor(run_val, dtype=torch.float32))

        # Initialize alpha to the initial values
        self.alpha = self.alpha_init

        # Linear layer to map the combined tensor to the desired output dimension
        self.output_layer = nn.Linear(sum(self.rnn_dims), self.output_dim)

    def forward(self, z_prev, rnn_hs):
        """
        Forward pass of the Combiner.

        Args:
            z_prev (torch.Tensor): Previous latent state (i.e. z at time t-1)
                                   tensor of shape (batch_size, z_dim).
            rnn_hs (list of torch.Tensor): List of RNN hidden state tensors,
                                           each of shape (batch_size, rnn_dim).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, output_dim).
        """
        # Combine z_prev and all RNN hidden states using the current alpha values
        combined = []
        for i, rnn_h in enumerate(rnn_hs):
            # Map z_prev to the current RNN dimension
            z_mapped = self.activation(self.input_layers[i](z_prev))
            # Combine with the RNN hidden state using the corresponding alpha
            combined.append(self.alpha[i] * z_mapped + (1 - self.alpha[i]) * rnn_h)

        # Concatenate all combined tensors
        combined = torch.cat(combined, dim=1)

        # Update alpha to the running values after the first forward pass
        self.alpha = self.alpha_run

        # Map the combined tensor to the desired output dimension
        output = self.output_layer(combined)
        
        return output


class DKF_CombinerFactory(nn.Module):
    """
    A factory class that creates the appropriate combiner based on the type.

    Args:
        z_dim (int): The dimension of the latent state `z`.
        rnn_dims (int or list): The dimension(s) of the RNN hidden states.
                                If an int, a single RNN dimension is assumed.
                                If a list, it specifies multiple RNN dimensions.
        output_dim (int): The dimension of the output tensor.
        activation (str): The activation function to use ('relu' or 'tanh').
        combiner_type (str): The type of combiner to use ('linear' or 'mlp').
        alpha_init (float or list): The initial value(s) of alpha for the first
                                    forward pass (only used if combiner_type='linear').
        alpha (float or list): The value(s) of alpha to use for subsequent
                               forward passes (only used if combiner_type='linear').
        learnable_alpha (bool or list): Whether to make alpha_init and alpha learnable
                                        (only used if combiner_type='linear').
        layers_dim (list): List of dimensions for the hidden layers in the MLP
                          (only used if combiner_type='mlp').
    """
    def __init__(self, z_dim, rnn_dims, output_dim, activation='tanh',
                 combiner_type='linear', alpha_init=0.5, alpha=0.5,
                 learnable_alpha=True, layers_dim=None):
        super(DKF_CombinerFactory, self).__init__()
        
        # Validate combiner type
        if combiner_type not in ['linear', 'mlp']:
            raise ValueError("combiner_type must be either 'linear' or 'mlp'")

        self.output_dim = output_dim
        
        if combiner_type == 'linear':
            # Initialize the linear combiner
            self.combiner = DKF_CombinerLinear(
                z_dim=z_dim,
                rnn_dims=rnn_dims,
                output_dim=output_dim,
                activation=activation,
                alpha_init=alpha_init,
                alpha=alpha,
                learnable_alpha=learnable_alpha
            )
        elif combiner_type == 'mlp':
            if layers_dim is None:
                raise ValueError("layers_dim must be provided for combiner_type='mlp'")
            # Initialize the MLP combiner
            self.combiner = DKF_CombinerMLP(
                z_dim=z_dim,
                rnn_dims=rnn_dims,
                output_dim=output_dim,
                activation=activation,
                layers_dim=layers_dim
            )

    def forward(self, z_prev, rnn_hs):
        """
        Forward pass of the Combiner.

        Args:
            z_prev (torch.Tensor): Previous latent state (i.e. z at time t-1)
                                   tensor of shape (batch_size, z_dim).
            rnn_hs (list of torch.Tensor): List of RNN hidden state tensors,
                                           each of shape (batch_size, rnn_dim).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, output_dim).
        """
        return self.combiner(z_prev, rnn_hs)

#------------------------------------------------------------------------------

class DKF_Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, encoder_layers_dim, activation='tanh',
                  device='cpu'):
        super(DKF_Encoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoder_layers_dim = encoder_layers_dim
        self.activation = activation
        self.device = device
        
        # Activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        
        input_dim = self.input_dim
        
        # Define the MLP layers for the encoder
        layers = []
        for i, dim in enumerate(encoder_layers_dim):
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation_fn)
            input_dim = dim
        
        # Final layers for mean and logvar
        layers.append(nn.Linear(input_dim, z_dim * 2))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, combined):
        """
        Forward pass of the Encoder.

        Args:
            combined (torch.Tensor): Input tensor which is the output tensor
                                     from the Combiner.

        Returns:
            tuple: A tuple containing:
                - z_mean (torch.Tensor): Mean of the latent states of shape
                                         (seq_len, batch_size, z_dim).
                - z_logvar (torch.Tensor): Log variance of the latent states
                                           of shape (seq_len, batch_size, z_dim).
        """
        # Pass through the MLP
        h = self.mlp(combined)
        mu = h[:, :self.z_dim]  # Mean
        logvar = h[:, self.z_dim:]  # Log variance

        return mu, logvar

#------------------------------------------------------------------------------
        
class DKF_Decoder(nn.Module):
    """
    A decoder module that maps latent states `z` to output states `x`.

    Args:
        dim_in (int): The dimension of the input latent state `z`.
        dim_out (int): The dimension of the output state `x`.
        layers_dim (list): List of dimensions for the hidden layers.
        activation (str): The activation function to use ('relu' or 'tanh').
        dropout_p (float): Dropout probability.
    """
    def __init__(self, dim_in, dim_out, layers_dim, activation='tanh', dropout_p=0):
        super(DKF_Decoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        #print(f"Decoder. dim_in: {dim_in}. dim_out: {dim_out}")
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        
        # Define the MLP layers
        layers = []
        input_dim = dim_in
        for i, dim in enumerate(layers_dim):
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        # Final layers for mean and logvar
        layers.append(nn.Linear(input_dim, dim_out * 2))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass of the Decoder.

        Args:
            z (torch.Tensor): Latent state tensor of shape
                              (seq_len, batch_size, dim_in).

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): Mean of the output tensor of shape
                                       (seq_len, batch_size, dim_out).
                - logvar (torch.Tensor): Log variance of the output tensor of
                                         shape (seq_len, batch_size, dim_out).
        """
        # Pass through the MLP layers
        h = z
        for layer in self.mlp:
            h = layer(h)

        # Split mean and logvar
        mean = h[:, :, :self.dim_out]
        logvar = h[:, :, self.dim_out:]
        return mean, logvar

#------------------------------------------------------------------------------

class DKF_LatentStateTransition(nn.Module):
    """
    A module that computes the state transition probability `p(z_t | z_{t-1})`.

    Args:
        z_dim (int): The dimension of the latent state `z`.
        layers_dim (list): List of dimensions for the hidden layers in the MLP.
    """
    def __init__(self, z_dim, layers_dim):
        super(DKF_LatentStateTransition, self).__init__()
        
        self.z_dim = z_dim
        self.layers_dim = layers_dim

        # Define the MLP layers
        layers = []
        input_dim = z_dim
        for i, dim in enumerate(layers_dim):
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        
        # Final layer (Output both mean and logvar)
        layers.append(nn.Linear(input_dim, z_dim * 2))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass of the LatentStateTransition.

        Args:
            z (torch.Tensor): Latent state sample at time t-1.
                              Tensor of shape (seq_len, batch_size, z_dim).

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): Mean of the state transition probability
                                       of shape (seq_len, batch_size, z_dim).
                - logvar (torch.Tensor): Log variance of the state transition
                                         probability of shape
                                         (seq_len, batch_size, z_dim).
        """
        h = self.mlp(z)
        z_mean = h[:, :, :self.z_dim]
        z_logvar = h[:, :, self.z_dim:]
        
        return z_mean, z_logvar

#------------------------------------------------------------------------------

class AR_DKF_LatentStateTransition(nn.Module):
    """
    A module that computes the state transition probability `p(z_t | z_{t-1})`.

    Args:
        z_dim (int): The dimension of the latent state `z`.
        h_x_dim (int): The dimension of the observation `x` transformed by
                       the forwards RNN.
        layers_dim (list): List of dimensions for the hidden layers in the MLP.
        use_z_t_minus_1 (bool): Whether to use `z_t_minus_1` in the computation.
                                If False, the computation will only use `h_x_t_minus_1`.
                                Default is True.
    """
    def __init__(self, z_dim, h_x_dim, layers_dim, use_z_t_minus_1=True):
        super(AR_DKF_LatentStateTransition, self).__init__()
        
        self.z_dim = z_dim
        self.h_x_dim = h_x_dim
        self.layers_dim = layers_dim
        self.use_z_t_minus_1 = use_z_t_minus_1

        # Define the MLP for state transition
        layers = []
        input_dim = z_dim + h_x_dim if use_z_t_minus_1 else h_x_dim
        for i, dim in enumerate(layers_dim):
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        
        # Final layer
        layers.append(nn.Linear(input_dim, z_dim * 2))  # Output both mean and logvar
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t_minus_1, h_x_t_minus_1):
        """
        Forward pass of the LatentStateTransition.

        Args:
            z_t_minus_1 (torch.Tensor or None): Latent state sample at time t-1.
                                                Tensor of shape
                                                (seq_len, batch_size, z_dim).
                                                Or None if use_z_t_minus_1 is
                                                False.
            h_x_t_minus_1 (torch.Tensor): x lagged by 1 passed through a
                                          forward RNN.
                                          Tensor of shape
                                          (seq_len, batch_size, h_x_dim).

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): Mean of the state transition probability
                                       of shape (seq_len, batch_size, z_dim).
                - logvar (torch.Tensor): Log variance of the state transition
                                         probability of shape
                                         (seq_len, batch_size, z_dim).
        """
        if self.use_z_t_minus_1:
            # Concatenate z_t_minus_1 and h_x_t_minus_1
            zx = torch.cat((z_t_minus_1, h_x_t_minus_1), -1)
        else:
            # Use only h_x_t_minus_1
            zx = h_x_t_minus_1
        
        # Pass through the MLP
        h = self.mlp(zx)
        
        # Compute mean and log variance
        z_mean = h[:, :, :self.z_dim]
        z_logvar = h[:, :, self.z_dim:]
        
        return z_mean, z_logvar

#------------------------------------------------------------------------------

class DKF_Base(ABC, nn.Module):
    def __init__(self, x_dim, z_dim=16, activation='tanh', q_b_rnn_hidden_size=64, 
                 q_b_rnn_num_layers=1, decoder_layers_dim=[64, 64], dropout_p=0,
                 device='cpu', learnable_init_z=True, combiner_type='linear',
                 combiner_layers_dim=64, combiner_alpha_init=0.1,
                 combiner_alpha=0.5, combiner_learnable_alpha=True,
                 combiner_output_dim=64, encoder_layers_dim=[128, 64]):
        super().__init__()

        # General parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        self.activation = activation
        self.device = device
        self.learnable_init_z = learnable_init_z
        
        # Backward RNN
        self.q_b_rnn_input_size = x_dim
        self.q_b_rnn_hidden_size = q_b_rnn_hidden_size
        self.q_b_rnn_num_layers = q_b_rnn_num_layers

        # Generation
        self.decoder_layers_dim = decoder_layers_dim

        # Combiner parameters
        self.combiner_type = combiner_type
        self.combiner_layers_dim = combiner_layers_dim
        self.combiner_alpha_init = combiner_alpha_init
        self.combiner_alpha = combiner_alpha
        self.combiner_learnable_alpha = combiner_learnable_alpha

        # Encoder parameters
        self.encoder_layers_dim = encoder_layers_dim

        # Combiner output dimension
        self.combiner_output_dim = combiner_output_dim

        # Instantiate shared components
        self.build_shared_components()

    def build_shared_components(self):
        """
        Initializes shared components of the DKF model.
        """
        # Define Z_0 as learnable parameters or fixed tensors
        if self.learnable_init_z:
            self.Z_0 = nn.Parameter(torch.zeros(1, 1, self.z_dim))
        else:
            self.register_buffer('Z_0', torch.zeros(1, 1, self.z_dim))
            
        
        self.q_b_rnn = DKF_Backward_RNN(
            input_size=self.q_b_rnn_input_size,
            hidden_size=self.q_b_rnn_hidden_size,
            num_layers=self.q_b_rnn_num_layers
        )
        
        self.encoder = DKF_Encoder(
            input_dim = self.combiner_output_dim,
            z_dim=self.z_dim,
            encoder_layers_dim=self.encoder_layers_dim,
            activation=self.activation,
            device=self.device
        )
        
        self.decoder = DKF_Decoder(
            dim_in=self.z_dim,
            dim_out=self.x_dim,
            layers_dim=self.decoder_layers_dim,
            activation=self.activation,
            dropout_p=self.dropout_p
        )
        
        self.sampler = DKF_Sampler(self.device)

    @abstractmethod
    def build_specific_components(self):
        """
        Abstract method to be implemented by subclasses to initialize
        model-specific components.
        """
        pass

    def forward(self, x):
        """
        Performs a forward pass through the DKF model.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, x_dim).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Original input tensor.
                - x_hat (torch.Tensor): Reconstructed output tensor.
                - z_mean (torch.Tensor): Mean of the latent states.
                - z_logvar (torch.Tensor): Log variance of the latent states.
                - z_transition_mean (torch.Tensor): Mean of the state
                  transition probability.
                - z_transition_logvar (torch.Tensor): Log variance of the state
                  transition probability.
        """
        _, batch_size, _ = x.shape

        z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
        
        # Get the sequence transformed by the backward RNN
        q_b_rnn_out = self.q_b_rnn(x)
        
        seq_len, batch_size, _ = q_b_rnn_out.shape

        # Output variables
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)

        sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0

        # Work throught the whole sequence
        for t in range(0, seq_len):
            
            # Combine z_{t-1} with output of the backward RNN for step t
            combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
            
            # Pass through the Encoder
            mu, logvar = self.encoder(combined)
            
            self.z_mean[t,:,:] = mu
            self.z_logvar[t,:,:] = logvar
            sampled_z_t = self.sampler(mu, logvar)
            self.sampled_z[t,:,:] = sampled_z_t
        
        # Compute the mean and logvar of the state transition probability
        # p(z_t | z_{t-1})
        z_t_minus_1 = torch.cat([z_0, self.sampled_z[:-1, :,:]], 0)
        self.z_transition_mean, self.z_transition_logvar = self.p_z(z_t_minus_1)

        # Reconstruct x from z
        self.x_hat, self.x_hat_logvar = self.decoder(self.sampled_z)

        return (x, self.x_hat, self.x_hat_logvar, self.z_mean, self.z_logvar,
                self.z_transition_mean, self.z_transition_logvar)

    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, x_dim).
            num_steps (int): Number of future steps to predict.

        Returns:
            torch.Tensor: Tensor of shape (num_steps, batch_size, x_dim)
            containing the predicted future steps.
        """
        with torch.no_grad():
            seq_len, batch_size, x_dim = x.shape
            z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
            q_b_rnn_out = self.q_b_rnn(x)
            
            # Output variables
            self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
            
            sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0
            
            # Work throught the whole sequence
            for t in range(0, seq_len):
                
                # Combine z_{t-1} with output of the backward RNN for step t
                combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
                
                # Pass through the Encoder
                mu, logvar = self.encoder(combined)
                sampled_z_t = self.sampler(mu, logvar)
                self.sampled_z[t,:,:] = sampled_z_t

            # Reconstruct x from z
            x_hat, _ = self.decoder(self.sampled_z)
            
            # Start with the last inferred latent state
            z_pred = self.sampled_z[-1:, :, :]

            # Start to predict x at the end of the given sequence
            predictions = torch.zeros(num_steps, batch_size, x_dim).to(self.device)
            for s in range(num_steps):
                # Get the parameters (mean and variance) of the transition
                # distribution p(z_t|z_{t-1})
                z_pred_mean, z_pred_logvar = self.p_z(z_pred)

                # Sample from p(z_t|z_{t-1}) distribution
                z_pred = self.sampler(z_pred_mean, z_pred_logvar)
                
                x_pred, _ = self.decoder(z_pred)
                
                predictions[s,:,:] = x_pred

        # Append predictions to the reconstructed x
        extended_x = torch.cat([x_hat, predictions], dim=0)
        
        return extended_x

    def save_model(self, filename):
        torch.save(self.to('cpu').state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

#------------------------------------------------------------------------------

class DKF(DKF_Base):
    def __init__(self, x_dim, z_dim=8, activation='tanh', q_b_rnn_hidden_size=64, 
                 q_b_rnn_num_layers=1, decoder_layers_dim=[64, 64], dropout_p=0,
                 device='cpu', learnable_init_z=True, combiner_type='linear',
                 combiner_layers_dim=[64], combiner_alpha_init=0.1,
                 combiner_alpha=0.5, combiner_learnable_alpha=True,
                 combiner_output_dim=64, encoder_layers_dim=[128, 64],
                 latent_transition_layers_dim=[128]):
        super().__init__(x_dim, z_dim, activation, q_b_rnn_hidden_size,
                         q_b_rnn_num_layers, decoder_layers_dim, dropout_p,
                         device, learnable_init_z, combiner_type,
                         combiner_layers_dim, combiner_alpha_init,
                         combiner_alpha, combiner_learnable_alpha,
                         combiner_output_dim, encoder_layers_dim)
        
        self.latent_transition_layers_dim = latent_transition_layers_dim
        self.build_specific_components()

    def build_specific_components(self):
        """
        Initializes specific components for the DKF model.
        """
        self.combiner = DKF_CombinerFactory(
            z_dim=self.z_dim,
            rnn_dims=self.q_b_rnn_hidden_size,
            output_dim=self.combiner_output_dim,
            activation=self.activation,
            combiner_type=self.combiner_type,
            alpha_init=self.combiner_alpha_init,
            alpha=self.combiner_alpha,
            learnable_alpha=self.combiner_learnable_alpha,
            layers_dim=self.combiner_layers_dim
        )
        
        self.p_z = DKF_LatentStateTransition(
            z_dim=self.z_dim,
            layers_dim=self.latent_transition_layers_dim
        )
        
#------------------------------------------------------------------------------
    
class AR_DKF(DKF_Base):
    def __init__(self, x_dim, z_dim=8, activation='tanh', pz_f_rnn_hidden_size=64, 
                  pz_f_rnn_num_layers=1, q_b_rnn_hidden_size=64, q_b_rnn_num_layers=1, 
                  decoder_layers_dim=[64, 64], dropout_p=0, device='cpu',
                  learnable_init_z=True, learnable_init_x=False, combiner_type='linear',
                  combiner_layers_dim=[64], combiner_alpha_init=0.1,
                  combiner_alpha=0.5, combiner_learnable_alpha=True,
                  combiner_output_dim=64, encoder_layers_dim=[128, 64],
                  latent_transition_layers_dim=[128]):
        self.pz_f_rnn_hidden_size = pz_f_rnn_hidden_size
        self.pz_f_rnn_num_layers = pz_f_rnn_num_layers
        self.learnable_init_x = learnable_init_x
        self.latent_transition_layers_dim = latent_transition_layers_dim
        
        super().__init__(x_dim, z_dim, activation, q_b_rnn_hidden_size,
                          q_b_rnn_num_layers, decoder_layers_dim, dropout_p,
                          device, learnable_init_z, combiner_type,
                          combiner_layers_dim, combiner_alpha_init,
                          combiner_alpha, combiner_learnable_alpha,
                          combiner_output_dim, encoder_layers_dim)
        
        self.build_specific_components()

    def build_specific_components(self):
        """
        Initializes specific components for the AR_DKF model.
        """
        self.combiner = DKF_CombinerFactory(
            z_dim=self.z_dim,
            rnn_dims=self.q_b_rnn_hidden_size,
            output_dim=self.combiner_output_dim,
            activation=self.activation,
            combiner_type=self.combiner_type,
            alpha_init=self.combiner_alpha_init,
            alpha=self.combiner_alpha,
            learnable_alpha=self.combiner_learnable_alpha,
            layers_dim=self.combiner_layers_dim
        )
        
        self.pz_f_rnn = DKF_Forward_RNN(
            input_size=self.x_dim,
            hidden_size=self.pz_f_rnn_hidden_size,
            num_layers=self.pz_f_rnn_num_layers
        )
        
        self.p_z = AR_DKF_LatentStateTransition(
            z_dim=self.z_dim,
            h_x_dim=self.pz_f_rnn_hidden_size,
            layers_dim=self.latent_transition_layers_dim
        )

        # Define X_0 as learnable parameters or fixed tensors
        if self.learnable_init_x:
            self.X_0 = nn.Parameter(torch.zeros(1, 1, self.x_dim))
        else:
            self.register_buffer('X_0', torch.zeros(1, 1, self.x_dim))

    def forward(self, x):
        """
        Performs a forward pass through the AR_DKF model.
        """
        _, batch_size, _ = x.shape

        # Use the learnable or fixed parameters Z_0 and X_0
        z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
        x_0 = self.X_0.expand(1, batch_size, self.x_dim).to(self.device)
        
        # Pass the sequence [x_0, x_1, ..., x_{T-1}] to the Forward RNN
        x_lagged = torch.cat((x_0, x[:-1, :, :]), 0)
        h_x_lagged = self.pz_f_rnn(x_lagged)
        
        # Get the sequence [x_1, ..., x_T] transformed by the backward RNN
        q_b_rnn_out = self.q_b_rnn(x)

        seq_len, batch_size, _ = q_b_rnn_out.shape

        # Output variables
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)

        sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0

        # Work throught the whole sequence
        for t in range(0, seq_len):
            
            # Combine z_{t-1} with output of the backward RNN for step t
            combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
            
            # Pass through the Encoder
            mu, logvar = self.encoder(combined)
            
            self.z_mean[t,:,:] = mu
            self.z_logvar[t,:,:] = logvar
            sampled_z_t = self.sampler(mu, logvar)
            self.sampled_z[t,:,:] = sampled_z_t
        
        # Get the mean and logvar of the state transition probability
        # p(z_t|z_{t-1}, x_{0:t-1})
        z_lagged = torch.cat([z_0, self.sampled_z[:-1, :,:]], 0)
        self.z_transition_mean, self.z_transition_logvar = self.p_z(z_lagged,
                                                                    h_x_lagged)

        # Reconstruct x from z
        self.x_hat, self.x_hat_logvar = self.decoder(self.sampled_z)

        return (x, self.x_hat, self.x_hat_logvar, self.z_mean, self.z_logvar,
                self.z_transition_mean, self.z_transition_logvar)
    
    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.
        """
        with torch.no_grad():
            sampler = DKF_Sampler(self.device)
            seq_len, batch_size, x_dim = x.shape
            
            # Use the learnable or fixed parameters Z_0
            z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
 
            q_b_rnn_out = self.q_b_rnn(x)
            
            #-----------------------------------------------------------
            # Output variables
            self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
            
            sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0
            
            # Work throught the whole sequence
            for t in range(0, seq_len):
                
                # Combine z_{t-1} with output of the backward RNN for step t
                combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
                
                # Pass through the Encoder
                mu, logvar = self.encoder(combined)
                sampled_z_t = self.sampler(mu, logvar)
                self.sampled_z[t,:,:] = sampled_z_t

            # Reconstruct x from z
            x_hat, _ = self.decoder(self.sampled_z)
 
            #------------------------------------------------------------
            
            # Pass x throught the forward RNN
            h_x_lagged = self.pz_f_rnn(x)
            
            # Start with the last inferred latent state
            z_pred = self.sampled_z[-1:, :, :]
 
            predictions = torch.zeros(num_steps, batch_size, x_dim).to(self.device)
            for s in range(num_steps):
                # Get the parameters (mean and variance) of the transition
                # distribution p(z_t|z_{t-1}) (normal distribution)
                z_pred_mean, z_pred_logvar = self.p_z(z_pred,
                                                      h_x_lagged[-1:, :, :])

                # sample from p(z_t|z_{t-1}, x_{0:t-1}) distribution
                z_pred = sampler(z_pred_mean, z_pred_logvar)
                
                x_pred, _ = self.decoder(z_pred)
                
                x = torch.cat((x[1:,:,:], x_pred), dim=0)
                h_x_lagged = self.pz_f_rnn(x)
                
                predictions[s,:,:] = x_pred

            # Append predictions to the reconstructed x
            extended_x = torch.cat([x_hat, predictions], dim=0)
        
        return extended_x
    
#------------------------------------------------------------------------------
    
class FAR_DKF(DKF_Base):
    def __init__(self, x_dim, z_dim=8, activation='tanh', pz_f_rnn_hidden_size=64, 
                 pz_f_rnn_num_layers=1, q_b_rnn_hidden_size=64, q_b_rnn_num_layers=1, 
                 q_f_rnn_hidden_size=64, q_f_rnn_num_layers=1, decoder_layers_dim=[64, 64], 
                 dropout_p=0, device='cpu', learnable_init_z=True, learnable_init_x=False, 
                 combiner_type='linear', combiner_layers_dim=[64], combiner_alpha_init=0.1,
                 combiner_alpha=0.5, combiner_learnable_alpha=True, combiner_output_dim=64, 
                 encoder_layers_dim=[128, 64], latent_transition_layers_dim=[128]):
        self.pz_f_rnn_hidden_size = pz_f_rnn_hidden_size
        self.pz_f_rnn_num_layers = pz_f_rnn_num_layers
        self.q_f_rnn_hidden_size = q_f_rnn_hidden_size
        self.q_f_rnn_num_layers = q_f_rnn_num_layers
        self.learnable_init_x = learnable_init_x
        self.latent_transition_layers_dim = latent_transition_layers_dim
        
        super().__init__(x_dim, z_dim, activation, q_b_rnn_hidden_size,
                         q_b_rnn_num_layers, decoder_layers_dim, dropout_p,
                         device, learnable_init_z, combiner_type,
                         combiner_layers_dim, combiner_alpha_init,
                         combiner_alpha, combiner_learnable_alpha,
                         combiner_output_dim, encoder_layers_dim)
        
        self.build_specific_components()

    def build_specific_components(self):
        """
        Initializes specific components for the FAR_DKF model.
        """
        self.combiner = DKF_CombinerFactory(
            z_dim=self.z_dim,
            rnn_dims=[self.q_b_rnn_hidden_size, self.q_f_rnn_hidden_size],
            output_dim=self.combiner_output_dim,
            activation=self.activation,
            combiner_type=self.combiner_type,
            alpha_init=self.combiner_alpha_init,
            alpha=self.combiner_alpha,
            learnable_alpha=self.combiner_learnable_alpha,
            layers_dim=self.combiner_layers_dim
        )
        
        self.pz_f_rnn = DKF_Forward_RNN(
            input_size=self.x_dim,
            hidden_size=self.pz_f_rnn_hidden_size,
            num_layers=self.pz_f_rnn_num_layers
        )
        
        self.q_f_rnn = DKF_Forward_RNN(
            input_size=self.x_dim,
            hidden_size=self.q_f_rnn_hidden_size,
            num_layers=self.q_f_rnn_num_layers
        )
        
        self.p_z = AR_DKF_LatentStateTransition(
            z_dim=self.z_dim,
            h_x_dim=self.pz_f_rnn_hidden_size,
            layers_dim=self.latent_transition_layers_dim
        )

        # Define X_0 as learnable parameters or fixed tensors
        if self.learnable_init_x:
            self.X_0 = nn.Parameter(torch.zeros(1, 1, self.x_dim))
        else:
            self.register_buffer('X_0', torch.zeros(1, 1, self.x_dim))

    def forward(self, x):
        """
        Performs a forward pass through the FAR_DKF model.
        """
        _, batch_size, _ = x.shape

        # Use the learnable or fixed parameters Z_0 and X_0
        z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
        x_0 = self.X_0.expand(1, batch_size, self.x_dim).to(self.device)
        
        # Pass the sequence [x_0, x_1, ..., x_{T-1}] to the Forward RNN
        x_lagged = torch.cat((x_0, x[:-1, :, :]), 0)
        h_x_lagged = self.pz_f_rnn(x_lagged)
        
        # Get the sequence [x_1, ..., x_T] transformed by the backward RNN
        q_b_rnn_out = self.q_b_rnn(x)
        # Get the sequence [x_1, ..., x_T] transformed by the forward RNN
        q_f_rnn_out = self.q_f_rnn(x)

        seq_len, batch_size, _ = q_b_rnn_out.shape

        # Output variables
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)

        sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0

        # Work throught the whole sequence
        for t in range(0, seq_len):
            
            # Combine z_{t-1} with output of the backward RNN and
            # output of the forward RNN for step t
            combined = self.combiner(sampled_z_t,
                                     [q_b_rnn_out[t,:,:], q_f_rnn_out[t,:,:]])
            
            # Pass through the Encoder
            mu, logvar = self.encoder(combined)
            
            self.z_mean[t,:,:] = mu
            self.z_logvar[t,:,:] = logvar
            sampled_z_t = self.sampler(mu, logvar)
            self.sampled_z[t,:,:] = sampled_z_t
        
        # Get the mean and logvar of the state transition probability
        # p(z_t|z_{t-1}, x_{0:t-1})
        z_lagged = torch.cat([z_0, self.sampled_z[:-1, :,:]], 0)
        self.z_transition_mean, self.z_transition_logvar = self.p_z(z_lagged,
                                                                    h_x_lagged)

        # Reconstruct x from z
        self.x_hat, self.x_hat_logvar = self.decoder(self.sampled_z)

        return (x, self.x_hat, self.x_hat_logvar, self.z_mean, self.z_logvar,
                self.z_transition_mean, self.z_transition_logvar)
    
    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.
        """
        with torch.no_grad():
            sampler = DKF_Sampler(self.device)
            seq_len, batch_size, x_dim = x.shape
            
            # Use the learnable or fixed parameters Z_0
            z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
 
            # Pass x throught the backward RNN
            q_b_rnn_out = self.q_b_rnn(x)
            
            # Pass x throught the forward RNN for the encoder
            q_f_rnn_out = self.q_f_rnn(x)
            
            #-----------------------------------------------------------
            # Output variables
            self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
            
            sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0
            
            # Work throught the whole sequence
            for t in range(0, seq_len):
                
                # Combine z_{t-1} with output of the backward RNN and with
                # the output of the forward RNN for step t
                combined = self.combiner(sampled_z_t,
                                         [q_b_rnn_out[t,:,:], q_f_rnn_out[t,:,:]])
                
                # Pass through the Encoder
                mu, logvar = self.encoder(combined)
                sampled_z_t = self.sampler(mu, logvar)
                self.sampled_z[t,:,:] = sampled_z_t

            # Reconstruct x from z
            x_hat, _ = self.decoder(self.sampled_z)
            
            #------------------------------------------------------------
            
            # Pass x throught the forward RNN  p(z_t|z_{t-1}, x_{0:t-1})
            h_x_lagged = self.pz_f_rnn(x)
            
            # Start with the last inferred latent state
            z_pred = self.sampled_z[-1:, :, :]
 
            predictions = torch.zeros(num_steps, batch_size, x_dim).to(self.device)
            for s in range(num_steps):
                # Get the parameters (mean and variance) of the transition
                # distribution p(z_t|z_{t-1}, x_{0:t-1}) (normal distribution)
                z_pred_mean, z_pred_logvar = self.p_z(z_pred,
                                                      h_x_lagged[-1:, :, :])

                # sample from p(z_t|z_{t-1}, x_{0:t-1}) distribution
                z_pred = sampler(z_pred_mean, z_pred_logvar)
                
                x_pred, _ = self.decoder(z_pred)
                
                x = torch.cat((x[1:,:,:], x_pred), dim=0)
                h_x_lagged = self.pz_f_rnn(x)
                
                predictions[s,:,:] = x_pred

            # Append predictions to the reconstructed x
            extended_x = torch.cat([x_hat, predictions], dim=0)
        
        return extended_x
    
#------------------------------------------------------------------------------
    
class BAR_DKF(DKF_Base):
    def __init__(self, x_dim, z_dim=8, activation='tanh', pz_f_rnn_hidden_size=64, 
                  pz_f_rnn_num_layers=1, q_b_rnn_hidden_size=64, q_b_rnn_num_layers=1, 
                  decoder_layers_dim=[64, 64], dropout_p=0, device='cpu',
                  learnable_init_z=True, learnable_init_x=False, combiner_type='linear',
                  combiner_layers_dim=[64], combiner_alpha_init=0.1,
                  combiner_alpha=0.5, combiner_learnable_alpha=True,
                  combiner_output_dim=64, encoder_layers_dim=[128, 64],
                  latent_transition_layers_dim=[128]):
        self.pz_f_rnn_hidden_size = pz_f_rnn_hidden_size
        self.pz_f_rnn_num_layers = pz_f_rnn_num_layers
        self.learnable_init_x = learnable_init_x
        self.latent_transition_layers_dim = latent_transition_layers_dim
        
        super().__init__(x_dim, z_dim, activation, q_b_rnn_hidden_size,
                          q_b_rnn_num_layers, decoder_layers_dim, dropout_p,
                          device, learnable_init_z, combiner_type,
                          combiner_layers_dim, combiner_alpha_init,
                          combiner_alpha, combiner_learnable_alpha,
                          combiner_output_dim, encoder_layers_dim)
        
        self.build_specific_components()

    def build_specific_components(self):
        """
        Initializes specific components for the BAR_DKF model.
        """
        # Replace the Backward RNN by a Bidirectional RNN
        self.q_b_rnn = DKF_Bidirectional_RNN(
            input_size=self.q_b_rnn_input_size,
            hidden_size=self.q_b_rnn_hidden_size,
            num_layers=self.q_b_rnn_num_layers,
            combine_type="weighted_sum",
            learnable_alpha=True
        )
        
        self.combiner = DKF_CombinerFactory(
            z_dim=self.z_dim,
            rnn_dims=self.q_b_rnn.output_size,
            output_dim=self.combiner_output_dim,
            activation=self.activation,
            combiner_type=self.combiner_type,
            alpha_init=self.combiner_alpha_init,
            alpha=self.combiner_alpha,
            learnable_alpha=self.combiner_learnable_alpha,
            layers_dim=self.combiner_layers_dim
        )
        
        self.pz_f_rnn = DKF_Forward_RNN(
            input_size=self.x_dim,
            hidden_size=self.pz_f_rnn_hidden_size,
            num_layers=self.pz_f_rnn_num_layers
        )
        
        self.p_z = AR_DKF_LatentStateTransition(
            z_dim=self.z_dim,
            h_x_dim=self.pz_f_rnn_hidden_size,
            layers_dim=self.latent_transition_layers_dim
        )

        # Define X_0 as learnable parameters or fixed tensors
        if self.learnable_init_x:
            self.X_0 = nn.Parameter(torch.zeros(1, 1, self.x_dim))
        else:
            self.register_buffer('X_0', torch.zeros(1, 1, self.x_dim))

    def forward(self, x):
        """
        Performs a forward pass through the BAR_DKF model.
        """
        _, batch_size, _ = x.shape

        # Use the learnable or fixed parameters Z_0 and X_0
        z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
        x_0 = self.X_0.expand(1, batch_size, self.x_dim).to(self.device)
        
        # Pass the sequence [x_0, x_1, ..., x_{T-1}] to the Forward RNN
        x_lagged = torch.cat((x_0, x[:-1, :, :]), 0)
        h_x_lagged = self.pz_f_rnn(x_lagged)
        
        # Get the sequence [x_1, ..., x_T] transformed by the bidirectional RNN
        q_b_rnn_out = self.q_b_rnn(x)

        seq_len, batch_size, _ = q_b_rnn_out.shape

        # Output variables
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)

        sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0

        # Work throught the whole sequence
        for t in range(0, seq_len):
            
            # Combine z_{t-1} with output of the backward RNN for step t
            combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
            
            # Pass through the Encoder
            mu, logvar = self.encoder(combined)
            
            self.z_mean[t,:,:] = mu
            self.z_logvar[t,:,:] = logvar
            sampled_z_t = self.sampler(mu, logvar)
            self.sampled_z[t,:,:] = sampled_z_t
        
        # Get the mean and logvar of the state transition probability
        # p(z_t|z_{t-1}, x_{0:t-1})
        z_lagged = torch.cat([z_0, self.sampled_z[:-1, :,:]], 0)
        self.z_transition_mean, self.z_transition_logvar = self.p_z(z_lagged,
                                                                    h_x_lagged)

        # Reconstruct x from z
        self.x_hat, self.x_hat_logvar = self.decoder(self.sampled_z)

        return (x, self.x_hat, self.x_hat_logvar, self.z_mean, self.z_logvar,
                self.z_transition_mean, self.z_transition_logvar)
    
    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.
        """
        with torch.no_grad():
            sampler = DKF_Sampler(self.device)
            seq_len, batch_size, x_dim = x.shape
            
            # Use the learnable or fixed parameters Z_0
            z_0 = self.Z_0.expand(1, batch_size, self.z_dim).to(self.device)
 
            q_b_rnn_out = self.q_b_rnn(x)
            
            #-----------------------------------------------------------
            # Output variables
            self.sampled_z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
            
            sampled_z_t = z_0.squeeze(0)  # Remove the extra dimension from z_0
            
            # Work throught the whole sequence
            for t in range(0, seq_len):
                
                # Combine z_{t-1} with output of the bidirectional RNN for step t
                combined = self.combiner(sampled_z_t, [q_b_rnn_out[t,:,:]])
                
                # Pass through the Encoder
                mu, logvar = self.encoder(combined)
                sampled_z_t = self.sampler(mu, logvar)
                self.sampled_z[t,:,:] = sampled_z_t

            # Reconstruct x from z
            x_hat, _ = self.decoder(self.sampled_z)
            
            #------------------------------------------------------------
            
            # Pass x throught the forward RNN
            h_x_lagged = self.pz_f_rnn(x)
            
            # Start with the last inferred latent state
            z_pred = self.sampled_z[-1:, :, :]
 
            predictions = torch.zeros(num_steps, batch_size, x_dim).to(self.device)
            for s in range(num_steps):
                # Get the parameters (mean and variance) of the transition
                # distribution p(z_t|z_{t-1}) (normal distribution)
                z_pred_mean, z_pred_logvar = self.p_z(z_pred,
                                                      h_x_lagged[-1:, :, :])

                # sample from p(z_t|z_{t-1}, x_{0:t-1}) distribution
                z_pred = sampler(z_pred_mean, z_pred_logvar)
                
                x_pred, _ = self.decoder(z_pred)
                
                x = torch.cat((x[1:,:,:], x_pred), dim=0)
                h_x_lagged = self.pz_f_rnn(x)
                
                predictions[s,:,:] = x_pred

            # Append predictions to the reconstructed x
            extended_x = torch.cat([x_hat, predictions], dim=0)
        
        return extended_x
