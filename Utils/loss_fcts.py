#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss Functions for Deep Kalman Filter (DKF) Models

This script defines loss functions used for training Deep Kalman Filter (DKF) models.
It includes both Mean Squared Error (MSE) and Weighted MSE loss functions, as well as
a Kullback-Leibler (KL) divergence term for variational inference.

Author: Christophe Louargant
Date: December 20, 2024
@email: 
@version: 1.0

License: MIT (see LICENSE file for details)
"""

import torch

#--------------------------------------------------------------------------
def loss_function(x, x_hat, x_hat_logvar, z_mean, z_logvar,
                  z_transition_mean, z_transition_logvar, beta,
                  loss_type='weighted_mse'):
    """
    Compute the total loss for a variational autoencoder (VAE) with a weighted 
    reconstruction loss and a Kullback-Leibler (KL) divergence term.

    Parameters:
    -----------
    x : torch.Tensor
        Ground truth data with shape (seq_len, batch_size, x_dim).
    x_hat : torch.Tensor
        Reconstructed data from the VAE with shape
        (seq_len, batch_size, x_dim).
    x_hat_logvar : torch.Tensor
        Log variance of the reconstructed data with shape
        (seq_len, batch_size, x_dim).
    z_mean : torch.Tensor
        Mean of the latent variable distribution with shape 
        (seq_len, batch_size, x_dim).
    z_logvar : torch.Tensor
        Log variance of the latent variable distribution with shape 
        (seq_len, batch_size, x_dim).
    z_transition_mean : torch.Tensor
        Mean of the transition distribution in the latent space with shape 
        (seq_len, batch_size, x_dim).
    z_transition_logvar : torch.Tensor
        Log variance of the transition distribution in the latent space with 
        shape (seq_len, batch_size, x_dim).
    beta : float
        Weighting factor for the KL divergence term.
    loss_type : str
        Type of reconstruction loss to use. Options:
        - 'mse': Mean Squared Error (MSE) loss.
        - 'weighted_mse': Weighted Mean Squared Error (MSE) loss.

    Returns:
    --------
    total_loss : torch.Tensor
        The total loss, which is the sum of the reconstruction loss and the 
        KL divergence loss.

    Notes:
    ------
    - The reconstruction loss can be either MSE or Weighted MSE.
    - The KL divergence loss measures the difference between the latent
      variable distribution and the transition distribution in the latent space.
    - Both losses are normalized by the sequence length (`seq_len`) and
      averaged over the batch.
    - The total loss is a combination of the reconstruction loss and the 
      KL divergence loss, weighted by the `beta` parameter.
    """
    
    seq_len, batch_size, x_dim = x.shape
    
    # Define the reconstruction loss based on the loss_type
    if loss_type == 'weighted_mse':
        def weighted_mse_loss(x, x_hat, x_hat_logvar):
            """
            Compute the weighted mean squared error (MSE) loss for reconstruction.

            Parameters:
            -----------
            x : torch.Tensor
                Ground truth data with shape (seq_len, batch_size, x_dim).
            x_hat : torch.Tensor
                Reconstructed data with shape (seq_len, batch_size, x_dim).
            x_hat_logvar : torch.Tensor
                Log variance of the reconstructed data with shape
                (seq_len, batch_size, x_dim).

            Returns:
            --------
            loss : torch.Tensor
                The weighted MSE loss normalized by the sequence length and
                averaged over the batch.
            """
            var = x_hat_logvar.exp()
            loss = torch.div((x - x_hat)**2, var)
            
            loss += x_hat_logvar
            loss = loss.sum(dim=2)  # Sum over the x_dim
            loss = loss.sum(dim=0)  # Sum over the sequence length
            loss = loss.mean()  # Mean over the batch
            return loss / seq_len
        
        reconstruction_loss = weighted_mse_loss(x, x_hat, x_hat_logvar)
    
    elif loss_type == 'mse':
        def mse_loss(x, x_hat):
            """
            Compute the mean squared error (MSE) loss for reconstruction.

            Parameters:
            -----------
            x : torch.Tensor
                Ground truth data with shape (seq_len, batch_size, x_dim).
            x_hat : torch.Tensor
                Reconstructed data with shape (seq_len, batch_size, x_dim).

            Returns:
            --------
            loss : torch.Tensor
                The MSE loss normalized by the sequence length and
                averaged over the batch.
            """
            loss = (x - x_hat)**2
            loss = loss.sum(dim=2)  # Sum over the x_dim
            loss = loss.sum(dim=0)  # Sum over the sequence length
            loss = loss.mean()  # Mean over the batch
            return loss / seq_len
        
        reconstruction_loss = mse_loss(x, x_hat)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}."
                         "Choose 'mse' or 'weighted_mse'.")
    
    # Compute the KL divergence loss
    kl_loss = (z_transition_logvar - z_logvar +
               torch.div((z_logvar.exp() + 
                         (z_transition_mean - z_mean).pow(2)),
                         z_transition_logvar.exp()))
    
    kl_loss = kl_loss.sum(dim=2)  # Sum over the x_dim
    kl_loss = kl_loss.sum(dim=0)  # Sum over the sequence length
    kl_loss = kl_loss.mean()  # Mean over the batch
    kl_loss = kl_loss / seq_len
                
    # Combine the reconstruction loss and the KL divergence loss
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss