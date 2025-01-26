
---

# Deep Kalman Filter (DKF) Models

`dkf_models.py` defines a set of classes for implementing a Deep Kalman Filter (DKF) model in PyTorch. The DKF is a probabilistic model that combines ideas from both Kalman filters and deep learning to model time series data. The script is organized into several modules, each responsible for different parts of the DKF model. Below is a detailed breakdown of the key components and their functionalities:

---

## 1. **DKF_Forward_RNN**
- **Purpose**: Processes input sequences using a forward RNN (LSTM).
- **Key Parameters**:
  - `input_size`: Input size for the RNN.
  - `hidden_size`: Hidden state size of the RNN.
  - `num_layers`: Number of RNN layers.
- **Forward Pass**: Takes an input tensor of shape `(seq_len, batch_size, input_size)` and returns the RNN output of shape `(seq_len, batch_size, hidden_size)`.

---

## 2. **DKF_Backward_RNN**
- **Purpose**: Processes input sequences in reverse order using a backward RNN (LSTM).
- **Key Parameters**:
  - `input_size`: Input feature dimension.
  - `hidden_size`: Hidden state size of the RNN.
  - `num_layers`: Number of RNN layers.
- **Forward Pass**: Reverses the input sequence, processes it through the RNN, and then reverses the output back to the original order.

---

## 3. **DKF_Bidirectional_RNN**
- **Purpose**: Processes input sequences in both forward and backward order using a bidirectional RNN.
- **Key Parameters**:
  - `input_size`: Input feature dimension.
  - `hidden_size`: Hidden state size of the LSTM.
  - `num_layers`: Number of recurrent layers in the LSTM.
  - `combine_type`: Method to combine forward and backward outputs (e.g., "concat", "sum", "weighted_sum", "concat_linear").
  - `learnable_alpha`: Whether the alpha parameter in the weighted sum should be learnable.
- **Forward Pass**: Combines the forward and backward RNN outputs using the specified method.

---

## 4. **DKF_Sampler**
- **Purpose**: Samples from a normal distribution using the reparameterization trick.
- **Key Parameters**:
  - `device`: Device to use ('cpu' or 'cuda').
- **Forward Pass**: Takes mean and log variance tensors and returns a sampled tensor using the reparameterization trick.

---

## 5. **DKF_Combiner**
- **Purpose**: Abstract base class for combining the previous latent state `z_{t-1}` with RNN hidden states.
- **Key Parameters**:
  - `z_dim`: Dimension of the latent state.
  - `rnn_dims`: Dimensions of the RNN hidden states.
  - `output_dim`: Dimension of the output tensor.
  - `activation`: Activation function ('relu' or 'tanh').

---

## 6. **DKF_CombinerMLP**
- **Purpose**: Combines `z_{t-1}` with RNN hidden states using a Multi-Layer Perceptron (MLP).
- **Key Parameters**:
  - `layers_dim`: List of dimensions for the MLP hidden layers.
- **Forward Pass**: Concatenates `z_{t-1}` and RNN hidden states, then passes them through an MLP.

---

## 7. **DKF_CombinerLinear**
- **Purpose**: Combines `z_{t-1}` with RNN hidden states using a linear combination.
- **Key Parameters**:
  - `alpha_init`: Initial value(s) of alpha for the first forward pass.
  - `alpha`: Value(s) of alpha for subsequent forward passes.
  - `learnable_alpha`: Whether alpha is learnable.
- **Forward Pass**: Combines `z_{t-1}` and RNN hidden states using a weighted sum with alpha.

---

## 8. **DKF_CombinerFactory**
- **Purpose**: Factory class to create the appropriate combiner based on the type ('linear' or 'mlp').
- **Key Parameters**:
  - `combiner_type`: Type of combiner to use.
  - `layers_dim`: List of dimensions for the MLP hidden layers (if using 'mlp').
  - `alpha_init`, `alpha`, `learnable_alpha`: Parameters for the linear combiner.

---

## 9. **DKF_Encoder**
- **Purpose**: Encodes the combined input into latent space (mean and log variance).
- **Key Parameters**:
  - `input_dim`: Input feature dimension.
  - `z_dim`: Latent state dimension.
  - `encoder_layers_dim`: List of dimensions for the encoder MLP layers.
  - `activation`: Activation function.
  - `device`: Device to use.
- **Forward Pass**: Takes the combined input and returns the mean and log variance of the latent states.

---

## 10. **DKF_Decoder**
- **Purpose**: Decodes latent states back to the original data space.
- **Key Parameters**:
  - `dim_in`: Input latent state dimension.
  - `dim_out`: Output data dimension.
  - `layers_dim`: List of dimensions for the decoder MLP layers.
  - `activation`: Activation function.
  - `dropout_p`: Dropout probability.
- **Forward Pass**: Takes latent states and returns the mean and log variance of the reconstructed data.

---

## 11. **DKF_LatentStateTransition**
- **Purpose**: Computes the state transition probability `p(z_t | z_{t-1})`.
- **Key Parameters**:
  - `z_dim`: Latent state dimension.
  - `layers_dim`: List of dimensions for the MLP layers.
- **Forward Pass**: Takes the previous latent state and returns the mean and log variance of the transition probability.

---

## 12. **AR_DKF_LatentStateTransition**
- **Purpose**: Computes the state transition probability `p(z_t | z_{t-1}, x_{0:t-1})`.
- **Key Parameters**:
  - `z_dim`: Latent state dimension.
  - `h_x_dim`: Dimension of the observation transformed by the forward RNN.
  - `layers_dim`: List of dimensions for the MLP layers.
  - `use_z_t_minus_1`: Whether to use `z_{t-1}` in the computation.
- **Forward Pass**: Takes the previous latent state and the transformed observation, and returns the mean and log variance of the transition probability.

---

## 13. **DKF_Base**
- **Purpose**: Abstract base class for the DKF model.
- **Key Parameters**:
  - `x_dim`: Input data dimension.
  - `z_dim`: Latent state dimension.
  - `activation`: Activation function.
  - `q_b_rnn_hidden_size`: Hidden size of the backward RNN.
  - `q_b_rnn_num_layers`: Number of layers in the backward RNN.
  - `decoder_layers_dim`: List of dimensions for the decoder MLP layers.
  - `dropout_p`: Dropout probability.
  - `device`: Device to use.
  - `learnable_init_z`: Whether to use learnable initial latent state.
  - `combiner_type`: Type of combiner to use.
  - `combiner_layers_dim`: List of dimensions for the combiner MLP layers.
  - `combiner_alpha_init`, `combiner_alpha`, `combiner_learnable_alpha`: Parameters for the combiner.
  - `combiner_output_dim`: Output dimension of the combiner.
  - `encoder_layers_dim`: List of dimensions for the encoder MLP layers.
- **Methods**:
  - `build_shared_components`: Initializes shared components like backward RNN, encoder, decoder, and sampler.
  - `forward`: Performs a forward pass through the model.
  - `predict`: Predicts future steps based on the input sequence.
  - `save_model`, `load_model`: Save and load model parameters.

---

## 14. **DKF**
- **Purpose**: Implements the basic DKF model.
- **Key Parameters**:
  - `latent_transition_layers_dim`: List of dimensions for the latent state transition MLP layers.
- **Methods**:
  - `build_specific_components`: Initializes specific components like the combiner and latent state transition.

---

## 15. **AR_DKF**
- **Purpose**: Implements the Autoregressive DKF model.
- **Key Parameters**:
  - `pz_f_rnn_hidden_size`: Hidden size of the forward RNN.
  - `pz_f_rnn_num_layers`: Number of layers in the forward RNN.
  - `learnable_init_x`: Whether to use learnable initial observation.
  - `latent_transition_layers_dim`: List of dimensions for the latent state transition MLP layers.
- **Methods**:
  - `build_specific_components`: Initializes specific components like the forward RNN, combiner, and latent state transition.
  - `forward`: Performs a forward pass through the AR_DKF model.
  - `predict`: Predicts future steps based on the input sequence.

---

## 16. **FAR_DKF**
- **Purpose**: Implements the Forward Autoregressive DKF model.
- **Key Parameters**:
  - `pz_f_rnn_hidden_size`: Hidden size of the forward RNN.
  - `pz_f_rnn_num_layers`: Number of layers in the forward RNN.
  - `q_f_rnn_hidden_size`: Hidden size of the forward RNN for the encoder.
  - `q_f_rnn_num_layers`: Number of layers in the forward RNN for the encoder.
  - `learnable_init_x`: Whether to use learnable initial observation.
  - `latent_transition_layers_dim`: List of dimensions for the latent state transition MLP layers.
- **Methods**:
  - `build_specific_components`: Initializes specific components like the forward RNN, combiner, and latent state transition.
  - `forward`: Performs a forward pass through the FAR_DKF model.
  - `predict`: Predicts future steps based on the input sequence.

---

## 17. **BAR_DKF**
- **Purpose**: Implements the Bidirectional Autoregressive DKF model.
- **Key Parameters**:
  - `pz_f_rnn_hidden_size`: Hidden size of the forward RNN.
  - `pz_f_rnn_num_layers`: Number of layers in the forward RNN.
  - `learnable_init_x`: Whether to use learnable initial observation.
  - `latent_transition_layers_dim`: List of dimensions for the latent state transition MLP layers.
- **Methods**:
  - `build_specific_components`: Initializes specific components like the bidirectional RNN, combiner, and latent state transition.
  - `forward`: Performs a forward pass through the BAR_DKF model.
  - `predict`: Predicts future steps based on the input sequence.

---

## Summary
The script provides a modular implementation of the DKF model, allowing for different configurations such as linear or MLP combiners, autoregressive models, and fforward autoregressive models. Each component is designed to be flexible and can be customized based on the specific requirements of the application.

---

## Class Diagram

Below is a simplified class diagram that shows the relationships between the different classes in `dkf_models.py`:

```
DKF_Base (Abstract)
├── DKF
├── AR_DKF
├── FAR_DKF
└── BAR_DKF

DKF_Combiner (Abstract)
├── DKF_CombinerMLP
└── DKF_CombinerLinear

DKF_CombinerFactory

DKF_Forward_RNN
DKF_Backward_RNN
DKF_Bidirectional_RNN

DKF_Sampler

DKF_Encoder
DKF_Decoder

DKF_LatentStateTransition
AR_DKF_LatentStateTransition
```

---

## How the Classes Interact

1. **DKF_Base**: The base class for all DKF models. It initializes shared components like the backward RNN, encoder, decoder, and sampler.
2. **DKF**: Implements the basic DKF model.
3. **AR_DKF**: Adds a forward RNN to the DKF model for autoregressive modeling.
4. **FAR_DKF**: Extends AR_DKF by adding a forward RNN for the encoder.
5. **BAR_DKF**: Replaces the backward RNN with a bidirectional RNN.
6. **DKF_Combiner**: Abstract base class for combining latent states with RNN outputs.
7. **DKF_CombinerMLP** and **DKF_CombinerLinear**: Implement different methods for combining latent states with RNN outputs.
8. **DKF_Encoder** and **DKF_Decoder**: Encode and decode latent states to and from the original data space.
9. **DKF_LatentStateTransition** and **AR_DKF_LatentStateTransition**: Compute the mean and log-variance for the state transition probability.

---
