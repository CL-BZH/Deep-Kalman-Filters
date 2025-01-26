import numpy as np
from sklearn.linear_model import LinearRegression

import random

import seaborn as sns
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples, seq_len, input_dim, noise_var):
    assert num_samples > seq_len
    
    data = np.zeros((num_samples, seq_len, input_dim))
    noise = np.zeros((num_samples, seq_len, input_dim))
    noise_std = np.sqrt(noise_var)
    gaussian_noise = np.random.normal(0, noise_std, num_samples)

    series = 0.5 * np.sin(0.1*np.arange(num_samples))
    
    for t in range(len(series) - seq_len):
        seq = series[t:t+seq_len]

        data[t,:,:] = seq.reshape(-1,1)
        noise[t,:,:] = gaussian_noise[t:t+seq_len].reshape(-1,1)

    data = data[:t,:,:]
    noise = noise[:t,:,:]
    return data, noise
    

class AutoRegressiveModel:
    """
    An Auto-Regressive (AR) model that predicts future values based on a fixed
    number of past observations.
    Supports batch processing for efficient prediction.
    """
    def __init__(self, p):
        """
        Initialize the AR model.

        Parameters:
        -----------
        p : int
            The order of the AR model
            (number of past observations to use for prediction).
        """
        self.p = p
        self.model = LinearRegression()

    def fit(self, data):
        """
        Fit the AR model to the training data.

        Parameters:
        -----------
        data : np.ndarray
            The training data with shape (num_samples, sequence_length).
        """
        X, y = self._create_ar_dataset(data)
        self.model.fit(X, y)

    def predict(self, data, prediction_length):
        """
        Predict future values using the AR model.

        Parameters:
        -----------
        data : np.ndarray
            The input data with shape (num_samples, sequence_length).
        prediction_length : int
            The number of future time steps to predict.

        Returns:
        --------
        np.ndarray
            The predicted values with shape (num_samples, prediction_length).
        """
        num_samples = data.shape[0]
        predictions = []

        # Initialize the current sequences for all samples
        current_sequences = data[:, -self.p:].reshape(num_samples, self.p)

        for _ in range(prediction_length):
            # Predict the next value for all samples in the batch
            next_values = self.model.predict(current_sequences)
            predictions.append(next_values)

            # Update the current sequences by appending the predicted values
            current_sequences = np.hstack((current_sequences[:, 1:],
                                           next_values.reshape(-1, 1)))

        return np.array(predictions).T

    def _create_ar_dataset(self, data):
        """
        Create the dataset for training the AR model.

        Parameters:
        -----------
        data : np.ndarray
            The training data with shape (num_samples, sequence_length).

        Returns:
        --------
        X : np.ndarray
            The input features for training.
        y : np.ndarray
            The target values for training.
        """
        num_samples, seq_length, _ = data.shape
        XX, yy = [], []
        
        for s in range(num_samples):
            X, y = [], []
            for i in range(self.p, seq_length):
                X.append(data[s, i-self.p:i, 0])
                y.append(data[s, i, 0])
            X, y = np.vstack(X), np.vstack(y)
            XX.append(X)
            yy.append(y)
        
        XX, yy = np.vstack(XX), np.vstack(yy)
        print(f"XX shape: {XX.shape}. yy shape: {yy.shape}")
        return XX, yy
 
    
# Example usage of the AR model
if __name__ == "__main__":
    # Generate synthetic data
    input_size = 1100
    training_seq_len = 50
    eval_seq_len = 20
    x_dim = 1
    noise_var = 0.004
    x, e = generate_synthetic_data(input_size, training_seq_len + eval_seq_len,
                                   x_dim, noise_var)

    # Prepare data
    np.random.shuffle(x)
    l_train = input_size // 2
    l_val = l_train // 2
    l = l_train + l_val
    prediction_batch_size = 200

    # Prepare training and evaluation data
    training_data = x[:l_train, :training_seq_len]
    eval_data = x[l:l+prediction_batch_size, :training_seq_len + eval_seq_len]

    # Generate noise for evaluation sequences
    noise_for_eval = np.random.normal(0, np.sqrt(noise_var),
                                      size=(prediction_batch_size,
                                            training_seq_len, 1))
    # Add noise to eval_data
    noisy_eval_data = eval_data[:,:training_seq_len,:] + noise_for_eval
    
    # Train the AR model
    # Use p past observations for prediction
    p = training_seq_len - 1
    ar_model = AutoRegressiveModel(p)
    ar_model.fit(training_data)

    # Evaluate the AR model
    ar_predictions = ar_model.predict(noisy_eval_data[:, :training_seq_len],
                                      eval_seq_len)


    indexes = random.sample(range(1, prediction_batch_size), 3)
    
    for i in indexes:
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # Draw separation with prediction part
        plt.axvline(x=training_seq_len-1, linestyle='-.', color='black')
        
        # Plot ground truth
        plt.plot(eval_data[i, :training_seq_len+eval_seq_len, 0],
                 label='Ground Truth', linestyle=':', color='blue')
    
        # Plot noisy data
        plt.plot(noisy_eval_data[i, :training_seq_len, 0], label='Ground Truth',
                 linestyle='-', marker='.', color='blue')
        
        # Plot predictions
        plt.plot(np.arange(training_seq_len, training_seq_len + eval_seq_len),
                 ar_predictions[0,i,:], alpha=0.6, color='lime',
                 linestyle='-', marker='.', label='Predictions')
    
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('AR Evaluation')
        plt.legend()
        plt.grid(True)
        plt.show()  
        
    print(f"AR pred shape: {ar_predictions.shape}")
    # Compare predictions with ground truth
    mse_pred = ((eval_data[:, training_seq_len:, 0] - ar_predictions)**2).mean()
    print(f"AR Model MSE Prediction: {mse_pred}")