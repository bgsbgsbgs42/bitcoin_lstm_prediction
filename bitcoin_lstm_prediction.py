import time
import threading
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration loading
with open('configs.json', 'r') as config_file:
    configs = json.load(config_file)

tstart = time.time()

# Helper Functions for plotting and predicting
def plot_results(predicted_data, true_data):
    """Plot the predicted vs true data"""
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def predict_sequences_multiple(model, data, window_size, prediction_len):
    """Predict sequence of prediction_len steps before shifting prediction run forward by prediction_len steps"""
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:], verbose=0)[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results_multiple(predicted_data, true_data, prediction_len):
    """Plot multiple prediction sequences against true data"""
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label=f'Prediction {i+1}')
    plt.legend()
    plt.show()

# Data ETL Class
class ETL:
    """Extract, Transform, Load class for data preprocessing"""
    
    def __init__(self):
        self.normalise = True
    
    def create_clean_datafile(self, filename_in, filename_out, batch_size, x_window_size, y_window_size, y_col, filter_cols, normalise=True):
        """Create a clean h5 data file from a source data file"""
        print("> Creating x & y data files...")
        
        # Read source data
        df = pd.read_csv(filename_in, index_col=0)
        
        # Filter columns if needed
        if filter_cols:
            df = df[filter_cols]
        
        # Create clean dataset
        data = self._create_clean_dataset(df, x_window_size, y_window_size, y_col)
        
        # Save to h5 file
        with h5py.File(filename_out, 'w') as hf:
            hf.create_dataset('x', data=data[0])
            hf.create_dataset('y', data=data[1])
        
        print(f"> Clean datasets created in file `{filename_out}`")
    
    def _create_clean_dataset(self, data, x_window_size, y_window_size, y_col):
        """Create clean dataset from dataframe"""
        data_x = []
        data_y = []
        
        # Normalize data
        if self.normalise:
            data = (data - data.min()) / (data.max() - data.min())
        
        # Get column index for y prediction
        if isinstance(y_col, int):
            y_col_idx = y_col
        else:
            y_col_idx = list(data.columns).index(y_col)
        
        # Create sliding window sequences
        for i in range(len(data) - x_window_size - y_window_size):
            x_window = data.iloc[i:(i + x_window_size)].values
            y_window = data.iloc[(i + x_window_size):(i + x_window_size + y_window_size)][y_col].values
            data_x.append(x_window)
            data_y.append(y_window)
        
        return np.array(data_x), np.array(data_y)
    
    def generate_clean_data(self, filename, batch_size, start_index=0):
        """Generator to yield batches of data from h5 file"""
        while True:
            with h5py.File(filename, 'r') as hf:
                num_batches = len(hf['x']) // batch_size
                
                if start_index >= len(hf['x']):
                    start_index = 0
                
                end_index = min(start_index + batch_size, len(hf['x']))
                
                x_data = hf['x'][start_index:end_index]
                y_data = hf['y'][start_index:end_index]
                
                start_index += batch_size
                if start_index >= len(hf['x']):
                    start_index = 0
                
                yield (x_data, y_data)

# LSTM Model Builder
def build_network(layers):
    """Build a LSTM model with the given layer sizes"""
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(layers[1], 
                   input_shape=(None, layers[0]),
                   return_sequences=(len(layers) > 3)))
    model.add(Dropout(0.2))
    
    # Additional LSTM layers if needed
    if len(layers) > 3:
        for layer_size in layers[2:-1]:
            model.add(LSTM(layer_size, return_sequences=False))
            model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(layers[-1], activation='linear'))
    
    start = time.time()
    model.compile(loss='mse', optimizer='adam')
    print(f"> Compilation Time: {time.time() - start}")
    
    return model

# Main Execution
if __name__ == "__main__":
    # Create ETL instance
    dl = ETL()
    
    # Create clean data file
    dl.create_clean_datafile(
        filename_in=configs['data']['filename'],
        filename_out=configs['data']['filename_clean'],
        batch_size=configs['data']['batch_size'],
        x_window_size=configs['data']['x_window_size'],
        y_window_size=configs['data']['y_window_size'],
        y_col=configs['data']['y_predict_column'],
        filter_cols=configs['data']['filter_columns'],
        normalise=True
    )
    
    print(f"> Generating clean data from: {configs['data']['filename_clean']} with batch_size: {configs['data']['batch_size']}")
    
    # Create data generators
    data_gen_train = dl.generate_clean_data(
        configs['data']['filename_clean'],
        batch_size=configs['data']['batch_size']
    )
    
    # Get dimensions from data
    with h5py.File(configs['data']['filename_clean'], 'r') as hf:
        nrows = hf['x'].shape[0]
        ncols = hf['x'].shape[2]
    
    # Split into train/test
    ntrain = int(configs['data']['train_test_split'] * nrows)
    steps_per_epoch = int((ntrain / configs['model']['epochs']) / configs['data']['batch_size'])
    print(f"> Clean data has {nrows} data rows. Training on {ntrain} rows with {steps_per_epoch} steps-per-epoch")
    
    # Build model
    model = build_network([ncols, 150, 150, 1])
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train model
    model.fit(
        next(data_gen_train)[0][:ntrain],  # Use next() to get batch data instead of generator
        next(data_gen_train)[1][:ntrain],
        batch_size=configs['data']['batch_size'],
        epochs=configs['model']['epochs'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model.save(configs['model']['filename_model'])
    print(f"> Model Trained! Weights saved in {configs['model']['filename_model']}")
    
    # Test model
    data_gen_test = dl.generate_clean_data(
        configs['data']['filename_clean'],
        batch_size=configs['data']['batch_size'],
        start_index=ntrain
    )
    
    ntest = nrows - ntrain
    steps_test = int(ntest / configs['data']['batch_size'])
    print(f"> Testing model on {ntest} data rows with {steps_test} steps")
    
    # Get test data
    test_x, test_y = next(data_gen_test)
    test_x = test_x[:ntest]
    test_y = test_y[:ntest]
    
    # Make predictions
    predictions = model.predict(test_x, verbose=1)
    
    # Save predictions
    with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
        hf.create_dataset('predictions', data=predictions)
        hf.create_dataset('true_values', data=test_y)
    
    # Plot results
    plot_results(predictions[:800], test_y[:800])
    
    # Make multiple sequence predictions
    window_size = 50  # number of steps to predict into the future
    predictions_multiple = predict_sequences_multiple(
        model,
        test_x,
        test_x[0].shape[0],
        window_size
    )
    
    plot_results_multiple(predictions_multiple, test_y, window_size)
    
    print(f"> Total execution time: {time.time() - tstart} seconds")
