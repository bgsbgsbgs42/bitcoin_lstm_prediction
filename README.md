# Bitcoin LSTM Price Prediction

This project implements a Long Short-Term Memory (LSTM) neural network to predict Bitcoin price movements based on historical data. The model uses time series data with configurable window sizes to forecast future price trends.

## Features

- Time series data preprocessing with normalization
- LSTM neural network model implementation
- Configurable sliding window approach for predictions
- Visual representation of prediction results
- Support for both single and multiple sequence predictions

## Requirements

- Python 3.9+
- TensorFlow 2.15.0+
- NumPy, Pandas, Matplotlib
- h5py for data storage

## Project Structure

```
bitcoin-lstm-prediction/
├── bitcoin_lstm_prediction.py    # Main script with ETL and model implementation
├── configs.json                  # Configuration parameters
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/                         # Directory for processed data
│   ├── clean_data.h5             # Preprocessed training data
│   └── predictions.h5            # Model predictions output
└── models/                       # Directory for saved models
    └── bitcoin_lstm_model.h5     # Trained model weights and architecture
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bitcoin-lstm-prediction.git
cd bitcoin-lstm-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create the necessary directories:
```bash
mkdir -p data models
```

## Dataset

The model expects a CSV file with Bitcoin historical price data. At minimum, the dataset should include:
- Open prices
- High prices
- Low prices
- Close prices
- (Optional) Volume and other technical indicators

You can obtain historical Bitcoin data from sources like:
- Yahoo Finance
- CoinMarketCap
- Binance API
- CoinGecko API

Place your data file in the project root directory and update the `configs.json` file with the correct filename.

## Configuration

The `configs.json` file contains all configurable parameters:

```json
{
  "data": {
    "filename": "bitcoin_historical_data.csv",  // Input data filename
    "filename_clean": "data/clean_data.h5",     // Processed data output
    "batch_size": 100,                         // Training batch size
    "x_window_size": 60,                       // Input window size (days)
    "y_window_size": 1,                        // Prediction window size (days)
    "y_predict_column": "Close",               // Target column to predict
    "filter_columns": ["Open", "High", "Low", "Close"],  // Columns to use
    "train_test_split": 0.8                    // Train/test split ratio
  },
  "model": {
    "filename_model": "models/bitcoin_lstm_model.h5",  // Model save path
    "filename_predictions": "data/predictions.h5",     // Predictions save path
    "epochs": 25                               // Training epochs
  }
}
```

## Usage

1. Run the main script:
```bash
python bitcoin_lstm_prediction.py
```

2. The script will:
   - Load and preprocess the data
   - Train the LSTM model
   - Generate predictions
   - Plot prediction results
   - Save the model and predictions

## Visualization

The script generates two types of visualizations:

1. **Single Prediction Plot**: Shows the model's prediction against the actual price data in a continuous time series.

2. **Multiple Sequence Predictions**: Displays multiple prediction sequences overlaid on the actual data, useful for analyzing the model's performance over different starting points.

## Model Architecture

The LSTM architecture consists of:
- Input layer matching the dimensionality of the input features
- Two LSTM layers with 150 units each and dropout (0.2) for regularization
- Dense output layer for price prediction

## Customization

You can customize the model by:

1. Modifying the network architecture in the `build_network` function
2. Adjusting hyperparameters in the `configs.json` file
3. Adding additional features to the input data
4. Implementing different preprocessing techniques in the ETL class

## Performance Optimization

For better performance:
- Use GPU acceleration for faster training
- Experiment with different window sizes
- Add more technical indicators as input features
- Implement feature engineering for better signal extraction
- Try different LSTM architectures (stacked, bidirectional, etc.)

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses TensorFlow and Keras for the implementation of LSTM networks.

## Disclaimer

This tool is for educational and research purposes only. Cryptocurrency investments are volatile and risky. Do not use these predictions as financial advice for trading decisions.
