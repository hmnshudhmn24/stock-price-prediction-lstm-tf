# 📈 Stock Price Prediction using LSTM (TensorFlow)

This project demonstrates how to use Long Short-Term Memory (LSTM) networks to forecast stock prices using historical data. The model is built with TensorFlow/Keras and processes time series data for regression prediction.

## 🔍 Overview

- ✅ Fetch historical stock data (default: AAPL)
- ✅ Preprocess & normalize data
- ✅ Train LSTM model
- ✅ Predict & visualize results

## 📦 Dependencies

```bash
pip install yfinance tensorflow matplotlib scikit-learn pandas numpy
```

## 🚀 How to Run

1. Make sure you have Python 3 installed.
2. Install the dependencies listed above.
3. Run the script:
```bash
python stock_lstm.py
```

The script trains an LSTM model and plots predicted vs actual stock prices.

## 🧠 Model Architecture

- LSTM (50 units)
- Dense (1 output unit)

## 📊 Output

Displays a plot comparing real vs predicted stock prices.

## 📁 Files

- `stock_lstm.py`: Main Python script
- `README.md`: Documentation
