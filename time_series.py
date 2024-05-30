import pandas as pd
import pandas as pd1
import numpy as np
import math
import matplotlib.pyplot as plt  # Visualization
import matplotlib.dates as mdates  # Formatting dates
import seaborn as sns  # Visualization
from sklearn.preprocessing import MinMaxScaler
import torch  # Library for implementing Deep Neural Network
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

import yfinance as yf
from datetime import date, timedelta, datetime

import sys
import warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Instrument name')
    parser.add_argument('--mode', help='Local file or Y Finance Api', default='yf')
    parser.add_argument('--date', help='Last date to pick', default='today')
    args = parser.parse_args()

    mode = args.mode.upper()
    df_copy = 0

    if mode != 'YF':
        # for ease of running the script, only the instrument name
        # as the argument on command
        def convert_filename(filename):
            new_filename = '/Users/bayankaran/Workspace/MLE-Lyapunov/Instruments/Full/' + filename.upper() + '.csv'
            return new_filename

        instrument_name = convert_filename(args.name)
        print("Instrument: ", instrument_name)

        df1 = pd.read_csv(convert_filename(args.name))

        start_date = df1.iloc[0, 0]
        end_date = df1.iloc[-1, 0]
        df2 = pd.read_csv(convert_filename(args.name))
        df2['Date'] = pd.to_datetime(df2['Date'])
        # Filter data based on start_date and end_date (inclusive)
        filtered_df = df2[(df2['Date'] >= start_date) & (df2['Date'] <= end_date)]

        # Assuming specific columns for Yahoo Finance-like output (modify as needed)
        yahoo_like_df = filtered_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Set 'Date' as the index
        yahoo_like_df.set_index('Date', inplace=True)
        df = yahoo_like_df
        df_copy = df
    else:
        if args.date.upper() == 'TODAY':
            end_date = date.today().strftime("%Y-%m-%d")  # end date for our data retrieval will be current date
        else:
            end_date = args.date

        start_date = '1990-01-01'  # Beginning date for our historical data retrieval
        df = yf.download(args.name.upper(), start=start_date, end=end_date)  # Function used to fetch the data
        df_copy = df

    print(args.name.upper(), "Start Date: ", start_date, "End Date: ", end_date)
    last_date_for_printing = df_copy.index[-1].strftime('%Y-%m-%d')

    def data_plot(df):
        df_plot = df.copy()
        print(df_plot.head())
        ncols = 2
        nrows = int(round(df_plot.shape[1] / ncols, 0))

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               sharex=True, figsize=(14, 7))
        for i, ax in enumerate(fig.axes):
            sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
            ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.tight_layout()
        plt.show()

    data_plot(df)

    # Train-Test Split
    # Setting 80 percent data for training
    training_data_len = math.ceil(len(df) * .80)

    # Splitting the dataset
    train_data = df[:training_data_len].iloc[:, :1]
    test_data = df[training_data_len:].iloc[:, :1]
    print("train_data_shape: ", train_data.shape, "test_data_shape: ", test_data.shape)

    # Selecting Open Price values
    dataset_train = train_data.Open.values
    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1, 1))
    print(dataset_train.shape)

    # Selecting Open Price values
    dataset_test = test_data.Open.values
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1, 1))
    print(dataset_test.shape)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaling dataset
    scaled_train = scaler.fit_transform(dataset_train)

    print(scaled_train[:5])
    # Normalizing values between 0 and 1
    scaled_test = scaler.fit_transform(dataset_test)
    print(*scaled_test[:5])  # prints the first 5 rows of scaled_test

    # Create sequences and labels for training data
    sequence_length = 50  # Number of time steps to look back
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i + sequence_length])
        y_train.append(scaled_train[i + 1:i + sequence_length + 1])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    print(X_train.shape, y_train.shape)

    # Create sequences and labels for testing data
    sequence_length = 30  # Number of time steps to look back
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i + sequence_length])
        y_test.append(scaled_test[i + 1:i + sequence_length + 1])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    class LSTMModel(nn.Module):
        # input_size : number of features in input at each time step
        # hidden_size : Number of LSTM units
        # num_layers : number of LSTM layers
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMModel, self).__init__()  # initializes the parent class nn.Module
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):  # defines forward pass of the neural network
            out, _ = self.lstm(x)
            out = self.linear(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1

    # Define the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    batch_size = 16
    # Create DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for batch training
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 50
    # num_epochs = 40
    train_hist = []
    test_hist = []
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
    x = np.linspace(1, num_epochs, num_epochs)
    plt.plot(x, train_hist, scalex=True, label="Training loss")
    plt.plot(x, test_hist, label="Test loss")
    plt.legend()
    plt.show()

    # Define the number of future time steps to forecast
    num_forecast_steps = 30

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_test.squeeze().cpu().numpy()

    # Use the last 30 data points as the starting point
    historical_data = sequence_to_plot[-1]
    print(historical_data.shape)

    # Initialize a list to store the forecasted values
    forecasted_values = []

    # Use the trained model to forecast future values
    with torch.no_grad():
        for _ in range(num_forecast_steps * 2):
            # Prepare the historical_data tensor
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            # Use the model to predict the next value
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence by removing the oldest value and adding the predicted value
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

    # Generate future dates
    last_date = test_data.index[-1]

    # Generate the next 30 dates
    future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)

    # Concatenate the original index with the future dates
    combined_index = test_data.index.append(future_dates)

    # set the size of the plot
    plt.rcParams['figure.figsize'] = [14, 4]

    # Test data
    plt.plot(test_data.index[-100:-30], test_data.Open[-100:-30], label="test_data", color="b")
    # reverse the scaling transformation
    original_cases = scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()

    # the historical data used as input for forecasting
    plt.plot(test_data.index[-30:], original_cases, label='actual values', color='green')

    # Forecasted Values
    # reverse the scaling transformation
    forecasted_cases = scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten()
    # plotting the forecasted values
    plt.plot(combined_index[-60:], forecasted_cases, label='forecasted values', color='red')

    plt.xlabel('Time Step in days')
    plt.ylabel('Value')
    plt.legend()

    title = args.name.upper() + " | " + args.mode.upper() + " | " + last_date_for_printing + " Time Series"
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main()
