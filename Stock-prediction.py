import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load historical stock price data (replace 'AAPL.csv' with your data file)
df = pd.read_csv('bharat.csv')

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(df)

# Define the number of previous days to use for prediction
look_back = 30

# Create a function to prepare the data
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data, look_back)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape the data for LSTM input (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to the original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# Calculate root mean squared error (RMSE)
train_rmse = np.sqrt(mean_squared_error(df.values[look_back:train_size], train_predict))
test_rmse = np.sqrt(mean_squared_error(df.values[train_size + look_back:], test_predict))

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Plot the predictions

train_plot = np.empty_like(df)
train_plot[:, :] = np.nan
train_plot[look_back:train_size, :] = train_predict

test_plot = np.empty_like(df)
test_plot[:, :] = np.nan
test_plot[train_size + look_back:, :] = test_predict

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual Stock Price', color='b')
plt.plot(df.index, train_plot, label='Train Predictions', color='g')
plt.plot(df.index, test_plot, label='Test Predictions', color='r')
plt.legend()
plt.show()
