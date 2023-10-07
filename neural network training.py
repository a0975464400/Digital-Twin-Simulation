import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
data_path = '/Users/a0975464400/Desktop/Essay/Project/Dataset/sensor_data.csv'
sensor_data = pd.read_csv(data_path)

# Preprocess data
def preprocess_data(data):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['speed', 'distance_to_next_car', 'road_grade']])
    
    # Convert target to one-hot encoding
    target = pd.get_dummies(data['acc_action']).values
    
    return scaled_data, target, scaler

# Prepare sequences for LSTM
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = target[i+seq_length]
        X.append(seq)
        y.append(label)
    
    return np.array(X), np.array(y)

# Constants
SEQ_LENGTH = 10

# Preprocess data and create sequences
scaled_data, target, scaler = preprocess_data(sensor_data)
X, y = create_sequences(scaled_data, target, SEQ_LENGTH)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, verbose=1)

# Evaluate model performance
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

train_accuracy, val_accuracy
