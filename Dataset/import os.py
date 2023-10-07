import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sqlite3

# This sets the backend for matplotlib to ensure plots are displayed
import matplotlib
matplotlib.use('TkAgg')

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' does not exist.")
        return None

    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully!")
        print(data.head())  # Display the first 5 rows for inspection
        return data
    except Exception as e:
        print(f"Error loading the data file: {e}")
        return None

def check_columns(data, required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    return not bool(missing_cols), missing_cols

def clean_data(data):
    data = data.dropna()
    # Handle outliers
    for column in data.columns:
        upper_limit = data[column].mean() + 3 * data[column].std()
        lower_limit = data[column].mean() - 3 * data[column].std()
        data[column] = data[column].apply(lambda x: upper_limit if x > upper_limit else lower_limit if x < lower_limit else x)
    return data

def visualize_data(data):
    if 'speed' in data.columns:
        data['speed'].hist()
        plt.show()
    if 'speed' in data.columns and 'distance_to_next_car' in data.columns:
        data.plot(x='speed', y='distance_to_next_car', kind='scatter')
        plt.show()

def engineer_features(data):
    if 'speed' in data.columns and 'distance_to_next_car' in data.columns:
        data['speed_distance'] = data['speed'] * data['distance_to_next_car']
    return data

def train_model(data):
    X = data.drop('acc_action', axis=1)
    y = data['acc_action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    return clf

def save_to_database(data, db_name, table_name):
    try:
        conn = sqlite3.connect(db_name)
        data.to_sql(table_name, conn, if_exists='replace')
        print("Data saved to the database successfully!")
    except Exception as e:
        print(f"Error saving data to the database: {e}")

def main():
    filepath = '/Users/a0975464400/Desktop/Essay/Project/Dataset/archive-2/sensor_raw.csv'
    required_columns = ['speed', 'distance_to_next_car', 'road_grade', 'acc_action']

    data = load_data(filepath)

    if data is not None:
        is_valid, missing_cols = check_columns(data, required_columns)
        if is_valid:
            data = clean_data(data)
            visualize_data(data)
            data = engineer_features(data)
            clf = train_model(data)
            save_to_database(data, 'my_database.db', 'my_table')
        else:
            print(f"Error: Missing columns in the data: {', '.join(missing_cols)}")

if __name__ == '__main__':
    main()
