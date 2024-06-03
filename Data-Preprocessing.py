# data_preprocessing.py

"""
Data Preprocessing Module for Data Summarization with Bilevel Optimization

This module contains functions for collecting, cleaning, normalizing, and preparing data for optimization and summarization.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction

Libraries/Tools:
- pandas
- numpy
- scikit-learn

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataPreprocessing:
    def __init__(self, raw_data_dir='data/raw/', processed_data_dir='data/processed/'):
        """
        Initialize the DataPreprocessing class.
        
        :param raw_data_dir: str, directory containing raw data
        :param processed_data_dir: str, directory to save processed data
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

    def load_data(self, filename):
        """
        Load data from a CSV file.
        
        :param filename: str, name of the CSV file
        :return: DataFrame, loaded data
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing null values and duplicates.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.dropna().drop_duplicates()
        return data

    def normalize_data(self, data):
        """
        Normalize the data using standard scaling.
        
        :param data: DataFrame, input data
        :return: DataFrame, normalized data
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        return pd.DataFrame(normalized_data, columns=data.columns)

    def extract_features(self, data, n_components=10):
        """
        Extract features using Principal Component Analysis (PCA).
        
        :param data: DataFrame, input data
        :param n_components: int, number of principal components to extract
        :return: DataFrame, data with extracted features
        """
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data)
        feature_data = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
        return feature_data

    def save_data(self, data, filename):
        """
        Save the processed data to a CSV file.
        
        :param data: DataFrame, processed data
        :param filename: str, name of the output CSV file
        """
        os.makedirs(self.processed_data_dir, exist_ok=True)
        filepath = os.path.join(self.processed_data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def preprocess(self, filename, n_components=10):
        """
        Execute the full preprocessing pipeline.
        
        :param filename: str, name of the raw data file
        :param n_components: int, number of principal components to extract
        """
        # Load data
        data = self.load_data(filename)
        
        # Clean data
        data = self.clean_data(data)
        
        # Normalize data
        data = self.normalize_data(data)
        
        # Extract features
        data = self.extract_features(data, n_components)
        
        # Save processed data
        self.save_data(data, 'processed_data.csv')
        print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocessing = DataPreprocessing(raw_data_dir='data/raw/', processed_data_dir='data/processed/')
    
    # Execute the preprocessing pipeline
    preprocessing.preprocess('raw_data.csv', n_components=10)
    print("Data preprocessing completed and data saved.")
