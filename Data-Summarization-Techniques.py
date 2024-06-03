# data_summarization_techniques.py

"""
Data Summarization Techniques for Data Summarization with Bilevel Optimization

This module contains functions for implementing various data summarization techniques to reduce and summarize data effectively.

Techniques Used:
- Clustering
- Principal Component Analysis (PCA)
- Autoencoders

Libraries/Tools:
- numpy
- pandas
- scikit-learn
- tensorflow
- keras

"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class DataSummarization:
    def __init__(self, data, n_clusters=5, n_components=10, encoding_dim=5):
        """
        Initialize the DataSummarization class.
        
        :param data: DataFrame, input data
        :param n_clusters: int, number of clusters for clustering
        :param n_components: int, number of principal components for PCA
        :param encoding_dim: int, dimension of the encoding layer for autoencoders
        """
        self.data = data
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.encoding_dim = encoding_dim

    def cluster_data(self):
        """
        Apply KMeans clustering to the data.
        
        :return: DataFrame, clustered data with cluster labels
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(scaled_data)
        clustered_data = self.data.copy()
        clustered_data['Cluster'] = cluster_labels
        return clustered_data

    def apply_pca(self):
        """
        Apply Principal Component Analysis (PCA) to the data.
        
        :return: DataFrame, data with principal components
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        pca = PCA(n_components=self.n_components)
        principal_components = pca.fit_transform(scaled_data)
        pca_data = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(self.n_components)])
        return pca_data

    def build_autoencoder(self, input_dim):
        """
        Build an autoencoder model.
        
        :param input_dim: int, dimension of the input data
        :return: Model, autoencoder model
        """
        input_layer = Input(shape=(input_dim,))
        encoding_layer = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoding_layer = Dense(input_dim, activation='sigmoid')(encoding_layer)
        autoencoder = Model(inputs=input_layer, outputs=decoding_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def apply_autoencoder(self, epochs=50, batch_size=32):
        """
        Apply autoencoder to the data for dimensionality reduction.
        
        :param epochs: int, number of training epochs
        :param batch_size: int, batch size for training
        :return: DataFrame, data with reduced dimensions
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        autoencoder = self.build_autoencoder(scaled_data.shape[1])
        autoencoder.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
        encoded_data = autoencoder.predict(scaled_data)
        encoded_df = pd.DataFrame(encoded_data, columns=[f'Feature_{i+1}' for i in range(self.encoding_dim)])
        return encoded_df

    def summarize_data(self):
        """
        Apply all summarization techniques and save the results.
        """
        clustered_data = self.cluster_data()
        clustered_data.to_csv('data/processed/clustered_data.csv', index=False)
        print("Clustered data saved to data/processed/clustered_data.csv")

        pca_data = self.apply_pca()
        pca_data.to_csv('data/processed/pca_data.csv', index=False)
        print("PCA data saved to data/processed/pca_data.csv")

        autoencoder_data = self.apply_autoencoder()
        autoencoder_data.to_csv('data/processed/autoencoder_data.csv', index=False)
        print("Autoencoder data saved to data/processed/autoencoder_data.csv")

if __name__ == "__main__":
    data = pd.read_csv('data/processed/processed_data.csv')
    summarization = DataSummarization(data, n_clusters=5, n_components=10, encoding_dim=5)
    
    # Apply data summarization techniques
    summarization.summarize_data()
    print("Data summarization completed and results saved.")
