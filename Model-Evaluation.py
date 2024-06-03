# model_evaluation.py

"""
Model Evaluation Module for Data Summarization with Bilevel Optimization

This module contains functions for evaluating the performance of the data summarization and dimensionality reduction techniques.

Techniques Used:
- Explained variance
- Reconstruction error
- Computational efficiency

Libraries/Tools:
- numpy
- pandas
- scikit-learn
- matplotlib

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

class ModelEvaluation:
    def __init__(self, original_data_path='data/processed/processed_data.csv', results_dir='results/'):
        """
        Initialize the ModelEvaluation class.
        
        :param original_data_path: str, path to the original processed data
        :param results_dir: str, directory to save evaluation results
        """
        self.original_data = pd.read_csv(original_data_path)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def load_reduced_data(self, filename):
        """
        Load reduced data from a CSV file.
        
        :param filename: str, name of the CSV file
        :return: DataFrame, loaded reduced data
        """
        data = pd.read_csv(filename)
        return data

    def evaluate_explained_variance(self, reduced_data):
        """
        Evaluate explained variance for the reduced data.
        
        :param reduced_data: DataFrame, reduced data
        :return: float, explained variance score
        """
        pca = PCA(n_components=reduced_data.shape[1])
        pca.fit(reduced_data)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        return explained_variance

    def evaluate_reconstruction_error(self, reduced_data, technique):
        """
        Evaluate reconstruction error for the reduced data.
        
        :param reduced_data: DataFrame, reduced data
        :param technique: str, name of the dimensionality reduction technique
        :return: float, mean squared error
        """
        scaler = StandardScaler()
        original_data_scaled = scaler.fit_transform(self.original_data)
        reduced_data_scaled = scaler.fit_transform(reduced_data)
        
        if technique == 'PCA':
            pca = PCA(n_components=reduced_data.shape[1])
            reconstructed_data = pca.inverse_transform(reduced_data_scaled)
        elif technique == 't-SNE':
            tsne = TSNE(n_components=reduced_data.shape[1])
            reconstructed_data = tsne.fit_transform(reduced_data_scaled)
        elif technique == 'Isomap':
            isomap = Isomap(n_components=reduced_data.shape[1])
            reconstructed_data = isomap.fit_transform(reduced_data_scaled)
        elif technique == 'LLE':
            lle = LocallyLinearEmbedding(n_components=reduced_data.shape[1])
            reconstructed_data = lle.fit_transform(reduced_data_scaled)
        else:
            raise ValueError("Unknown technique: " + technique)
        
        mse = mean_squared_error(original_data_scaled, reconstructed_data)
        return mse

    def evaluate_computational_efficiency(self, start_time, end_time):
        """
        Evaluate computational efficiency.
        
        :param start_time: float, start time of the operation
        :param end_time: float, end time of the operation
        :return: float, computational time
        """
        computational_time = end_time - start_time
        return computational_time

    def evaluate(self, technique):
        """
        Perform full evaluation for a given dimensionality reduction technique.
        
        :param technique: str, name of the dimensionality reduction technique
        """
        reduced_data_path = f'data/processed/{technique.lower()}_data.csv'
        reduced_data = self.load_reduced_data(reduced_data_path)

        explained_variance = self.evaluate_explained_variance(reduced_data)
        reconstruction_error = self.evaluate_reconstruction_error(reduced_data, technique)
        
        metrics = {
            'explained_variance': explained_variance,
            'reconstruction_error': reconstruction_error
        }
        
        metrics_path = os.path.join(self.results_dir, f'{technique.lower()}_evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Evaluation metrics for {technique} saved to {metrics_path}")

    def plot_evaluation_results(self, technique, metrics):
        """
        Plot evaluation results.
        
        :param technique: str, name of the dimensionality reduction technique
        :param metrics: dict, evaluation metrics
        """
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Evaluation Results for {technique}')
        plt.savefig(os.path.join(self.results_dir, f'{technique.lower()}_evaluation_results.png'))
        plt.show()

if __name__ == "__main__":
    evaluator = ModelEvaluation(original_data_path='data/processed/processed_data.csv', results_dir='results/')
    
    # Evaluate PCA
    evaluator.evaluate('PCA')
    
    # Evaluate t-SNE
    evaluator.evaluate('t-SNE')
    
    # Evaluate Isomap
    evaluator.evaluate('Isomap')
    
    # Evaluate LLE
    evaluator.evaluate('LLE')

    print("Model evaluation completed and results saved.")
