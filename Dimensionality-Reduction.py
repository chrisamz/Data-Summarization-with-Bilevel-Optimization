# dimensionality_reduction.py

"""
Dimensionality Reduction Techniques for Data Summarization with Bilevel Optimization

This module contains functions for implementing various dimensionality reduction techniques to reduce the number of variables while preserving essential information.

Techniques Used:
- Linear dimensionality reduction
- Nonlinear dimensionality reduction

Libraries/Tools:
- numpy
- pandas
- scikit-learn

"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

class DimensionalityReduction:
    def __init__(self, data, n_components=10):
        """
        Initialize the DimensionalityReduction class.
        
        :param data: DataFrame, input data
        :param n_components: int, number of components to reduce to
        """
        self.data = data
        self.n_components = n_components

    def standardize_data(self):
        """
        Standardize the data using standard scaling.
        
        :return: ndarray, standardized data
        """
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(self.data)
        return standardized_data

    def apply_pca(self):
        """
        Apply Principal Component Analysis (PCA) to the data.
        
        :return: DataFrame, data with principal components
        """
        standardized_data = self.standardize_data()
        pca = PCA(n_components=self.n_components)
        principal_components = pca.fit_transform(standardized_data)
        pca_data = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(self.n_components)])
        return pca_data

    def apply_tsne(self):
        """
        Apply t-Distributed Stochastic Neighbor Embedding (t-SNE) to the data.
        
        :return: DataFrame, data with t-SNE components
        """
        standardized_data = self.standardize_data()
        tsne = TSNE(n_components=self.n_components)
        tsne_components = tsne.fit_transform(standardized_data)
        tsne_data = pd.DataFrame(tsne_components, columns=[f't-SNE{i+1}' for i in range(self.n_components)])
        return tsne_data

    def apply_isomap(self):
        """
        Apply Isomap to the data.
        
        :return: DataFrame, data with Isomap components
        """
        standardized_data = self.standardize_data()
        isomap = Isomap(n_components=self.n_components)
        isomap_components = isomap.fit_transform(standardized_data)
        isomap_data = pd.DataFrame(isomap_components, columns=[f'Isomap{i+1}' for i in range(self.n_components)])
        return isomap_data

    def apply_lle(self):
        """
        Apply Locally Linear Embedding (LLE) to the data.
        
        :return: DataFrame, data with LLE components
        """
        standardized_data = self.standardize_data()
        lle = LocallyLinearEmbedding(n_components=self.n_components)
        lle_components = lle.fit_transform(standardized_data)
        lle_data = pd.DataFrame(lle_components, columns=[f'LLE{i+1}' for i in range(self.n_components)])
        return lle_data

    def save_data(self, data, filename):
        """
        Save the reduced data to a CSV file.
        
        :param data: DataFrame, reduced data
        :param filename: str, name of the output CSV file
        """
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def reduce_and_save(self):
        """
        Apply all dimensionality reduction techniques and save the results.
        """
        pca_data = self.apply_pca()
        self.save_data(pca_data, 'data/processed/pca_data.csv')
        
        tsne_data = self.apply_tsne()
        self.save_data(tsne_data, 'data/processed/tsne_data.csv')

        isomap_data = self.apply_isomap()
        self.save_data(isomap_data, 'data/processed/isomap_data.csv')

        lle_data = self.apply_lle()
        self.save_data(lle_data, 'data/processed/lle_data.csv')
        print("Dimensionality reduction completed and results saved.")

if __name__ == "__main__":
    data = pd.read_csv('data/processed/processed_data.csv')
    dim_reduction = DimensionalityReduction(data, n_components=10)
    
    # Apply dimensionality reduction techniques
    dim_reduction.reduce_and_save()
    print("Dimensionality reduction completed and results saved.")
