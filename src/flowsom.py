from typing import Optional

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from fs_dataclasses import (
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)
from fs_plotting import fs_plot_mst, fs_plot_som


class FlowSOM:
    """
    Implementation of the FlowSOM algorithm, a method used in the analysis of
    flow and mass cytometry data. It utilizes a Self-Organizing Map (SOM),
    followed by construction of a Minimum Spanning Tree (MST) and
    Metaclustering to provide a visualization of the data clusters and their
    relations.

    Attributes
    ----------
    som_param: FlowSOM_SOMParameters
        The parameters used for training the Self Organizing Map.
    mst_param: FlowSOM_MSTParameters
        The parameters used for building the Minimum Spanning Tree.
    hcc_param: FlowSOM_HCCParameters
        The parameters used for the Hierarchical Consensus Metaclustering.
    data: pd.DataFrame
        The data that was used to fit the model.
    som: MiniSom
        The trained Self Organizing Map.
    mst: scipy.sparse.coo.coo_matrix
        The built Minimum Spanning Tree.
    hcc: np.ndarray
        The generated Metaclusters.
    """

    def __init__(
        self,
        som_param: FlowSOM_SOMParameters = FlowSOM_SOMParameters(),
        mst_param: FlowSOM_MSTParameters = FlowSOM_MSTParameters(),
        hcc_param: FlowSOM_HCCParameters = FlowSOM_HCCParameters(),
    ):
        self.som_param = som_param
        self.mst_param = mst_param
        self.hcc_param = hcc_param
        self.data = None
        self.som = None
        self.mst = None
        self.hcc = None

    def fit(self, data: pd.DataFrame, verbose=False):
        """
        Fit the FlowSOM model on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to be clustered.
        verbose : bool, optional
            If true, print progress updates. Default is False.

        Returns
        -------
        self : object
            Returns self.
        """
        self.data = data

        # Train Self Organizing Map
        som = MiniSom(
            self.som_param.shape[0],
            self.som_param.shape[1],
            data.shape[1],
            sigma=self.som_param.sigma,
            learning_rate=self.som_param.alpha,
            neighborhood_function=self.som_param.neighbourhood_function,
            activation_distance=self.som_param.activiation_distance,
        )
        som.train_batch(
            data.values, num_iteration=self.som_param.n_epochs, verbose=verbose
        )
        self.som = som

        # Build Minimum Spanning Tree
        weights = som.get_weights()
        nodes = weights.reshape(-1, weights.shape[-1])
        distance_matrix = squareform(pdist(nodes, self.mst_param.distance_metric))
        self.mst = minimum_spanning_tree(distance_matrix)

        # Hierarchical Consensus Metaclustering
        consensus_matrix = np.zeros((nodes.shape[0], nodes.shape[0]))
        for _ in range(self.hcc_param.n_bootstrap):
            model = AgglomerativeClustering(n_clusters=self.hcc_param.n_clusters)
            clustering = model.fit_predict(nodes)
            for i in range(len(clustering)):
                for j in range(i + 1, len(clustering)):
                    if clustering[i] == clustering[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1
        consensus_matrix /= self.hcc_param.n_bootstrap
        Z = linkage(consensus_matrix, method="complete")
        self.hcc = fcluster(Z, self.hcc_param.n_clusters, criterion="maxclust")

        return self

    def predict(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Apply the trained FlowSOM model to predict the clusters for new data.

        Parameters
        ----------
        data : pd.DataFrame
            New input data to be clustered.

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame of original data and their predicted clusters.
        """
        if self.som is None:
            print(
                "[Error]: SOM has not been trained. Call 'fit' method before"
                + "prediction."
            )
            return None
        if self.mst is None:
            print(
                "[Error]: MST has not been computed. Call 'fit' method before"
                + "prediction."
            )
            return None
        if self.hcc is None:
            print(
                "[Error]: HCC has not been performed. Call 'fit' method before"
                + "prediction."
            )
            return None

        # Get the winning neuron for each data point in the SOM
        winners = np.array([self.som.winner(x) for x in data.values])

        # Map winners to their corresponding metaclusters in the hierarchical
        # consensus clustering
        winners_1d = np.ravel_multi_index(winners.T, self.som.get_weights().shape[:2])
        clusters = self.hcc[winners_1d]

        return clusters

    def fit_predict(self, data: pd.DataFrame, verbose=False):
        """
        Fits the model to the given data and then returns the predicted cluster
        assignments.

        Args:
            data (pd.DataFrame): Data to fit the model to.
            verbose (bool, optional): If True, print progress messages.
            Defaults to False.

        Returns:
            np.ndarray: Predicted cluster assignments.
        """
        self.fit(data, verbose)
        return self.predict(data)

    # TODO(Elias): Implement this
    def as_df(self, lazy=True):
        pass

    # TODO(Elias): Implement this
    def as_adata(self, lazy=True):
        pass

    def plot_som(self):
        if self.som is None:
            print("[Error]: SOM has not been trained yet")
            return None
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None

        fs_plot_som(self)

    def plot_mst(self):
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None
        if self.som is None:
            print("[Error]: SOM has not been trained yet")
            return None
        if self.mst is None:
            print("[Error]: MST has not been constructed yet")
            return None

        fs_plot_mst(self)
