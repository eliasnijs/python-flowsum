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
from fs_plotting import fs_plot_feature_planes, fs_plot_mst, fs_plot_som
from fs_reporting import report


class FlowSOM:
    """
    FlowSOM is a class representing the FlowSOM algorithm. This algorithm is widely used
    in flow and mass cytometry data analysis.

    The FlowSOM model includes a Self-Organizing Map (SOM), construction of a
    Minimum Spanning Tree (MST), and Metaclustering. These components work in tandem to
    generate visualizations of data clusters and their relationships.

    Attributes
    ----------
    som_param: FlowSOM_SOMParameters
        Parameters dictating the behavior and construction of the Self-Organizing Map.
    mst_param: FlowSOM_MSTParameters
        Parameters governing the construction of the Minimum Spanning Tree.
    hcc_param: FlowSOM_HCCParameters
        Parameters specifying the nature and construction of the Hierarchical Consensus
        Metaclustering.
    data: pd.DataFrame
        The dataset used to fit the model.
    som: MiniSom
        The trained Self-Organizing Map after fitting to the data.
    mst: scipy.sparse.coo.coo_matrix
        The resulting Minimum Spanning Tree after construction.
    hcc: np.ndarray
        The resulting Metaclusters post-generation.

    Usage
    -----
    >>> fs = FlowSOM(data, som_param, mst_param, hcc_param)
    >>> fs.fit(data)

    Notes
    -----
    The FlowSOM class provides a comprehensive implementation of the FlowSOM algorithm,
    providing users with a powerful tool for cytometry data analysis. All components of
    the algorithm (SOM, MST, and Metaclustering) have tunable parameters for flexibility
    and fine-tuning.
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
        Trains the FlowSOM model on the provided dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be clustered.
        verbose : bool, optional
            If set to True, progress updates will be displayed during training. Defaults
            to False.

        Returns
        -------
        self : object
            The trained model instance.

        Usage
        -----
        >>> model = FlowSOM().fit(data, verbose=True)

        Notes
        -----
        This function is instrumental in training the FlowSOM model on a provided
        dataset. Training involves adjusting the model parameters to best fit the data.
        The verbose parameter can be used to monitor the progress of training. Upon
        completion, the function returns the trained model instance.
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
            random_seed=0,
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
        Utilizes the trained FlowSOM model to designate cluster assignments for new,
        unseen data.

        Parameters
        ----------
        data : pd.DataFrame
            The fresh input dataset that requires clustering.

        Returns
        -------
        predictions : pd.DataFrame
            A DataFrame combining the original data and their corresponding predicted
            cluster assignments.

        Usage
        -----
        >>> predictions = model.predict(new_data)

        Notes
        -----
        This function is particularly useful when you have a pre-trained FlowSOM model
        and wish to apply it to new data for clustering.
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
        Fit the model to the given data and return predicted cluster assignments.

        This function trains the FlowSOM model on the provided data and returns the
        predicted cluster assignments for each data point. If verbose is True, it will
        print progress updates during training.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to fit the model to.

        verbose : bool, optional
            If True, print progress updates. Default is False.

        Returns
        -------
        np.ndarray
            The predicted cluster assignments for each data point.

        Usage
        -----
        >>> predicted_clusters = fit_predict(data, verbose=True)

        Notes
        -----
        This function combines the steps of fitting the model to the data and predicting
        cluster assignments in one step. It is especially useful in cases where the
        model fit and predictions are desired in one go.
        """
        self.fit(data, verbose)
        return self.predict(data)

    def plot_som(self, save=None, show=True, show_clusters=True):
        """
        Generates a grid of star charts representing the FlowSOM
        Self-Organizing Map (SOM). Each neuron in the SOM is visualized as a star chart,
        with the neuron's weights acting as dimensions.

        Parameters
        ----------
        save : str, optional
            Path where the generated plot will be saved. If None, the plot will not be
            saved. Default is None.
        show : bool, optional
            If True, the plot will be displayed. Default is True.
        show_clusters : bool, optional
            If True, different clusters in the SOM will be marked with different colors.
            Default is True.

        Usage
        -----
        >>> fs_plot_som(fs, save="path/to/save", show=True, show_clusters=True)

        Notes
        -----
        The fs_plot_som function presents the trained FlowSOM Self-Organizing Map in a
        visually engaging manner. Each neuron's weights are depicted as star charts in a
        grid format, allowing for easy comparison and identification of patterns. If
        enabled, the function can also highlight different clusters within the SOM with
        distinct colors.
        """
        if self.som is None:
            print("[Error]: SOM has not been trained yet")
            return None
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None

        fs_plot_som(self, save, show, show_clusters)

    def plot_mst(self, save=None, show=True, show_clusters=True):
        """
        Generates a Minimum Spanning Tree (MST) plot for the FlowSOM model, where each
        node in the grid is visualized as a star chart.

        Parameters
        ----------
        save : str, optional
            Path where the generated plot will be saved. If None, the plot will not be
            saved. Default is None.
        show : bool, optional
            If True, the plot will be displayed. Default is True.
        show_clusters : bool, optional
            If True, different clusters in the MST will be marked with different colors.
            Default is True.

        Usage
        -----
        >>> fs_plot_mst(fs, save="path/to/save", show=True, show_clusters=True)

        Notes
        -----
        The fs_plot_mst function provides a visual representation of the
        Minimum Spanning Tree (MST) created by the FlowSOM model. Each node in the MST,
        corresponding to a neuron in the SOM grid, is visualized as a star chart. The
        function allows easy visualization and analysis of the relationships between
        different neurons (and therefore data clusters) in the trained model.
        """
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None
        if self.som is None:
            print("[Error]: SOM has not been trained yet")
            return None
        if self.mst is None:
            print("[Error]: MST has not been constructed yet")
            return None

        fs_plot_mst(self, save, show, show_clusters)

    def plot_feature_planes(self, save=None, show=True):
        """
        Generates a plot showcasing the feature planes of the Self-Organizing Map (SOM)
        in the FlowSOM model.

        Parameters
        ----------
        save : str, optional
            Path where the generated plot will be saved. If None, the plot will not be
            saved. Default is None.
        show : bool, optional
            If True, the plot will be displayed. Default is True.

        Usage
        -----
        >>> fs_plot_feature_planes(fs, save="path/to/save", show=True)

        Notes
        -----
        The fs_plot_feature_planes function visualizes the feature planes of the
        trained SOM in the FlowSOM model. Each subplot corresponds to a particular
        feature from the data, represented as a heatmap on the SOM grid. This plot
        provides an insightful visualization of how different features contribute to the
        formation of clusters in the SOM.
        """
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None
        if self.som is None:
            print("[Error]: SOM has not been trained yet")
            return None

        fs_plot_feature_planes(self, save, show)

    # TODO(Elias): Implement this
    def as_df(self, lazy=True):
        pass

    # TODO(Elias): Implement this
    def as_adata(self, lazy=True):
        pass

    # TODO(Elias): Implement this
    def report(self, path, generate_images=True):
        return report(self, path, generate_images)
