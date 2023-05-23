from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from flowsom_h import (
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)


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
        print(som.get_weights().shape)

        # Build Minimum Spanning Tree
        weights = som.get_weights()
        weights_2d = weights.reshape(-1, weights.shape[-1])
        distance_matrix = squareform(pdist(weights_2d, self.mst_param.distance_metric))
        self.mst = minimum_spanning_tree(distance_matrix)

        # Hierarchical Consensus Metaclustering
        consensus_matrix = np.zeros((self.mst.shape[0], self.mst.shape[0]))
        for _ in range(self.hcc_param.n_bootstrap):
            subsample = np.random.choice(
                len(distance_matrix), size=len(distance_matrix), replace=True
            )
            subsample_distance_matrix = distance_matrix[subsample][:, subsample]
            Z = linkage(subsample_distance_matrix, method=self.hcc_param.linkage_method)
            labels = fcluster(Z, self.hcc_param.n_clusters, criterion="maxclust")
            consensus_matrix += labels[:, None] == labels

        consensus_matrix /= self.hcc_param.n_bootstrap
        Z = linkage(consensus_matrix, method=self.hcc_param.linkage_method)
        self.hcc = fcluster(Z, self.hcc_param.n_clusters, criterion="maxclust")

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
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
            print("SOM has not been trained. Call 'fit' method before prediction.")
        if self.mst is None:
            print("MST has not been computed. Call 'fit' method before prediction.")
        if self.hcc is None:
            print("HCC has not been performed. Call 'fit' method before prediction.")

        # Get the winning neuron for each data point in the SOM
        winners = np.array([self.som.winner(x) for x in data.values])

        # Map winners to their corresponding metaclusters in the hierarchical
        # consensus clustering
        winners_1d = np.ravel_multi_index(winners.T, self.som.get_weights().shape[:2])
        clusters = self.hcc[winners_1d]

        # Return original data with predicted clusters
        predictions = data.copy()
        predictions["cluster"] = clusters

        return predictions

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

    def plot_som(self):
        """
        Plots the SOM as a grid of star charts.
        Each neuron in the SOM is represented as a star chart, with the weights
        of the neuron serving as dimensions.
        """
        if self.som is None:
            print("[AssertionError]: SOM has not been trained yet")
        if self.data is None:
            print("[AssertionError]: Data is not loaded")

        # Get the weights of the SOM
        weights = self.som.get_weights()
        max_weight = np.max(weights)

        # Get the column names from the DataFrame to use as labels
        labels = self.data.columns.values.tolist()

        # Define the properties for the radar chart
        num_vars = weights.shape[-1]
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        labels += labels[:1]

        fig, ax = plt.subplots(
            self.som_param.shape[0],
            self.som_param.shape[1],
            subplot_kw=dict(polar=True),
            figsize=(21, 21),
        )
        fig.suptitle("SOM Node Star Charts", fontsize=18, fontweight="bold")
        plt.subplots_adjust(wspace=1, hspace=1)

        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

        for i in range(self.som_param.shape[0]):
            for j in range(self.som_param.shape[1]):
                values = np.concatenate((weights[i, j], weights[i, j][:1]))
                ax[i, j].set_rorigin(0)
                ax[i, j].set_ylim(0, max_weight)  # Adjusts the radius
                ax[i, j].set_xticks(angles[:-1])
                ax[i, j].set_xticklabels(labels[:-1])
                ax[i, j].fill(angles, values, "b", alpha=0.1)
                ax[i, j].set_title(f"Node: ({i}, {j})")
                ax[i, j].set_yticklabels([])
                for k in range(num_vars):
                    ax[i, j].plot(
                        [0, angles[k], angles[k + 1], 0],
                        [0, values[k], values[k], 0],
                        color=colors[k],
                    )
                    ax[i, j].fill(
                        [angles[k], angles[k + 1], 0],
                        [values[k], values[k], 0],
                        color=colors[k],
                        alpha=0.5,
                    )

        plt.savefig("data/plot_som.png")

    def plt_mst(self):
        pass
