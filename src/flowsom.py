from math import pi

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
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

        dense_mst = self.mst.toarray()

        # Hierarchical Consensus Metaclustering
        consensus_matrix = np.zeros((dense_mst.shape[0], dense_mst.shape[0]))
        for _ in range(self.hcc_param.n_bootstrap):
            bootstrap_sample = np.random.choice(
                np.arange(dense_mst.shape[0]), dense_mst.shape[0], replace=True
            )
            subsample_distance_matrix = squareform(
                pdist(dense_mst[bootstrap_sample, :])
            )
            Z = linkage(subsample_distance_matrix, method=self.hcc_param.linkage_method)
            clustering = fcluster(Z, self.hcc_param.n_clusters, criterion="maxclust")

            for i in range(len(clustering)):
                for j in range(i + 1, len(clustering)):
                    if clustering[i] == clustering[j]:
                        consensus_matrix[bootstrap_sample[i], bootstrap_sample[j]] += 1

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

    def _plot_star_chart(self, ax, max_weight, angles, values, colors, legend):
        """
        Plot a star (radar/spider) chart on a given matplotlib Axes.

        Parameters:
        ax (matplotlib.axes.Axes): The Axes object to draw the star chart on.
        max_weight (float): The maximum value for the radial axis, defining the
                            outermost circle of the star chart.
        angles (list of float): The angles in radians at which to place each axis of the
                                star chart. The last angle should be the same as the
                                first to close the plot.
        values (list of float): The values to plot for each axis. The order of the
                                values should correspond to the order of the angles.
        colors (list of str): The colors to use for each axis. The order of the colors
                              should correspond to the order of the angles and values.
        legend (list): A list to which the filled areas of the plot will be appended for
                       later use in a legend.

        This function creates a star chart where each axis represents a different
        category, and the length of the axis represents the value for that category. The
        axes are arranged in a circular fashion, and consecutive axes are connected to
        each other, forming a star shape.

        The chart is drawn on the given Axes object, and the filled areas of the plot
        are added to the given legend list.

        The grid and the border (spine) of the polar plot are semi-transparent to
        improve readability.
        """
        ax.set_rorigin(0)
        ax.set_ylim(0, max_weight)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(alpha=0.25)
        ax.spines["polar"].set_alpha(0.25)
        for k in range(len(angles) - 1):
            ax.plot(
                [0, angles[k], angles[k + 1], 0],
                [0, values[k], values[k], 0],
                color=colors[k],
            )
            (fill,) = ax.fill(
                [angles[k], angles[k + 1], 0],
                [values[k], values[k], 0],
                color=colors[k],
                alpha=0.5,
            )
            legend.append(fill)

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
            figsize=(self.som_param.shape[0] * 2, self.som_param.shape[1] * 2),
        )
        fig.suptitle("SOM Node Star Charts")
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        legend_fills = []
        for i in range(self.som_param.shape[0]):
            for j in range(self.som_param.shape[1]):
                values = np.concatenate((weights[i, j], weights[i, j][:1]))
                self._plot_star_chart(
                    ax[i, j], max_weight, angles, values, colors, legend_fills
                )

        legend = fig.legend(legend_fills, labels[:-1], loc="upper right")
        for line in legend.get_lines():
            line.set_linewidth(5)

        plt.savefig("data/plot_som.png")

    def plot_mst(self):
        if self.som is None:
            return None
        if self.mst is None:
            return None

        # Get the weights of the SOM
        weights = self.som.get_weights()
        weights = weights.reshape(-1, weights.shape[-1])
        max_weight = np.max(weights)

        # Get the column names from the DataFrame to use as labels
        labels = self.data.columns.values.tolist()

        # Define the properties for the radar chart
        num_vars = weights.shape[-1]
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        labels += labels[:1]

        mst = self.mst.toarray()
        G = nx.from_numpy_array(mst)
        pos = nx.kamada_kawai_layout(G)

        min_x = min(x for x, _ in pos.values())
        max_x = max(x for x, _ in pos.values())
        min_y = min(y for _, y in pos.values())
        max_y = max(y for _, y in pos.values())

        scale = 0.9
        offset = (1 - scale) / 2
        pos = {
            node: (
                scale * (x - min_x) / (max_x - min_x) + offset,
                scale * (y - min_y) / (max_y - min_y) + offset,
            )
            for node, (x, y) in pos.items()
        }

        gs = gridspec.GridSpec(1000, 1000)
        fig = plt.figure(figsize=(25, 25))

        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        mcluster_colors = plt.cm.tab20(np.linspace(0, 1, self.hcc_param.n_clusters))

        legend_fills = []
        for i_node, (x, y) in pos.items():
            grid_x = int(x * 1000)
            grid_y = int(y * 1000)
            ax = plt.subplot(
                gs[grid_y - 10 : grid_y + 10, grid_x - 10 : grid_x + 10], polar=True
            )
            ax.set_autoscale_on(False)
            values = np.concatenate((weights[i_node], weights[i_node][:1]))
            self._plot_star_chart(ax, max_weight, angles, values, colors, legend_fills)

        ax_edges = fig.add_subplot(111)
        ax_edges.set_facecolor("#00000000")

        ax_mclustering = fig.add_subplot(111)
        ax_mclustering.set_facecolor("#00000000")
        for i_node, (x, y) in pos.items():
            cluster = self.hcc[i_node]
            circle = Circle(
                (x, 1 - y), radius=0.012, color=mcluster_colors[cluster - 1], alpha=0.18
            )
            ax_mclustering.add_artist(circle)

        for edge in G.edges:
            node1, node2 = edge
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            ax_edges.plot(
                [x1, x2],
                [1 - y1, 1 - y2],
                color="k",
                linewidth=0.5,
                transform=plt.gca().transAxes,
            )

        legend = fig.legend(legend_fills, labels[:-1])
        for line in legend.get_lines():
            line.set_linewidth(5)

        ax_edges.set_xticklabels([])
        ax_edges.set_yticklabels([])
        plt.savefig("data/plot_som.png")
