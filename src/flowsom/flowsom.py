from typing import Optional

import anndata
import dask.dataframe as dd
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering

from .fs_dataclasses import (
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)
from .fs_plotting import fs_plot_feature_planes, fs_plot_mst, fs_plot_som
from .fs_reporting import fs_report
from .fs_utils import fs_log


class FlowSOM(BaseEstimator, ClusterMixin):
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
    >>> model = FlowSOM(som_param, mst_param, hcc_param).fit(data)

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
        n_clusters=None,
    ):
        self.som_param = som_param
        self.mst_param = mst_param
        self.hcc_param = hcc_param
        self.data = None
        self.som = None
        self.mst = None
        self.hcc = None
        if n_clusters is not None:
            self.hcc_param.n_clusters = n_clusters

    def fit(self, X: pd.DataFrame, y=None, verbose=False):
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
        data = X
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        self.data = data

        # Train Self Organizing Map
        fs_log("constructing model (1/3): training SOM...", verbose)
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
            data.values, num_iteration=self.som_param.n_iterations, verbose=verbose
        )
        self.som = som

        # Save average marker values for the fitted data, tranform so all values
        # are positive and normalize
        winners = np.array([som.winner(x) for x in data.values])
        winners = np.ravel_multi_index(winners.T, som.get_weights().shape[:2])

        n_nodes = self.som_param.shape[0] * self.som_param.shape[1]
        nodes_avg_markers = np.zeros((n_nodes, len([*data.columns])))

        lowest_marker = 0
        for i in range(n_nodes):
            cells = np.where(winners == i)[0]
            nodes_avg_markers[i] = np.mean(data.values[cells], axis=0)
            node_lowest_marker = np.min(nodes_avg_markers[i])
            if node_lowest_marker < lowest_marker:
                lowest_marker = node_lowest_marker
        for i in range(n_nodes):
            nodes_avg_markers[i] -= lowest_marker
            print(np.all(nodes_avg_markers[i] >= 0))

        min_marker = np.min(nodes_avg_markers)
        max_marker = np.max(nodes_avg_markers)
        nodes_avg_markers = (nodes_avg_markers - min_marker) / (max_marker - min_marker)

        self.nodes_avg_markers = nodes_avg_markers.reshape(
            self.som_param.shape[0], self.som_param.shape[1], len([*self.data.columns])
        )

        # Build Minimum Spanning Tree
        fs_log("constructing model (2/3): building MST...", verbose)
        weights = som.get_weights()
        nodes = weights.reshape(-1, weights.shape[-1])
        distance_matrix = squareform(pdist(nodes, self.mst_param.distance_metric))
        self.mst = minimum_spanning_tree(distance_matrix)

        # Hierarchical Consensus Metaclustering
        fs_log(
            "constructing model (3/3): building hierarchical consensus clusters...",
            verbose,
        )
        self.hcc = AgglomerativeClustering(
            n_clusters=self.hcc_param.n_clusters, linkage=self.hcc_param.linkage_method
        ).fit_predict(nodes)

        fs_log("model constructed\n", verbose)
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

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

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

    def plot_som(self, save=None, show=True, show_mclusters=True):
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
        show_mclusters : bool, optional
            If True, different clusters in the SOM will be marked with different colors.
            Default is True.

        Usage
        -----
        >>> fs_plot_som(fs, save="path/to/save", show=True, show_mclusters=True)

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
        if self.hcc is None:
            print("[Error]: HCC has not been trained yet")
            return None

        fs_plot_som(self, save, show, show_mclusters)

    def plot_mst(self, save=None, show=True, show_mclusters=True):
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
        show_mclusters : bool, optional
            If True, different clusters in the MST will be marked with different colors.
            Default is True.

        Usage
        -----
        >>> fs_plot_mst(fs, save="path/to/save", show=True, show_mclusters=True)

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
        if self.hcc is None:
            print("[Error]: HCC has not been trained yet")
            return None

        fs_plot_mst(self, save, show, show_mclusters)

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

    def report(self, save="report/", generate_images=True, verbose=True):
        """
        Generates a detailed report for a trained FlowSOM model.

        This function produces a report in markdown syntax that provides an overview of
        the FlowSOM analysis results, including key statistics, model parameters, and
        various visualizations. The report is written to a user-specified directory. If
        'generate_images' is True, the function generates new images in the 'images'
        subdirectory under the directory specified in 'save'. If False, it assumes the
        images are already present in the subdirectory.

        Parameters
        ----------
        save : str, optional
            The directory to which the report will be saved. Defaults to "report/".

        generate_images : bool, optional
            If set to True, the function will generate new images in the 'images'
            subdirectory under the directory specified in 'save'. If False, it will use
            the images already present in this subdirectory. Defaults to True.

        verbose : bool
            If set to True, the function will log the state of the function. If False,
            it will not. Defaults to True.

        Returns
        -------
        None

        Usage
        -----
        >>> model.report(save="report/", generate_images=True)

        Notes
        -----
        This function is instrumental in understanding the results of a FlowSOM
        analysis. The generated report includes key statistics at both metacluster and
        individual neuron levels, as well as a comprehensive summary of the model
        parameters. In addition, it provides various visualizations, which can give
        insights into the structure and organization of the input data.
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
        if self.hcc is None:
            print("[Error]: HCC has not been trained yet")
            return None
        return fs_report(self, save, generate_images, verbose)

    def as_df(self, lazy=True, n_partitions=10):
        """
        Converts the input data to a DataFrame.

        Parameters
        ----------
        lazy : bool, optional
            If set to True, a Dask DataFrame will be returned, which allows for
            lazy evaluation. If False, a pandas DataFrame will be returned. Defaults
            to True.

        Returns
        -------
        df : dask.DataFrame or pandas.DataFrame
            The input data as a DataFrame.

        Usage
        -----
        >>> model = FlowSOM().fit(data)
        >>> df = model.as_df(lazy=True)
        """
        if self.data is None:
            print("[Error]: Data is not loaded")
            return None
        if lazy:
            return dd.from_pandas(self.data, npartitions=n_partitions)
        else:
            return self.data.copy()

    def as_adata(self, lazy=True, n_partitions=10):
        """
        Converts the input data to an AnnData object.

        Parameters
        ----------
        lazy : bool, optional
            If set to True, the AnnData object's X attribute, which stores the
            data matrix, will be a Dask array, which allows for lazy evaluation.
            If False, it will be a standard numpy array. Defaults to True.

        n_partitions : int, optional
            Number of partitions for the Dask DataFrame. Only used when `lazy` is True.
            Defaults to 10.

        Returns
        -------
        adata : anndata.AnnData
            The input data as an AnnData object.

        Usage
        -----
        >>> model = FlowSOM().fit(data)
        >>> adata = model.as_adata(lazy=True)
        """

        if self.data is None:
            print("[Error]: Data is not loaded")
            return None
        if self.hcc is None:
            print("[Error]: HCC has not been trained yet")
            return None

        if lazy:
            X = dd.from_pandas(self.data, npartitions=n_partitions).to_dask_array(
                lengths=True
            )
        else:
            X = self.data.values

        adata = anndata.AnnData(
            X=X,
            var=pd.DataFrame(index=self.data.columns),
        )

        adata.uns["FlowSOM"] = {"metaclusters": self.hcc}

        return adata
