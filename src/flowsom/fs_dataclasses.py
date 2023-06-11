from dataclasses import dataclass


@dataclass
class FlowSOM_SOMParameters:
    """
    Data class for storing parameters related to the Self Organizing Map (SOM) part of
    the FlowSOM algorithm.

    Attributes
    ----------
    n_epochs : int, optional
        Number of epochs for SOM training. Default is 10000.
    shape : tuple[int, int], optional
        The shape (rows, columns) of the SOM grid. Default is (10, 10).
    sigma : float, optional
        The spread of the neighborhood function for SOM training. Default is 1.0.
    alpha : float, optional
        The learning rate for SOM training. Default is 0.5.
    neighbourhood_function : str, optional
        The type of neighbourhood function to use in SOM training. Possible values are
        "gaussian", "mexican_hat", "bubble", "triangle".Default is "gaussian".
    activation_distance : str, optional
        The distance measure to use in the SOM. Possible values are "euclidean",
        "cosine". Default is "euclidean".
    """

    n_epochs: int = 10
    shape: tuple[int, int] = (10, 10)
    sigma: float = 0.67
    alpha: float = 0.05
    neighbourhood_function: str = "gaussian"
    activiation_distance: str = "euclidean"


@dataclass
class FlowSOM_MSTParameters:
    """
    Data class for storing parameters related to the Minimum Spanning Tree (MST) part of
    the FlowSOM algorithm.

    Attributes
    ----------
    distance_metric : str, optional
        The distance metric to use when creating the MST. Possible values are
        "euclidean", "manhattan", "cosine". Default is "euclidean".
    """

    distance_metric: str = "euclidean"


@dataclass
class FlowSOM_HCCParameters:
    """
    Data class for storing parameters related to the
    Hierarchical Consensus Clustering (HCC) part of the FlowSOM algorithm.

    Attributes
    ----------
    n_clusters : int, optional
        The number of clusters to form in HCC. Default is 10.
    linkage_method : str, optional
        The linkage method to use in HCC. Possible values are "single", "complete",
        "average", "ward". Default is "ward".
    n_bootstrap : int, optional
        The number of bootstrap samples to generate in HCC. Default is 100.
    """

    n_clusters: int = 12
    linkage_method: str = "average"
