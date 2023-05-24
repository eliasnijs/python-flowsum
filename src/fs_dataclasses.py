from dataclasses import dataclass


@dataclass
class FlowSOM_SOMParameters:
    n_epochs: int = 10000
    shape: tuple[int, int] = (10, 10)
    sigma: float = 1.0
    alpha: float = 0.5
    neighbourhood_function: str = "gaussian"
    activiation_distance: str = "euclidean"


@dataclass
class FlowSOM_MSTParameters:
    distance_metric: str = "euclidean"


@dataclass
class FlowSOM_HCCParameters:
    n_clusters: int = 10
    linkage_method: str = "ward"
    n_bootstrap: int = 100
