import os

import numpy as np

from .fs_utils import fs_log


def fs_report(model, save="report/", generate_images=True, verbose=True):
    """
    Generates a detailed report for a trained FlowSOM model.

    This function produces a report in markdown syntax that provides an overview of the
    FlowSOM analysis results, including key statistics, model parameters, and various
    visualizations. The report is written to a user-specified directory. If
    'generate_images' is True, the function generates new images in the 'images'
    subdirectory under the directory specified in 'save'. If False, it assumes the
    images are already present in the subdirectory.

    Parameters
    ----------
    model : object
        The trained FlowSOM model for which the report is to be generated.

    save : str, optional
        The directory to which the report will be saved. Defaults to "report/".

    generate_images : bool, optional
        If set to True, the function will generate new images in the 'images'
        subdirectory under the directory specified in 'save'. If False, it will use the
        images already present in this subdirectory. Defaults to True.

    verbose : bool
        If set to True, the function will log the state of the function. If False,
        it will not. Defaults to True.

    Returns
    -------
    None

    Usage
    -----
    > fs_report(model, save="report/", generate_images=True)

    Notes
    -----
    This function is instrumental in understanding the results of a FlowSOM analysis.
    The generated report includes key statistics at both metacluster and individual
    neuron levels, as well as a comprehensive summary of the model parameters. In
    addition, it provides various visualizations, which can give insights into the
    structure and organization of the input data.
    """

    # Setup folder structure
    fs_log("generating report", verbose)
    os.makedirs(save, exist_ok=True)
    report_file = os.path.join(save, "report.md")
    image_folder = os.path.join(save, "images/")
    os.makedirs(image_folder, exist_ok=True)

    # Generate the plots
    if generate_images:
        fs_log("generating plots (1/5): feature planes", verbose)
        model.plot_feature_planes(save=image_folder + "feature_planes.png", show=False)
        fs_log("generating plots (2/5): som without clusters ", verbose)
        model.plot_som(
            save=image_folder + "som_noclusters.png", show=False, show_mclusters=False
        )
        fs_log("generating plots (3/5): mst without clusters", verbose)
        model.plot_mst(
            save=image_folder + "mst_noclusters.png", show=False, show_mclusters=False
        )
        fs_log("generating plots (4/5): som with clusters", verbose)
        model.plot_som(save=image_folder + "som_wclusters.png", show=False)
        fs_log("generating plots (5/5): mst with clusters", verbose)
        model.plot_mst(save=image_folder + "mst_wclusters.png", show=False)

    report_string = f"""
# :microscope: FlowSOM Analysis Report
This report summarizes the results of a FlowSOM analysis.

Parameter | Value
--- | ---
Total Number of Cells | `{model.data.shape[0]}`
Total Number of Metaclusters | `{model.hcc_param.n_clusters}`
Total Number of Clusters | `{model.som_param.shape[0]*model.som_param.shape[1]}`
Markers Used | `{list(model.data.columns)}`

## Model Parameters

The following parameters were used to train the FlowSOM model:

### Self-Organizing Map (SOM) Parameters

Parameter | Value
--- | ---
Number of Epochs | `{model.som_param.n_epochs}`
Shape of SOM Grid | `{model.som_param.shape}`
Neighbourhood ($\\sigma$) | `{model.som_param.sigma}`
Learning Rate ($\\alpha$) | `{model.som_param.alpha}`
Neighbourhood Function | `{model.som_param.neighbourhood_function}`
Activation Distance | `{model.som_param.activiation_distance}`

### Minimum Spanning Tree (MST) Parameters

Parameter | Value
--- | ---
Distance Metric | `{model.mst_param.distance_metric}`

### Hierarchical Consensus Clustering (HCC) Parameters

Parameter | Value
--- | ---
Number of Clusters | `{model.hcc_param.n_clusters}`
Linkage Method | `{model.hcc_param.linkage_method}`

## Analysis Results

### Visualisations
The following images visualize different aspects of the FlowSOM analysis.

#### Overview

| | no metaclusters | with metaclusters |
-|-|-
Grid | ![SOM no clusters ](images/som_noclusters.png) | ![SOM](images/som_wclusters.png)
MST | ![MST no clusters](images/mst_noclusters.png) | ![MST](images/mst_wclusters.png)

#### Feature Planes

![Feature Planes](images/feature_planes.png)

### Numerical Results
Detailed numerical data, derived from the different parts of the FlowSOM analysis, are
presented in this section.

The first table provides an overview of each metacluster, its constituent clusters, and
the proportion of total cells they contain. It offers insights into the organization of
data at the metacluster level, enabling an understanding of the larger structures
within the dataset

Metacluster | Nr. of Cells | % of Cells | Nr. of Clusters | Clusters
--- | --- | --- | --- | ---
"""

    fs_log("compiling numerical results", verbose)
    # Predictions based on training data
    winners = np.array([model.som.winner(x) for x in model.data.values])
    winners = np.ravel_multi_index(winners.T, model.som.get_weights().shape[:2])

    cell_metaclusters = model.hcc[winners]
    metacluster_counts = np.bincount(cell_metaclusters)

    # Add metacluster numerical data to the report
    for metacluster in np.unique(model.hcc):
        n_clusters = np.count_nonzero(model.hcc == metacluster)
        clusters = np.nonzero(model.hcc == metacluster)[0]
        n_cells = metacluster_counts[metacluster]
        p_cells = n_cells / model.data.shape[0] * 100
        report_string += (
            f"`{metacluster}` | `{n_cells}` | `{p_cells:.2f}%`"
            + f"| `{n_clusters}` | `{list(clusters)}`\n"
        )

    report_string += """

The second table provides a granular look at the distribution of cells among individual
neurons, and the affiliation of neurons to their respective metaclusters. It gives a
detailed view of the microstructures within each metacluster, enhancing the
understanding of data organization at the neuron level.

Neuron | Nr. of Cells | % of Total Cells | Metacluster | % in Metacluster
--- | --- | --- | --- | ---
"""

    # Add cluster numerical data to the report
    nodes, cell_counts = np.unique(winners, return_counts=True)
    for node in nodes:
        num_cells = cell_counts[node]
        percent_cells = num_cells / len(model.data) * 100
        metacluster = model.hcc[node]
        percent_in_metacluster = num_cells / metacluster_counts[metacluster] * 100
        report_string += (
            f"`{node + 1}` | `{num_cells}` | `{percent_cells:.2f}%`"
            + f"| `{metacluster}` | `{percent_in_metacluster:.2f}%`\n"
        )

    with open(report_file, "w") as f:
        f.write(report_string)

    fs_log("report finished", verbose)
