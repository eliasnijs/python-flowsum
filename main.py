import numpy as np
import readfcs

from src.flowsom import (
    FlowSOM,
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)

# Reading the data
data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data
data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
print(data)

# Training a model
som_param = FlowSOM_SOMParameters(n_epochs=100000, shape=(10, 10), alpha=0.05, sigma=1.0)
mst_param = FlowSOM_MSTParameters()
hcc_param = FlowSOM_HCCParameters(n_clusters=24)
model = FlowSOM(som_param, mst_param, hcc_param).fit(data, verbose=True)

# Visualising Result
model.plot_som(save="data/som_wmclusters.png", show=False, show_mclusters=True)
model.plot_mst(save="data/mst_wmclusters.png", show=False, show_mclusters=True)


def fs_plot_get_marker_avgs(fs):
    winners = np.array([fs.som.winner(x) for x in fs.data.values])
    winners = np.ravel_multi_index(winners.T, fs.som.get_weights().shape[:2])

    indices_by_node = {}
    for i in range(0, fs.som_param.shape[0] * fs.som_param.shape[1]):
        indices = np.where(winners == i)[0]
        indices_by_node[i] = np.mean(fs.data.values[indices], axis=0)

    addition = 0
    for markers in indices_by_node.values():
        minimum = np.min(markers)
        if minimum < addition:
            addition = minimum

    for node, markers in indices_by_node.items():
        indices_by_node[node] += addition
