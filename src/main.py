import numpy as np
import readfcs

from flowsom import FlowSOM
from fs_dataclasses import FlowSOM_HCCParameters, FlowSOM_SOMParameters

# ------------------------------------------------

data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data

data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# ------------------------------------------------
som_param = FlowSOM_SOMParameters(n_epochs=100000, shape=(10, 10), alpha=0.5)
hcc_param = FlowSOM_HCCParameters(n_bootstrap=10, n_clusters=12)

flowsom: FlowSOM = FlowSOM(som_param=som_param, hcc_param=hcc_param)
flowsom.fit(data, verbose=True)

flowsom.plot_som()
flowsom.plot_mst()

print("finished")
