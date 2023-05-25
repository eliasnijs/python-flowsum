import numpy as np
import readfcs

from flowsom import FlowSOM
from fs_dataclasses import (
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)

# Reading the data
data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data

data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

print(data)

# Traning a model
som_param = FlowSOM_SOMParameters(n_epochs=1000, shape=(10, 10), alpha=0.5, sigma=1.0)
mst_param = FlowSOM_MSTParameters()
hcc_param = FlowSOM_HCCParameters(n_bootstrap=10, n_clusters=12)

model = FlowSOM(som_param=som_param, hcc_param=hcc_param).fit(data, verbose=True)
