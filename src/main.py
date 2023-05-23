import numpy as np
import readfcs

from flowsom import FlowSOM
from flowsom_h import FlowSOM_HCCParameters, FlowSOM_SOMParameters

# ------------------------------------------------

fcs = readfcs.ReadFCS("resources/Levine_13dim.fcs")

data = fcs.data
data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# ------------------------------------------------

som_param = FlowSOM_SOMParameters(n_epochs=10000, shape=(10, 10))
hcc_param = FlowSOM_HCCParameters(n_bootstrap=50, n_clusters=12)

flowsom: FlowSOM = FlowSOM(som_param=som_param, hcc_param=hcc_param)
flowsom.fit(data, verbose=True)

flowsom.plot_mst()
