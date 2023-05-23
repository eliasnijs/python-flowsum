import numpy as np
import readfcs

from flowsom import FlowSOM
from flowsom_h import FlowSOM_SOMParameters

# ------------------------------------------------

fcs = readfcs.ReadFCS("resources/Levine_13dim.fcs")

data = fcs.data
data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# ------------------------------------------------

som_param = FlowSOM_SOMParameters(n_epochs=100000, shape=(10, 10))

flowsom: FlowSOM = FlowSOM(som_param)
flowsom.fit(data, verbose=True)

flowsom.plot_mst()
