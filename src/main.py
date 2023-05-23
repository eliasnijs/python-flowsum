import fcsparser
import numpy as np

from flowsom import FlowSOM
from flowsom_h import FlowSOM_SOMParameters

# ------------------------------------------------

(metadata, data) = fcsparser.parse("resources/Levine_13dim.fcs")
data = data.drop("label", axis=1)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# ------------------------------------------------

som_param = FlowSOM_SOMParameters(n_epochs=100000, shape=(10, 10))

flowsom: FlowSOM = FlowSOM(som_param)
flowsom.fit(data)

flowsom.plot_som()
