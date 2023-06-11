import readfcs

from src.flowsom import (
    FlowSOM,
    FlowSOM_HCCParameters,
    FlowSOM_MSTParameters,
    FlowSOM_SOMParameters,
)

# Reading the data
data = readfcs.ReadFCS("resources/Levine_32dim.fcs").data
data = data.drop("label", axis=1)

# Training a model
som_param = FlowSOM_SOMParameters(n_epochs=1, shape=(10, 10), alpha=0.05, sigma=0.67)
mst_param = FlowSOM_MSTParameters()
hcc_param = FlowSOM_HCCParameters(n_clusters=24)
model = FlowSOM(som_param, mst_param, hcc_param).fit(data, verbose=True)

# Generating report
model.report(save="data/report/", verbose=True)
