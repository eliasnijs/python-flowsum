import numpy as np
import readfcs

from src.flowsom import FlowSOM


def test_as_df():
    # Reading the data
    data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data
    data = data.drop("label", axis=1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Traning a model
    model = FlowSOM().fit(data, verbose=True)

    # Testing data
    df = model.as_df(lazy=False)
    assert data.shape[0] == df.shape[0], "Number of rows don't match"
    assert data.shape[1] == df.shape[1], "Number of columns don't match"
    assert np.allclose(data.values, df.values), "Data doens't match"


def test_as_df_lazy():
    # Reading the data
    data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data
    data = data.drop("label", axis=1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Traning a model
    model = FlowSOM().fit(data, verbose=True)

    # Testing data
    df = model.as_df(lazy=True)
    df = df.compute()
    assert data.shape[0] == df.shape[0], "Number of rows don't match"
    assert data.shape[1] == df.shape[1], "Number of columns don't match"
    assert np.allclose(data.values, df.values), "Data doens't match"


def test_as_adata():
    # Reading the data
    data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data
    data = data.drop("label", axis=1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Traning a model
    model = FlowSOM().fit(data, verbose=True)

    # Testing data
    adata = model.as_adata(lazy=False)
    values = adata.X
    assert data.shape[0] == values.shape[0], "Number of rows don't match"
    assert data.shape[1] == values.shape[1], "Number of columns don't match"
    assert np.allclose(data.values, values), "Data doens't match"


def test_as_adata_lazy():
    # Reading the data
    data = readfcs.ReadFCS("resources/Levine_13dim.fcs").data
    data = data.drop("label", axis=1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Traning a model
    model = FlowSOM().fit(data, verbose=True)

    # Testing data
    adata = model.as_adata()
    values = adata.X.compute()
    assert data.shape[0] == values.shape[0], "Number of rows don't match"
    assert data.shape[1] == values.shape[1], "Number of columns don't match"
    assert np.allclose(data.values, values), "Data doens't match"
