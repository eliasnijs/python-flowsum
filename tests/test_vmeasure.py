from sklearn.metrics.cluster import v_measure_score
from flowio import FlowData
import numpy as np

from src.flowsom import FlowSOM

def read_labelled_fcs(path):
     fcs_data = FlowData(path)
     npy_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
     y = npy_data[:, -1]
     mask = ~np.isnan(y)
     X = npy_data[mask, :-1]
     y = npy_data[mask, -1]
     if 0 not in y:
         y = y - 1
     y = y.astype(np.int32)
     return X, y

def test_vmeasure():
    min_vmeasure = 0.75
    path = "resources/Levine_13dim.fcs"

    # read in fcs file
    X, y = read_labelled_fcs(path)

    n_clusters = np.unique(y).shape[0]

    estimator = FlowSOM(n_clusters=n_clusters)
    estimator.fit(X, verbose=True)
    y_pred = estimator.predict(X)

    v_measure = v_measure_score(y, y_pred)
    print(v_measure)
    assert v_measure > min_vmeasure, f"v_measure {v_measure} is not higher than {min_vmeasure}"
