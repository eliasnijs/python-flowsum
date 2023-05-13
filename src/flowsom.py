import os

import dask.dataframe as dd
from FlowCytometryTools import FCMeasurement


class FlowSOM:
    def __init__(self, filepaths, n_metaclusters):
        self._log("New FlowSOM analysis:")
        self._log(f"    filespaths:       {filepaths}")
        self._log(f"    n_metaclusters:   {n_metaclusters}")

        for filepath in filepaths:
            r = self.readfcs(filepath)
            print(r.head())

        self.filepaths = filepaths
        self.n_metaclusters = n_metaclusters

    def readfcs(self, file):
        data = FCMeasurement(ID=os.path.basename(file), datafile=file)
        ddf = dd.from_pandas(data.data, npartitions=1)
        return ddf

    def _log(_, fstring):
        print("[FlowSOM]: ", end="")
        print(fstring)
