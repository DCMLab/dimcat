import os

import pandas as pd
from ms3 import write_tsv

from .data import Data
from .pipeline import PipelineStep


class TSVwriter(PipelineStep):
    def __init__(self, directory, suffix="", index=True):
        self.directory = directory
        self.suffix = suffix
        self.index = index

    def process_data(self, data: Data) -> Data:
        for group, df in data.iter(as_pandas=True):
            if group == ():
                group = "global"
            else:
                group = "-".join(group)
            tsv_name = f"{group}_{self.suffix}.tsv"
            tsv_path = os.path.join(self.directory, tsv_name)
            write_tsv(pd.DataFrame(df), tsv_path, index=self.index)
            print(f"Wrote {tsv_path}")
        return data
