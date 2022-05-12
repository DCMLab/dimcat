import os

import pandas as pd
from ms3 import write_tsv

from .data import Data
from .pipeline import PipelineStep
from .utils import make_suffix


class TSVwriter(PipelineStep):
    def __init__(self, directory, prefix=None, index=True):
        self.directory = directory
        self.prefix = prefix
        self.index = index

    def process_data(self, data: Data) -> Data:
        for group, df in data.iter():
            if self.prefix is None:
                name_components = [make_suffix(data.index_levels["processed"])]
            else:
                name_components = [self.prefix]
            if group == ():
                name_components.append("global")
            else:
                name_components.append(make_suffix(list(group)))
            name_components.extend(
                [
                    make_suffix(names)
                    for level, names in data.index_levels.items()
                    if level not in ("processed", "groups", "indices")
                ]
            )
            tsv_name = "-".join(c for c in name_components if c != "") + ".tsv"
            tsv_path = os.path.join(self.directory, tsv_name)
            write_tsv(pd.DataFrame(df), tsv_path, index=self.index)
            print(f"Wrote {tsv_path}")
        return data
