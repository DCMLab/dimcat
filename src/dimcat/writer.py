"""Writers are pipeline steps that write (potentially processed) data to disk."""
import os

import pandas as pd
from ms3 import write_tsv

from .data import Data
from .pipeline import PipelineStep
from .utils import make_suffix


class TSVWriter(PipelineStep):
    """This writer iterates through the processed data (pandas objects) and writes each to a TSV
    file with a file name individually constructed based on the applied pipeline steps.
    """

    def __init__(self, directory, prefix=None, index=True):
        """This pipeline step writes the processed data for each group to a TSV file.

        Parameters
        ----------
        directory : :obj:`str`
            Where to store the TSV files (existing files will be overwritten).
        prefix : :obj:`str`, optional
            If you pass a prefix it replaces the name of the applied analyzers in the filenames.
        index : :obj:`bool`, optional
            By default, the pandas MultiIndex is included as the first columns in the TSV file.
            Pass False to omit the index.
        """
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
