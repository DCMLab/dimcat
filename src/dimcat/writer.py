"""Writers are pipeline steps that write (potentially processed) data to disk."""
import os

import pandas as pd
from ms3 import resolve_dir, write_tsv

from .data import Corpus, Data
from .pipeline import PipelineStep
from .utils import make_suffix


class TSVWriter(PipelineStep):
    """This writer iterates through the processed data (pandas objects) and writes each to a TSV
    file with a file name individually constructed based on the applied pipeline steps.
    """

    def __init__(self, directory, prefix=None, index=True, round=None, fillna=None):
        """This pipeline step writes the processed data for each group to a TSV file.

        Parameters
        ----------
        directory : :obj:`str`
            Where to store the TSV files (existing files will be overwritten).
        prefix : :obj:`str`, optional
            If you pass a prefix it will be prepended to the filenames.
        index : :obj:`bool`, optional
            By default, the pandas MultiIndex is included as the first columns in the TSV file.
            Pass False to omit the index.
        round : :obj:`int`, optional
            If you want to round floats before writing, pass the number of decimal digits.
        fillna = scalar, :obj:`dict`, :obj:`pandas.Series`, or :obj:`pandas.DataFrame`, optional
            If you want to fill empty fields before writing TSVs, pass a ``value`` argument for
            :py:meth:`pandas.DataFrame.fillna`.
        """
        resolved_path = resolve_dir(directory)
        try:
            assert os.path.isdir(resolved_path), f"Path {directory} not found."
        except TypeError as e:
            raise TypeError(f"'{directory}' could not be interpreted as path. {e}")
        self.directory = resolved_path
        self.prefix = prefix
        self.index = index
        self.round = round
        self.fillna = fillna

    def make_filenames(self, data: Corpus) -> dict:
        """Returns a {group -> filename} dict."""
        result = {}
        pipeline_steps = [ps.filename_factory() for ps in reversed(data.pipeline_steps)]
        for group in data.indices.keys():
            name_components = [] if self.prefix is None else [self.prefix]
            if group == ():
                corpora = data.data.keys()
                if len(corpora) == 1:
                    name_components.append(corpora[0])
                else:
                    name_components.append("all")
            else:
                if isinstance(group, str):
                    name_components.append(group)
                else:
                    name_components.append(make_suffix(list(group)))
            name_components.extend(pipeline_steps)
            result[group] = "-".join(c for c in name_components if c != "")
        return result

    def process_data(self, data: Data) -> Data:
        filenames = self.make_filenames(data)
        for group, df in data.iter():
            tsv_name = filenames[group] + ".tsv"
            tsv_path = os.path.join(self.directory, tsv_name)
            df = pd.DataFrame(df)
            if self.round is not None:
                df = df.round(self.round)
            if self.fillna is not None:
                df = df.fillna(value=self.fillna)
            write_tsv(df, tsv_path, index=self.index)
            print(f"Wrote {tsv_path}")
        return data
