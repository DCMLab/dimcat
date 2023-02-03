"""Writers are pipeline steps that write (potentially processed) data to disk."""
import os
from typing import Any, List, Optional

import pandas as pd
from ms3 import resolve_dir, write_tsv

from .data import Dataset, GroupedData
from .pipeline import PipelineStep
from .utils import make_suffix


class TSVWriter(PipelineStep):
    """This writer iterates through the processed data (pandas objects) and writes each to a TSV
    file with a file name individually constructed based on the applied pipeline steps.
    """

    def __init__(
        self,
        directory: str,
        prefix: Optional[str] = None,
        index: bool = True,
        round: Optional[int] = None,
        fillna: Optional[Any] = None,
    ):
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

    def make_group_filenames(self, data: GroupedData) -> dict:
        """Returns a {group -> filename} dict."""
        result = {}
        pipeline_steps = [ps.filename_factory() for ps in reversed(data.pipeline_steps)]
        for group, _ in data.iter_grouped_indices():
            name_components: List[str] = [] if self.prefix is None else [self.prefix]
            if group == ():
                corpora = data.data.keys()
                if len(corpora) == 1:
                    name_components.append(corpora[0])
                else:
                    name_components.append("all")
            else:
                name_components.append(make_suffix(list(group)))
            name_components.extend(pipeline_steps)
            result[group] = "-".join(c for c in name_components if c != "")
        return result

    def make_index_filenames(self, data: Dataset) -> dict:
        """Returns a {group -> filename} dict."""
        result = {}
        pipeline_steps = [ps.filename_factory() for ps in reversed(data.pipeline_steps)]
        for index in data.indices:
            name_components: List[str] = [] if self.prefix is None else [self.prefix]
            name_components.append(make_suffix(list(index)))
            name_components.extend(pipeline_steps)
            result[index] = "-".join(c for c in name_components if c != "")
        return result

    def process_data(self, data: Dataset) -> Dataset:
        if isinstance(data, GroupedData):
            filenames = self.make_group_filenames(data)
            iterator = data.iter_grouped_results()
        else:
            filenames = self.make_index_filenames(data)
            iterator = data.iter_results()
        for key, df in iterator:
            tsv_name = filenames[key] + ".tsv"
            tsv_path = os.path.join(self.directory, tsv_name)
            df = pd.DataFrame(df)
            if self.fillna is not None:
                df = df.fillna(value=self.fillna)
            if self.round is not None:
                df = df.round(self.round)
            write_tsv(df, tsv_path, index=self.index)
            print(f"Wrote {tsv_path}")
        return data
