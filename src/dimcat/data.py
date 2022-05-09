"""Class hierarchy for data types."""
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from ms3 import Parse


class Data(ABC):
    """
    Subclasses are the dtypes that this library uses. Every PipelineStep accepts one or several
    dtypes.

    The initializer can set parameters influencing how the contained data will look and is able
    to create an object from an existing Data object to enable type conversion.
    """

    def __init__(self, data=None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        self._data = None
        """Protected attribute for storing and internally accessing the loaded data."""

        self.indices = {}
        """Indices for accessing individual pieces of data and associated metadata."""

        self.processed = {}
        """Subclasses store processed data here."""

        self.applied_pipeline = []
        """The sequence of applied PipelineSteps that has led to the current state."""

        self.index_levels = {
            "indices": ["corpus", "fname"],
            "groups": [],
            "slices": [],
            "processed": [],
        }
        """Keeps track of index level names. Also used for automatic naming of output files."""

        self.sliced = {}
        """Dict for sliced data facets."""

        self.slice_info = {}
        """Dict holding metadata of slices (e.g. the localkey of a segment)."""

        self.group2pandas = "group2dataframe"

        if data is not None:
            self.data = data

    @property
    @abstractmethod
    def data(self):
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data_object):
        """Implement setting the __data field after performing type check."""
        if data_object is not None:
            raise NotImplementedError
        self._data = data_object

    def copy(self):
        return self.__class__(data=self)

    def get(self):
        """Get all processed data at once."""
        return self.processed

    def iter(self):
        """Iterate through processed data."""
        for tup in self.processed.items():
            yield tup

    @abstractmethod
    def iter_facet(self, what):
        """Iterate through (potentially grouped) pieces of data."""
        for index_group in self.iter_groups():
            yield [self.get_item(index) for index in index_group]

    @abstractmethod
    def get_item(self, index):
        """Get an individual piece of data."""

    def iter_groups(self):
        """Iterate through groups of indices.

        Yields
        -------
        :obj:`tuple` of :obj:`str`
            A tuple of keys reflecting the group hierarchy
        :obj:`list` of :obj:`tuple`
            A list of IDs belonging to the same group.
        """
        if len(self.indices) == 0:
            raise ValueError("No data has been loaded.")
        for group in self.indices.items():
            yield group

    def track_pipeline(
        self,
        pipeline_step,
        group2pandas=None,
        indices=None,
        processed=None,
        grouper=None,
        slicer=None,
    ):
        """Store processed data and keep track of the applied pipeline_steps."""
        self.applied_pipeline.append(pipeline_step)
        if processed is not None:
            if isinstance(processed, str):
                processed = [processed]
            self.index_levels["processed"] = processed
        if indices is not None:
            if isinstance(indices, str):
                indices = [indices]
            self.index_levels["indices"] = indices
        if group2pandas is not None:
            self.group2pandas = group2pandas
        if grouper is not None:
            self.index_levels["groups"] = self.index_levels["groups"] + [grouper]
        if slicer is not None:
            self.index_levels["slices"].append(slicer)

    @abstractmethod
    def load(self):
        """Load data into memory."""

    def group_of_values2series(self, group_dict):
        """Turns an {ix -> result} into a Series or DataFrame."""
        series = pd.Series(group_dict, name=self.index_levels["processed"][0])
        index_level_names = self.index_levels["indices"]
        try:
            if series.index.nlevels == 1:
                index_level_names = index_level_names[0]
            series.index.rename(index_level_names, inplace=True)
        except (TypeError, ValueError):
            print(series.index)
            print(f"current: {series.index.names}, new: {index_level_names}")
            print(self.index_levels)
            raise
        return series

    def group_of_series2series(self, group_dict):
        """Turns an {ix -> result} into a Series or DataFrame."""
        series = pd.concat(group_dict.values(), keys=group_dict.keys())
        index_level_names = (
            self.index_levels["indices"] + self.index_levels["processed"]
        )
        try:
            series.index.rename(index_level_names, inplace=True)
        except (TypeError, ValueError):
            print(series.index)
            print(f"current: {series.index.names}, new: {index_level_names}")
            print(self.index_levels)
            raise
        return series

    def group2dataframe(self, group_dict):
        try:
            df = pd.concat(group_dict.values(), keys=group_dict.keys())
        except TypeError:
            print(group_dict)
            raise
        index_level_names = (
            self.index_levels["indices"] + self.index_levels["processed"]
        )
        try:
            df.index.rename(index_level_names, inplace=True)
        except (TypeError, ValueError):
            print(df.index)
            print(f"current: {df.index.names}, new: {index_level_names}")
            print(self.index_levels)
            raise
        return df

    def group2dataframe_unstacked(self, group_dict):
        return self.group2dataframe(group_dict).unstack()


def remove_corpus_from_ids(result):
    """Called when group contains corpus and removes redundant repetition from indices."""
    if isinstance(result, dict):
        without_corpus = {}
        for key, v in result.items():
            if isinstance(key[0], str):
                without_corpus[key[1:]] = v
            else:
                new_key = tuple(k[1:] for k in key)
                without_corpus[new_key] = v
        return without_corpus
    print(result)
    return result.droplevel(0)


class Corpus(Data):
    """Essentially a wrapper for a ms3.Parse object."""

    def __init__(self, data=None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object into a Corpus if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        super().__init__()
        self.pieces = {}
        """
        IDs and metadata of those pieces that have not been filtered out.::

            {(corpus, fname) ->
                {
                 'metadata' -> {key->value},
                 'matched_files' -> [namedtuple]
                }
            }

        """
        if data is None:
            self._data = Parse()
        else:
            self.data = data
        if len(kwargs) > 0:
            self.load(**kwargs)

    @property
    def data(self):
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    def data(self, data_object):
        """Check if the assigned object is suitable for conversion."""
        if not isinstance(data_object, Corpus):
            raise TypeError(f"{type(data_object)} could not be converted to a Corpus.")
        self._data = data_object._data
        self.pieces = dict(data_object.pieces)
        self.indices = dict(data_object.indices)
        self.processed = dict(data_object.processed)
        self.sliced = dict(data_object.sliced)
        self.slice_info = dict(data_object.slice_info)
        self.applied_pipeline = list(self.applied_pipeline)
        self.index_levels = dict(data_object.index_levels)
        self.group2pandas = data_object.group2pandas

    def get(self, as_pandas=False):
        if len(self.processed) == 0:
            print("No data has been processed so far.")
            return
        results = {group: result for group, result in self.iter(as_pandas=as_pandas)}
        if as_pandas:
            if len(results) == 1 and () in results:
                results = pd.concat(results.values())
            else:
                try:
                    results = pd.concat(
                        results.values(),
                        keys=results.keys(),
                        names=self.index_levels["groups"],
                    )
                except ValueError:
                    print(self.index_levels["groups"])
                    print(results.keys())
                    raise
        return results

    def convert_group2pandas(self, result_dict):
        converters = {
            "group_of_values2series": self.group_of_values2series,
            "group_of_series2series": self.group_of_series2series,
            "group2dataframe": self.group2dataframe,
            "group2dataframe_unstacked": self.group2dataframe_unstacked,
        }
        converter = converters[self.group2pandas]
        return converter(result_dict)

    def iter(self, as_pandas=False):
        """Iterate through processed data."""
        for group, result in self.processed.items():
            if as_pandas:
                yield group, self.convert_group2pandas(result)
            else:
                yield group, result

    def load(
        self,
        directory: List[str] = None,
        parse_tsv: bool = True,
        parse_scores: bool = False,
        ms: str = None,
    ):
        """
        Load and parse all of the desired raw data and metadata.

        Parameters
        ----------
        directory : str
            The path to all the data to load.
        parse_tsv : :obj:`bool`, optional
            By default, all detected TSV files are parsed and their type is inferred.
            Pass False to prevent parsing TSV files.
        parse_scores : :obj:`bool`, optional
            By default, detected scores files are not parsed to save time.
            Call super().__init__(parse_scores=True) if an analyzer needs
            to access parsed MuseScore XML.
        ms : :obj:`str`, optional
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse
            musicXML, MuseScore 2, and other formats by temporarily converting them. If you're
            using the standard path, you may try 'auto', or 'win' for Windows, 'mac' for MacOS,
            or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore
            executable is detected, all convertible files are automatically selected, otherwise
            only those that can be parsed without conversion.
        """
        if ms is not None:
            self.data.ms = ms
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.data.add_dir(
                    directory=d,
                )
        if parse_tsv:
            self.data.parse_tsv()
        if parse_scores:
            self.data.parse_mscx()
        if len(self.data._parsed_tsv) == 0 and len(self.data._parsed_mscx) == 0:
            print("No files have been parsed for analysis.")
        else:
            self.get_indices()

    def get_indices(self):
        """Fills self.pieces with metadata and IDs for all loaded data."""
        self.pieces = {}
        self.indices = {}
        # self.group_labels = {}
        for key in self.data.keys():
            view = self.data[key]
            for metadata, (fname, matched_files) in zip(
                view.metadata().to_dict(orient="records"),
                view.detect_ids_by_fname(parsed_only=True).items(),
            ):
                assert (
                    fname == metadata["fnames"]
                ), f"metadata() and pieces() do not correspond for key {key}, fname {fname}."
                piece_info = {}
                piece_info["metadata"] = metadata
                piece_info["matched_files"] = matched_files
                ID = (key, fname)
                self.pieces[ID] = piece_info
        self.indices[()] = list(self.pieces.keys())

    def iter_facet(self, what, unfold=False, concatenate=False):
        """Iterate through groups of potentially sliced facet DataFrames.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.
        concatenate : :obj:`bool`, optional
            By default, the returned list contains one DataFrame per ID contained in the group.
            Pass True to instead concatenate the DataFrames. Then, the list will contain only
            one DataFrame the components of which can be distinguished using its MultiIndex.

        Yields
        ------
        :obj:`tuple`
            Group identifier
        :obj:`dict` or :obj:`pandas.DataFrame`
            Default: {ID -> DataFrame}.
            If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
        """
        for group, index_group in self.iter_groups():
            result = {}
            for index in index_group:
                df = self.get_item(index, what, unfold)
                if df.shape[0] == 0:
                    continue
                result[index] = df
            if concatenate:
                if len(result) == 1:
                    # workaround necessary because of nasty "cannot handle overlapping indices;
                    # use IntervalIndex.get_indexer_non_unique" error
                    for id, df in result.items():
                        pass
                    new_index = [id + (i,) for i in df.index]
                    new_index = pd.MultiIndex.from_tuples(new_index)
                    df.index = new_index
                else:
                    result = pd.concat(result.values(), keys=result.keys())
                    result = {tuple(index_group): result}

            yield group, result

    def get_item(self, index, what, unfold=False, multiindex=False):
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Parameters
        ----------
        index : :obj:`tuple`
            (corpus, fname) or (corpus, fname, interval)
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.
        multiindex : :obj:`bool`, optional
            By default, has one level, which is a :obj:``pandas.IntervalIndex``. Pass True to
            prepend ``index`` as an additional index level.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        n_index_elements = len(index)
        if n_index_elements == 2:
            corpus, fname = index
            try:
                df = self.data[corpus][fname].get_dataframe(
                    what, unfold, interval_index=True
                )
            except FileNotFoundError:
                print(f"No {what} available for {index}. Returning empty DataFrame.")
                df = pd.DataFrame()
            return df
        elif n_index_elements == 3:
            if what in self.sliced:
                df = self.sliced[what][index]
            else:
                # corpus, fname, interval = index
                raise NotImplementedError(f"'{what}' would have to be sliced first.")
        else:
            raise NotImplementedError(
                f"'{index}': Indices can currently include 2 or 3 elements."
            )
        if multiindex:
            df = pd.concat([df], keys=[index], names=["corpus", "fname"])
        return df
