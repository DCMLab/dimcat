from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import marshmallow as mm
from dimcat.base import FriendlyEnumField, ListOfStringsField
from dimcat.data.resources import Feature, FeatureName
from dimcat.data.resources.base import DR, SomeSeries
from dimcat.data.resources.dc import FeatureSpecs, UnitOfAnalysis
from dimcat.data.resources.features import PhraseComponentName
from dimcat.data.resources.results import PhraseData, PhraseDataFormat
from dimcat.data.resources.utils import (
    drop_duplicated_ultima_rows,
    subselect_multiindex_from_df,
    transform_phrase_data,
)
from dimcat.steps.analyzers.base import Analyzer, DispatchStrategy


class PhraseDataAnalyzer(Analyzer):
    _allowed_features = (
        FeatureName.PhraseAnnotations,
        FeatureName.PhraseComponents,
        FeatureName.PhraseLabels,
    )
    _default_dimension_column = "duration_qb"
    _new_resource_type = PhraseData
    _output_package_name = "results"
    _requires_at_least_one_feature = True

    class Schema(Analyzer.Schema):
        columns = ListOfStringsField(metadata=dict(expose=False))
        components = ListOfStringsField(metadata=dict(expose=False))
        query = mm.fields.Str(allow_none=True, metadata=dict(expose=False))
        reverse = mm.fields.Bool(metadata=dict(expose=False))
        level_name = mm.fields.Str(metadata=dict(expose=False))
        format = FriendlyEnumField(PhraseDataFormat, metadata=dict(expose=False))
        drop_levels = mm.fields.Raw(metadata=dict(expose=False))
        drop_duplicated_ultima_rows = mm.fields.Bool(
            allow_none=True, metadata=dict(expose=False)
        )

    def __init__(
        self,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        columns: str | List[str] = "label",
        components: PhraseComponentName
        | Literal["phrase"]
        | Iterable[PhraseComponentName] = "body",
        query: Optional[str] = None,
        reverse: bool = False,
        level_name: str = "i",
        format: PhraseDataFormat = PhraseDataFormat.LONG,
        drop_levels: bool | int | str | Iterable[int | str] = False,
        drop_duplicated_ultima_rows: bool = False,
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        dimension_column: str = None,
    ):
        """

        Args:
            features:
                The Feature objects you want this Analyzer to process. If not specified, it will try to process all
                features present in a given Dataset's Outputs catalog.
            columns:
                Column(s) to include in the result.
            components:
                Which of the four phrase components to include, ∈ {'ante', 'body', 'codetta', 'post'}.
                For convenience, the string 'phrase' is also accepted, which is equivalent to ["body", "codetta"] and
                ``drop_duplicated_ultima_rows=True``.
            query:
                A convenient way to include only those phrases in the result that match the criteria
                formulated in the string query. A query is a string and generally takes the form
                "<column_name> <operator> <value>". Several criteria can be combined using boolean
                operators, e.g. "localkey_mode == 'major' & label.str.contains('/')". This option
                is particularly interesting when used on :class:`PhraseLabels` because it enables
                queries based on the properties of phrases such as
                "body_n_modulations == 0 & end_label.str.contains('IAC')".  For the columns
                containing tuples, you can used a special function to filter those rows that
                contain any of the specified values:
                "@tuple_contains(body_chords, 'V(94)', 'V(9)', 'V(4)')".
            reverse:
                Pass True to reverse the order of harmonies so that each phrase's last label comes
                first.
            level_name:
                Name of the index level representing the individual integer range for each phrase, starting at 0.
                This level replaces the original 'i' level which allows for tracing back each chord, because it allows
                for displaying the phrases in WIDE format.
            format: Can be LONG (default) or WIDE.
            drop_levels:
                Can be a boolean or any level specifier accepted by :meth:`pandas.MultiIndex.droplevel()`.
                If False (default), all levels are retained. If True, only the phrase_id level and
                the ``level_name`` are retained. In all other cases, the indicated (string or
                integer) value(s) must be valid and cause one of the index levels to be dropped.
                ``level_name`` cannot be dropped. Dropping 'phrase_id' will likely lead to an
                exception if a :class:`PhraseData` object will be displayed in WIDE format.
            drop_duplicated_ultima_rows:
                The default behaviour (when None), depends on the value of ``components``: If you set
                ``components='phrase'``, this setting defaults to True, otherwise to False; where
                False corresponds to the default where  each phrase body ends on a duplicate of the
                phrase's ultima label, with zero-duration, enabling the creation of PhraseData
                containing only phrase bodies (i.e., ``components='body'``), without losing information
                about the ultima label. When analyzing entire phrases, however, these duplicate
                rows may be unwanted and can be dropped by setting this option to True.
            strategy: Currently, only the default strategy GROUPBY_APPLY is implemented.
            smallest_unit:
                The smallest unit to consider for analysis. Defaults to SLICE, meaning that slice segments are analyzed
                if a slicer has been previously applied, piece units otherwise. The results for larger units can always
                be retrospectively retrieved by using :meth:`Result.combine_results()`, but not the other way around.
                Use this setting to reduce compute time by setting it to PIECE, CORPUS_GROUP, or GROUP where the latter
                uses the default groupby if a grouper has been previously applied, or the entire dataset, otherwise.
            dimension_column:
                Name of the column containing some dimension, e.g. to be interpreted as quantity (durations, counts,
                etc.) or as color.
        """
        super().__init__(
            features=features,
            strategy=strategy,
            smallest_unit=smallest_unit,
            dimension_column=dimension_column,
        )
        self._columns = None
        self._components = None
        self._drop_levels = None
        self._format = None
        self.drop_levels: bool | int | str | Iterable[int | str] = drop_levels
        self.drop_duplicated_ultima_rows: bool = drop_duplicated_ultima_rows
        self.query: str = query
        self.reverse: bool = reverse
        self.level_name: str = level_name
        self.columns = columns
        self.components = components
        self.format = format

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @columns.setter
    def columns(self, columns: str | List[str]):
        if columns is None:
            raise ValueError("columns cannot be None")
        if isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)
        self._columns = columns

    @property
    def components(self) -> List[PhraseComponentName]:
        return list(self._components)

    @components.setter
    def components(
        self,
        components: PhraseComponentName
        | Literal["phrase"]
        | Iterable[PhraseComponentName],
    ):
        if components is None:
            raise ValueError("components cannot be None")
        if isinstance(components, str):
            components = [components]
        else:
            components = list(components)
        if any(c.lower() == "phrase" for c in components):
            assert len(components) == 1, (
                "If you use the convenience value 'phrase', it must be the "
                "only component and will be converted to ['body', 'codetta']"
            )
            components = ["body", "codetta"]
            if self.drop_duplicated_ultima_rows is None:
                self.drop_duplicated_ultima_rows = True
        else:
            components = [PhraseComponentName(c).value for c in components]
        self._components = components

    @property
    def format(self) -> PhraseDataFormat:
        return self._format

    @format.setter
    def format(self, format: PhraseDataFormat):
        self._format = PhraseDataFormat(format)

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        phrase_df = feature.phrase_df
        if self.drop_duplicated_ultima_rows:
            phrase_df = drop_duplicated_ultima_rows(phrase_df)
        if self.query:
            if feature.name == "PhraseAnnotations":
                phrase_df = phrase_df.query(self.query)
            else:
                # for PhraseComponents and PhraseLabels, the filtering is performed on their respective feature df,
                # then the phrase_df (which corresponds to a PhraseAnnotations dataframe) is filtered based on the
                # result
                filtered_df = feature.df.query(self.query)
                # idx = filtered_df.index
                # mask = make_boolean_mask_from_set_of_tuples(
                #     phrase_df.index, set(idx), idx.names
                # )
                # phrase_df = phrase_df[mask]
                phrase_df = subselect_multiindex_from_df(phrase_df, filtered_df.index)
        phrase_data = transform_phrase_data(
            phrase_df=phrase_df,
            columns=self.columns,
            components=self.components,
            drop_levels=self.drop_levels,
            reverse=self.reverse,
            level_name=self.level_name,
        )
        # if isinstance(columns, str):
        #     value_column = columns
        #     formatted_column = None
        # else:
        #     value_column = columns[0]
        #     formatted_column = columns[1:]
        # default_groupby = self.default_groupby + ["phrase_id"]
        # df_format = PhraseDataFormat.WIDE if wide_format else PhraseDataFormat.LONG
        return phrase_data

    def resource_name_factory(self, resource: DR) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.phrase_data"
