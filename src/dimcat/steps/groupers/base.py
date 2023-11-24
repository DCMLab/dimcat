import logging
from typing import Dict, Hashable, List, MutableMapping, Optional, Sequence

import marshmallow as mm
import pandas as pd
from dimcat.base import is_subclass_of
from dimcat.data.datasets.processed import GroupedDataset
from dimcat.data.resources import Resource
from dimcat.data.resources.dc import DimcatIndex, DimcatResource, PieceIndex
from dimcat.dc_exceptions import GrouperNotSetUpError, ResourceAlreadyTransformed
from dimcat.steps.base import ResourceTransformation
from dimcat.utils import check_name
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Grouper(ResourceTransformation):
    # inherited from PipelineStep:
    _new_dataset_type = GroupedDataset
    _new_resource_type = None  # same as input
    _applicable_to_empty_datasets = True
    # inherited from FeatureProcessingStep:
    _allowed_features = None  # any
    _output_package_name = None  # transform 'features'
    _requires_at_least_one_feature = False

    class Schema(ResourceTransformation.Schema):
        level_name = mm.fields.Str()

    def __init__(self, level_name: str = "group", **kwargs):
        super().__init__(**kwargs)
        self._level_name: str = None
        self.level_name = level_name

    @property
    def level_name(self) -> str:
        return self._level_name

    @level_name.setter
    def level_name(self, level_name: str):
        check_name(level_name)
        self._level_name = level_name

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if self.level_name in resource.get_default_groupby():
            raise ResourceAlreadyTransformed(resource.resource_name, self.name)

    def _post_process_result(
        self,
        result: DimcatResource,
        original_resource: DimcatResource,
    ) -> DimcatResource:
        """Change the default_groupby value of the returned Feature."""
        result.update_default_groupby(self.level_name)
        return result

    def transform_resource(self, resource: DimcatResource) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return pd.concat([resource.df], keys=[self.level_name], names=[self.level_name])


class CorpusGrouper(Grouper):
    def __init__(self, level_name: str = "corpus", **kwargs):
        super().__init__(level_name=level_name, **kwargs)

    def _process_resource(self, resource: Resource) -> Resource:
        """Apply this PipelineStep to a :class:`Resource` and return a copy containing the output(s)."""
        resource = self._pre_process_resource(resource)
        if self.level_name not in resource.get_level_names():
            result = self._make_new_resource(resource)
        else:
            result = resource
        return self._post_process_result(result, resource)


class CustomPieceGrouper(Grouper):
    class Schema(Grouper.Schema):
        grouped_pieces = mm.fields.Nested(DimcatIndex.Schema)

        @mm.pre_load
        def deal_with_dict(self, data, **kwargs):
            if isinstance(data["grouped_pieces"], MutableMapping):
                if "dtype" not in data["grouped_pieces"] or not is_subclass_of(
                    data["grouped_pieces"]["dtype"], DimcatIndex
                ):
                    # dealing with a manually compiled DimcatConfig where grouped_pieces are a grouping dict
                    grouped_pieces = DimcatIndex.from_grouping(data["grouped_pieces"])
                    data["grouped_pieces"] = grouped_pieces.to_dict()
            return data

    @classmethod
    def from_grouping(
        cls,
        piece_groups: Dict[Hashable, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a CustomPieceGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouped_pieces = PieceIndex.from_grouping(
            grouping=piece_groups,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=grouped_pieces.names[0], grouped_pieces=grouped_pieces)

    def __init__(
        self,
        level_name: str = "piece_group",
        grouped_pieces: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)
        self._grouped_pieces: Optional[DimcatIndex] = None
        if grouped_pieces is not None:
            self.grouped_pieces = grouped_pieces

    @property
    def grouped_pieces(self) -> DimcatIndex:
        if self._grouped_pieces is None:
            return DimcatIndex.from_tuples([], (self.level_name, "corpus", "piece"))
        return self._grouped_pieces

    @grouped_pieces.setter
    def grouped_pieces(self, grouped_pieces: DimcatIndex):
        if isinstance(grouped_pieces, pd.Index):
            grouped_pieces = DimcatIndex(grouped_pieces)
        elif isinstance(grouped_pieces, dict):
            raise TypeError(
                f"Use {self.name}.from_dict() to create a {self.name}from a dictionary."
            )
        elif not isinstance(grouped_pieces, DimcatIndex):
            raise TypeError(f"Expected DimcatIndex, got {type(grouped_pieces)}")
        if grouped_pieces.names[-1] != "piece":
            raise ValueError(
                f"Expected last level to to be named 'piece', not {grouped_pieces.names[-1]}"
            )
        self._grouped_pieces = grouped_pieces

    def transform_resource(self, resource: DimcatResource) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return resource.align_with_grouping(self.grouped_pieces)

    def check_resource(self, resource: DimcatResource) -> None:
        if len(self.grouped_pieces) == 0:
            raise GrouperNotSetUpError(self.dtype)
        super().check_resource(resource)
