from typing import ClassVar, Optional

import marshmallow as mm
import pandas as pd
from dimcat import Dataset
from dimcat.data.resources import Feature, FeatureName
from dimcat.data.resources.dc import SliceIntervals
from dimcat.dc_exceptions import SlicerNotSetUpError
from dimcat.steps.slicers.base import Slicer


class AdjacencyGroupSlicer(Slicer):
    """This slicer and its subclasses slices resources by adjacency groups, that is, segments where a particular
    column (or combination thereof) has the same value over all rows."""

    _adjacency_group_column_name: ClassVar[Optional[str]] = None
    """Optional class variable that specifies the name of the column that contains the adjacency group.
    Defaults to each row, i.e., no extra grouping.
    """
    _required_feature: ClassVar[FeatureName]
    """Required for AdjacencyGroupSlicers, the type of Feature that needs to be present in a dataset to fit this
    slicer. """

    class Schema(Slicer.Schema):
        slice_intervals = mm.fields.Nested(SliceIntervals.Schema)

    def __init__(
        self,
        level_name: str = "adjacency_group",
        slice_intervals: Optional[SliceIntervals] = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)
        self._slice_intervals: Optional[SliceIntervals] = None
        if slice_intervals is not None:
            self.slice_intervals = slice_intervals

    @property
    def required_feature(self) -> FeatureName:
        if not self._required_feature:
            raise NotImplementedError(f"Please use a subclass of {self.name}.")
        return self._required_feature

    @property
    def slice_intervals(self) -> Optional[SliceIntervals]:
        return self._slice_intervals

    @slice_intervals.setter
    def slice_intervals(self, slice_intervals: SliceIntervals | pd.MultiIndex):
        if isinstance(slice_intervals, pd.MultiIndex):
            slice_intervals = SliceIntervals.from_index(slice_intervals)
        elif not isinstance(slice_intervals, SliceIntervals):
            raise TypeError(
                f"Expected SliceIntervals or pd.MultiIndex, got {type(slice_intervals)}"
            )
        self._slice_intervals = slice_intervals

    def fit_to_dataset(self, dataset: Dataset) -> None:
        """Set the slice intervals to the intervals provided by the relevant feature."""
        feature = dataset.get_feature(self.required_feature)
        self.slice_intervals = feature.get_slice_intervals(level_name=self.level_name)

    def get_slice_intervals(self, resource: Feature) -> SliceIntervals:
        """Get the slice intervals from the relevant feature."""
        if self.slice_intervals is None:
            if (
                resource.name == self.required_feature
            ):  # strict test for the exact feature, not subclasses
                self.slice_intervals = resource.get_slice_intervals(
                    level_name=self.level_name
                )
            else:
                raise SlicerNotSetUpError(self.dtype)
        return self.slice_intervals


class KeySlicer(AdjacencyGroupSlicer):
    """Slices resources by key."""

    _required_feature = "KeyAnnotations"

    def __init__(
        self,
        level_name: str = "localkey_slice",
        slice_intervals: Optional[SliceIntervals] = None,
        **kwargs,
    ):
        super().__init__(
            level_name=level_name, slice_intervals=slice_intervals, **kwargs
        )
