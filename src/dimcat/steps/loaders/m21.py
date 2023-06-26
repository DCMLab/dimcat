import dataclasses
from collections import defaultdict
from functools import cache
from inspect import signature
from typing import Dict, List

import music21 as m21
import pandas as pd

from .base import ScoreLoader
from .utils import get_m21_input_extensions


def default_list_dict():
    return defaultdict(list)


@dataclasses.dataclass
class CollectedElements:
    events: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    control: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    structure: List[dict] = dataclasses.field(default_factory=list)
    annotations: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    metadata: dict = dataclasses.field(default_factory=default_list_dict)
    part_ids: List[str] = dataclasses.field(default_factory=list)
    prelims: List[str] = dataclasses.field(default_factory=list)


IGNORED_PROPERTIES = [
    "activeSite",
    "classSet",
    "classSortOrder",
    "classes",
    "derivation",
    "equalityAttributes",
    "groups",
    "hasEditorialInformation",
    "hasStyleInformation",
    "id",
    "isStream",
    "priority",
    "sites",
    "style",
]


def m21_obj2dict(
    obj: m21.prebase.ProtoM21Object,
    exclude_private: bool = True,
    exclude_default: bool = True,
    exclude_callables: bool = True,
) -> dict:
    global IGNORED_PROPERTIES
    if isinstance(obj, m21.stream.Stream):
        raise TypeError("This function works for elements, not containers.")
    result = {}
    for attr in dir(obj):
        if exclude_private and attr.startswith("_"):
            continue
        if exclude_default and attr in IGNORED_PROPERTIES:
            continue
        value = getattr(obj, attr)
        if callable(value):
            if exclude_callables:
                continue
            else:
                try:
                    value = value.__call__()
                except Exception:
                    try:
                        value = str(signature(value))
                    except ValueError:
                        pass
        if isinstance(value, m21.prebase.ProtoM21Object):
            if isinstance(value, m21.beam.Beams):
                continue
            elif isinstance(value, m21.clef.Clef):
                value = f"{value.sign}{value.line}"
            elif isinstance(value, m21.duration.Duration):
                result[attr] = value.quarterLength
            elif isinstance(value, m21.editorial.Editorial):
                value = dict(value)
            elif isinstance(value, m21.pitch.Microtone):
                value = value.cents
            elif isinstance(value, m21.pitch.Pitch):
                value = value.nameWithOctave.replace("-", "b")
            else:
                raise NotImplementedError(
                    f"{type(value)} (value for {attr}): {m21_obj2dict(value)}"
                )
        if not value:
            continue
        result[attr] = value
    return result


class Music21Score:
    """Auxiliary class for extracting facets from a score parsed with music21."""

    def __init__(self, source: str):
        self.score = m21.converter.parse(source, forceSource=True)
        if isinstance(self.score, m21.stream.Opus):
            raise NotImplementedError("Opus not yet supported")
        if not isinstance(self.score, m21.stream.Score):
            raise ValueError(
                f"Music21 was expected to return a Score; got {type(self.score)} instead."
            )
        self.elements = CollectedElements()

    @cache
    def _get_method(self, element: m21.Music21Object):
        method_name = f"_parse_{element.__class__.__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)
        if isinstance(element, m21.spanner.Spanner):
            return self._parse_Spanner
        raise NotImplementedError(
            f"Method {method_name} for parsing {element.__class__} not implemented."
        )

    def parse(self):
        for top_level_element in self.score:
            self.parse_element(top_level_element)

    def parse_element(self, element: m21.Music21Object, **kwargs):
        method = self._get_method(element)
        method(element, **kwargs)

    def _parse_Measure(
        self, measure: m21.stream.Measure, measure_count: int, staff: int, **other_info
    ):
        """Inspired by https://github.com/MarkGotham/bar-measure/blob/5ddc7c6/Code/music21_application.py#L205"""
        end_repeat = False
        start_repeat = False
        time_sig = measure.timeSignature.ratioString if measure.timeSignature else None

        if measure.leftBarline and str(measure.leftBarline) == str(
            m21.bar.Repeat(direction="start")
        ):
            start_repeat = True
        if measure.rightBarline and str(measure.rightBarline) == str(
            m21.bar.Repeat(direction="end")
        ):
            end_repeat = True
        measure_info = dict(
            count=measure_count,
            staff=staff,
            qstamp=measure.offset,
            number=measure.measureNumber,
            name=measure.measureNumberWithSuffix(),
            nominal_length=measure.barDuration.quarterLength,
            actual_length=measure.duration.quarterLength,
            time_signature=time_sig,
            start_repeat=start_repeat,
            end_repeat=end_repeat,
            **other_info,
        )
        self.elements.structure.append(measure_info)

    def _parse_Metadata(self, metadata: m21.metadata.Metadata):
        for key, value in metadata.all():
            self.elements.metadata[key].append(value)

    def _parse_Part(self, part: m21.stream.Part):
        self.elements.part_ids.append(part.id)
        staff = len(self.elements.part_ids)
        measure_count = 0
        instrument_info = {}
        for element in part:
            if isinstance(element, m21.instrument.Instrument):
                instrument_info = m21_obj2dict(element)
            elif isinstance(element, m21.stream.Measure):
                measure_count += 1
                self._parse_Measure(
                    measure=element,
                    measure_count=measure_count,
                    staff=staff,
                    **instrument_info,
                )
            else:
                self.parse_element(element, measure_count=measure_count, staff=staff)

    def _parse_PartStaff(self, part_staff: m21.stream.PartStaff, **kwargs):
        self._parse_Part(part_staff, **kwargs)

    def _parse_ScoreLayout(self, score_layout: m21.layout.ScoreLayout, **kwargs):
        pass  # ignore

    def _parse_StaffGroup(self, staff_group: m21.layout.StaffGroup, **kwargs):
        pass  # ignore

    def _parse_Spanner(self, slur, **kwargs):
        pass  # ToDo

    def _parse_TextBox(self, text_box: m21.text.TextBox):
        self.elements.prelims.append(text_box.content)


class Music21Loader(ScoreLoader):
    accepted_file_extensions = get_m21_input_extensions()

    def _process_resource(self, resource: str) -> None:
        parsed_resource = m21.converter.parse(resource, forceSource=True)
        if isinstance(parsed_resource, m21.stream.Opus):
            raise NotImplementedError("Opus not yet supported")
