import dataclasses
import logging
from collections import defaultdict
from functools import cache
from inspect import signature
from typing import DefaultDict, List, Tuple

import music21 as m21
import pandas as pd
from dimcat.data.resources.base import Resource

from .base import ScoreLoader
from .utils import get_m21_input_extensions

logger = logging.getLogger(__name__)


def default_list_dict():
    return defaultdict(list)


@dataclasses.dataclass
class CollectedElements:
    events: List[dict] = dataclasses.field(default_factory=list)
    control: List[dict] = dataclasses.field(default_factory=list)
    structure: List[dict] = dataclasses.field(default_factory=list)
    annotations: List[str] = dataclasses.field(default_factory=list)
    metadata: DefaultDict = dataclasses.field(default_factory=default_list_dict)
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
    "stringPitches",
    "style",
]


def parse_Interval(interval: m21.interval.Interval) -> str:
    return interval.directedName


def parse_Key(key: m21.key.Key) -> int:
    return key.sharps


def parse_m21_object(
    obj: m21.prebase.ProtoM21Object,
    exclude_private: bool = True,
    exclude_default: bool = True,
    exclude_callables: bool = True,
    **higher_level_info,
) -> dict:
    global IGNORED_PROPERTIES
    if isinstance(obj, m21.stream.Stream):
        raise TypeError(
            f"This function works for elements, not containers such as {type(obj)}."
        )
    result = dict(higher_level_info)
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
            if isinstance(
                value,
                (
                    m21.beam.Beams,
                    m21.style.TextStyle,
                    m21.stream.Stream,
                ),
            ):
                continue
            elif isinstance(value, m21.scale.AbstractScale):
                value = parse_AbstractScale(value)
            elif isinstance(value, m21.chord.Chord):
                value = parse_Chord(value)
            elif isinstance(value, m21.clef.Clef):
                value = parse_Clef(value)
            elif isinstance(value, m21.scale.ConcreteScale):
                value = parse_ConcreteScale(value)
            elif isinstance(value, m21.duration.Duration):
                value = parse_Duration(value)
            elif isinstance(value, m21.editorial.Editorial):
                value = parse_Editorial(value)
            elif isinstance(value, m21.interval.Interval):
                value = parse_Interval(value)
            elif isinstance(value, m21.pitch.Microtone):
                value = parse_Microtone(value)
            elif isinstance(value, m21.pitch.Pitch):
                value = parse_Pitch(value)
            elif isinstance(value, m21.tie.Tie):
                value = parse_Tie(value)
            elif isinstance(value, m21.volume.Volume):
                value = parse_Volume(value)
            else:
                raise NotImplementedError(f"{type(value)} (value for {attr})")
        if not value:
            continue
        result[attr] = value
    return result


def parse_Volume(volume: m21.volume.Volume) -> float:
    return volume.velocity


def parse_Tie(tie: m21.tie.Tie) -> str:
    return tie.type


def parse_Barline(barline: m21.bar.Barline):
    return barline.type


def parse_Harmony(harmony: m21.harmony.Harmony) -> str:
    return harmony.figure


def parse_Microtone(microtone: m21.pitch.Microtone) -> float:
    return microtone.cents


def parse_Editorial(editorial: m21.editorial.Editorial) -> dict:
    return dict(editorial)


def parse_Duration(duration: m21.duration.Duration) -> float:
    return duration.quarterLength


def parse_ConcreteScale(concrete_scale: m21.scale.ConcreteScale) -> Tuple[str, ...]:
    return tuple(parse_Pitch(p) for p in concrete_scale.getPitches())


def parse_Clef(clef: m21.clef.Clef) -> str:
    return f"{clef.sign}{clef.line}"


def parse_Dynamic(dynamic: m21.dynamics.Dynamic) -> str:
    return dynamic.value


def parse_Chord(chord: m21.chord.Chord):
    return tuple(parse_Pitch(p) for p in chord)


def parse_AbstractScale(value):
    value = tuple(parse_Interval(i) for i in value.getIntervals())
    return value


def parse_TextExpression(text_expression: m21.expressions.TextExpression) -> str:
    return text_expression.content


def parse_Measure(measure: m21.stream.Measure, **higher_level_info):
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
    return dict(
        higher_level_info,
        qstamp=measure.offset,
        number=measure.measureNumber,
        name=measure.measureNumberWithSuffix(),
        nominal_length=measure.barDuration.quarterLength,
        actual_length=measure.duration.quarterLength,
        time_signature=time_sig,
        start_repeat=start_repeat,
        end_repeat=end_repeat,
    )


def parse_Pitch(pitch: m21.pitch.Pitch):
    return pitch.nameWithOctave.replace("-", "b")


def parse_TimeSignature(time_signature: m21.meter.TimeSignature) -> str:
    return time_signature.ratioString


class Music21Score:
    """Auxiliary class for extracting facets from a score parsed with music21."""

    def __init__(self, source: str):
        self.score = m21.converter.parse(source, forceSource=True)
        self.elements = CollectedElements()
        self.chord_id = -1

    @cache
    def _get_method(self, element: m21.Music21Object):
        method_name = f"_parse_{element.__class__.__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)
        if isinstance(element, m21.spanner.Spanner):
            return self._parse_Spanner
        if isinstance(element, m21.clef.Clef):
            return self._parse_Clef
        if isinstance(element, m21.harmony.Harmony):
            return self._parse_Harmony
        if isinstance(element, m21.key.KeySignature):
            return self._parse_Key
        raise NotImplementedError(
            f"Method {method_name} for parsing {element.__class__} not implemented."
        )

    def add_annotation(self, annotation: dict):
        self.elements.annotations.append(annotation)

    def add_control_event(self, event: str, event_info: dict):
        self.elements.control.append(dict(event=event, **event_info))

    def add_event(self, event: str, event_info: dict):
        self.elements.events.append(dict(event=event, **event_info))

    def add_metadata(self, key, value):
        self.elements.metadata[key].append(value)

    def add_structure(self, structure: dict):
        self.elements.structure.append(structure)

    def parse(self):
        if isinstance(self.score, m21.stream.Opus):
            for score in self.score.scores:
                self._parse_score(score)
        elif isinstance(self.score, m21.stream.Score):
            self._parse_score(self.score)
        else:
            raise NotImplementedError(
                f"music21.converter.parse() was expected to return a Score; got {type(self.score)} instead."
            )

    def _parse_score(self, score: m21.stream.Score):
        for top_level_element in score:
            _ = self.parse_element(top_level_element)

    def parse_element(self, element: m21.Music21Object, **kwargs) -> dict:
        method = self._get_method(element)
        return method(element, **kwargs)

    def _parse_Clef(self, clef_object: m21.clef.Clef, **higher_level_info):
        clef_info = dict(higher_level_info)
        clef_info["clef"] = parse_Clef(clef_object)
        self.add_control_event(event="Clef", event_info=clef_info)
        return clef_info

    def _parse_Chord(self, chord: m21.chord.Chord, **higher_level_info):
        self.chord_id += 1
        chord_info = dict(**higher_level_info, chord_id=self.chord_id)
        self.add_control_event(event="Chord", event_info=chord_info)
        for element in chord:
            _ = self.parse_element(element, **chord_info)

    def _parse_Dynamic(self, dynamic: m21.dynamics.Dynamic, **higher_level_info):
        dynamic_info = dict(**higher_level_info, dynamic=parse_Dynamic(dynamic))
        self.add_control_event(event="Dynamic", event_info=dynamic_info)
        return dynamic_info

    def _parse_Harmony(self, harmony: m21.harmony.Harmony, **higher_level_info):
        chord_symbol_info = dict(**higher_level_info, label=parse_Harmony(harmony))
        self.add_annotation(chord_symbol_info)
        return chord_symbol_info

    def _parse_Key(self, key: m21.key.Key, **higher_level_info):
        key_info = dict(higher_level_info, key_signature=parse_Key(key))
        self.add_control_event(event="Key", event_info=key_info)
        return key_info

    def _parse_Measure(
        self,
        measure: m21.stream.Measure,
        measure_count: int,
        staff: int,
        **higher_level_info,
    ):
        measure_info = parse_Measure(
            measure=measure,
            measure_count=measure_count,
            staff=staff,
            **higher_level_info,
        )
        voice_count = 0
        for element in measure:
            if isinstance(element, m21.meter.TimeSignature):
                # already handled by parse_Measure
                continue
            if isinstance(element, m21.bar.Barline):
                measure_info["barline"] = parse_Barline(element)
                continue
            if isinstance(element, m21.repeat.Fine):
                measure_info["marker"] = "fine"
                continue
            if isinstance(element, m21.repeat.Segno):
                measure_info["marker"] = "segno"
                continue
            if isinstance(element, m21.repeat.DalSegnoAlCoda):
                measure_info["jump_bwd"] = "segno"
                measure_info["play_until"] = "coda"
                measure_info["jump_fwd"] = "codab"
                continue
            if isinstance(element, m21.repeat.DalSegnoAlFine):
                measure_info["jump_bwd"] = "segno"
                measure_info["play_until"] = "fine"
                continue

            if isinstance(element, m21.stream.Voice):
                voice_count += 1
                tmp_measure_info = dict(measure_info, voice=voice_count)
                for sub_element in element:
                    _ = self.parse_element(sub_element, **tmp_measure_info)
                continue
            element_info = self.parse_element(element, **measure_info)
            if element_info and isinstance(
                element, (m21.clef.Clef, m21.tempo.MetronomeMark, m21.key.Key)
            ):
                measure_info.update(element_info)
        self.add_structure(measure_info)

    def _parse_Metadata(self, metadata: m21.metadata.Metadata):
        for key, value in metadata.all():
            self.add_metadata(key, value)

    def _parse_MetronomeMark(
        self, metronome_mark: m21.tempo.MetronomeMark, **higher_level_info
    ):
        metronome_mark_info = parse_m21_object(metronome_mark, **higher_level_info)
        self.add_control_event(event="MetronomeMark", event_info=metronome_mark_info)
        return metronome_mark_info

    def _parse_Note(self, note: m21.note.Note, **higher_level_info):
        note_info = parse_m21_object(note, **higher_level_info)
        self.add_event(event="Note", event_info=note_info)
        return note_info

    def _parse_Rest(self, rest: m21.note.Rest, **higher_level_info):
        rest_info = parse_m21_object(rest, **higher_level_info)
        self.add_event(event="Rest", event_info=rest_info)
        return rest_info

    def _parse_Part(self, part: m21.stream.Part):
        self.elements.part_ids.append(part.id)
        staff = len(self.elements.part_ids)
        measure_count = 0
        instrument_info = {}
        for element in part:
            if isinstance(element, m21.instrument.Instrument):
                instrument_info = parse_m21_object(element)
            elif isinstance(element, m21.stream.Measure):
                measure_count += 1
                self._parse_Measure(
                    measure=element,
                    measure_count=measure_count,
                    staff=staff,
                    **instrument_info,
                )
            else:
                _ = self.parse_element(
                    element, measure_count=measure_count, staff=staff
                )

    def _parse_PageLayout(self, page_layout: m21.layout.PageLayout, **kwargs):
        pass  # ignore

    def _parse_PartStaff(self, part_staff: m21.stream.PartStaff, **kwargs):
        """Treat PartStaff juast as any Part."""
        self._parse_Part(part_staff, **kwargs)

    def _parse_ScoreLayout(self, score_layout: m21.layout.ScoreLayout, **kwargs):
        pass  # ignore

    def _parse_StaffGroup(self, staff_group: m21.layout.StaffGroup, **kwargs):
        pass  # ignore

    def _parse_StaffLayout(self, staff_layout: m21.layout.StaffLayout, **kwargs):
        pass  # ignore

    def _parse_SystemLayout(self, system_layout: m21.layout.SystemLayout, **kwargs):
        pass  # ignore

    def _parse_Spanner(self, spanner, **kwargs):
        """ToDo"""
        logger.debug(f"Spanner not yet supported: {spanner}")

    def _parse_TextBox(self, text_box: m21.text.TextBox):
        self.elements.prelims.append(text_box.content)

    def _parse_TextExpression(
        self, text_expression: m21.expressions.TextExpression, **higher_level_info
    ):
        text_expression_info = dict(
            higher_level_info, text=parse_TextExpression(text_expression)
        )
        self.add_control_event(event="TextExpression", event_info=text_expression_info)
        return text_expression_info

    def _parse_TimeSignature(
        self, time_signature: m21.meter.TimeSignature, **higher_level_info
    ):
        time_signature_info = dict(
            higher_level_info, timesig=parse_TimeSignature(time_signature)
        )
        self.add_control_event(event="TimeSignature", event_info=time_signature_info)
        return time_signature_info

    def _parse_Voice(self, voice: m21.stream.Voice, **higher_level_info):
        for element in voice:
            _ = self.parse_element(element, **higher_level_info)


def make_dataframe(records: List[dict], drop_empty_columns: bool = True):
    df = pd.DataFrame.from_records(records)
    column_is_empty = df.isna().all()
    if column_is_empty.any():
        if drop_empty_columns:
            removed = df.columns[column_is_empty].tolist()
            logger.debug(f"Removing empty columns: {removed}")
            df = df.dropna(axis=1, how="all")
        else:
            logger.debug(f"Keeping empty columns:\n{column_is_empty}")

    return df


def make_metadata(metadata_dict) -> pd.Series:
    metadata = {
        k: v[0] if len(v) == 1 else ", ".join(str(e) for e in v)
        for k, v in metadata_dict.items()
        if v
    }
    return pd.Series(metadata)


class Music21Loader(ScoreLoader):
    """Extracts information from scores using music21."""

    _accepted_file_extensions = get_m21_input_extensions()
    _default_loader_name = "music21"

    def _process_resource(
        self,
        resource: Resource,
    ) -> None:
        ID = resource.ID
        filepath = resource.normpath
        score = Music21Score(filepath)
        score.parse()
        for facet_name, obj in zip(
            ("events", "control", "structure", "annotations", "metadata"),
            (
                make_dataframe(score.elements.events),
                make_dataframe(score.elements.control),
                make_dataframe(score.elements.structure),
                make_dataframe(score.elements.annotations),
                make_metadata(score.elements.metadata),
            ),
        ):
            self.add_piece_facet_dataframe(facet_name, ID, obj)
