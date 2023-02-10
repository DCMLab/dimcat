# Defining the harmony concepts.
from __future__ import annotations

import re
import typing
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache

import regex_spm
from pitchtypes import SpelledIntervalClass as SpelledIntervalClass
from pitchtypes import SpelledPitchClass as _SpelledPitchClass

T = typing.TypeVar("T")


class SpelledPitchClass(_SpelledPitchClass):
    def alteration_ic(self) -> SpelledIntervalClass:
        if self.alteration() > 0:
            alteration_symbol = "a" * self.alteration()
        elif self.alteration() == 0:
            alteration_symbol = "P"  # perfect unison, no alterations
        elif self.alteration() < 0:
            alteration_symbol = "d" * abs(self.alteration())
        else:
            raise ValueError(self.alteration)
        interval_class = SpelledIntervalClass(f"{alteration_symbol}1")
        return interval_class


class HarmonyRegexes:
    """a class to hold all the regular expressions in the harmony string input"""

    key_regex = re.compile(
        "^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I
    )  # case-insensitive

    arabic_degree_regex = re.compile("^(?P<modifiers>[#b]*)(?P<number>(\d+))$")
    roman_degree_regex = re.compile(
        "^(?P<modifiers>[#b]*)(?P<roman_numeral>([VI]+))$", re.I
    )

    figuredbass_regex = re.compile("(?P<figbass>(7|65|43|42|2|64|6))")

    added_tone_regex = re.compile(
        r"(\+[#b]*\d*)"
    )  # TODO: ?? Added tones are always preceded by '+'?? double check the standards?

    replacement_tone_regex = re.compile(r"([\^v][#b]*\d*)")


class ProtocolHarmony(typing.Protocol):
    @abstractmethod
    def root(self) -> SpelledPitchClass:
        pass

    @abstractmethod
    def key_if_tonicized(self) -> Key:
        pass

    @abstractmethod
    def pc_set(self) -> typing.List[SpelledPitchClass]:
        pass

    @abstractmethod
    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        pass


@dataclass(frozen=True)
class Key:
    root: SpelledPitchClass
    mode: typing.Literal["M", "m"]

    _M_intervals = ["P1", "M2", "M3", "P4", "P5", "M6", "M7"]
    _m_intervals = ["P1", "M2", "m3", "P4", "P5", "m6", "m7"]

    @classmethod
    def parse(cls, key_str: str) -> Key:
        if not isinstance(key_str, str):
            raise TypeError(f"expected string as input, got {key_str}")
        key_match = HarmonyRegexes.key_regex.fullmatch(key_str)
        if key_match is None:
            raise ValueError(
                f"could not match '{key_str}' with regex: '{HarmonyRegexes.key_regex.pattern}'"
            )
        mode = "M" if key_match["class"].isupper() else "m"
        root = SpelledPitchClass(key_match["class"].upper() + key_match["modifiers"])
        instance = cls(root=root, mode=mode)
        return instance

    def find_pc(self, degree: Degree) -> SpelledPitchClass:
        if self.mode == "M":
            intervals = self._M_intervals
        elif self.mode == "m":
            intervals = self._m_intervals
        else:
            raise ValueError(f"{self.mode=}")
        interval = intervals[degree.number - 1]
        pc = self.root + SpelledIntervalClass(interval)
        pc = SpelledPitchClass(
            pc.name()
        )  # TODO: do it in the right way later (look up customized SpelledPitchClass)
        pc = pc - pc.alteration_ic()
        return pc

    @property
    @cache
    def pcs(self) -> typing.List[SpelledPitchClass]:
        return [
            self.find_pc(degree=Degree(number=n, alteration=0)) for n in range(1, 8)
        ]

    def to_str(self) -> str:
        if self.mode == "m":
            resulting_str = str(self.root).lower()
        elif self.mode == "M":
            resulting_str = str(self.root)
        else:
            raise ValueError(f"not applicable mode")
        return resulting_str


@dataclass
class Degree:
    number: int
    alteration: int | bool  # when int: positive for "#", negative for "b", when bool: represent whether to use natural

    _numeral_scale_degree_dict = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
    }

    def __add__(self, other: typing.Self) -> typing.Self:
        """
        n steps (0 steps is unison) <-- degree (1 is unison)
        |
        V
        n steps (0 steps is unison) --> degree (1 is unison)

        """
        number = ((self.number - 1) + (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    def __sub__(self, other: typing.Self) -> typing.Self:
        number = ((self.number - 1) - (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    @classmethod
    def parse(cls, degree_str: str) -> typing.Self:
        """
        Examples of arabic_degree: b7, #2, 3, 5, #5, ...
        Examples of scale degree: bV, bIII, #II, IV, vi, vii
        """
        match = regex_spm.fullmatch_in(degree_str)
        match match:
            case HarmonyRegexes.roman_degree_regex:
                degree_number = cls._numeral_scale_degree_dict.get(
                    match["roman_numeral"]
                )
            case HarmonyRegexes.arabic_degree_regex:
                degree_number = int(match["number"])
            case _:
                raise ValueError(
                    f"could not match {match} with regex: {HarmonyRegexes.roman_degree_regex} or {HarmonyRegexes.arabic_degree_regex}"
                )
        modifiers_match = match["modifiers"]
        alteration = SpelledPitchClass(f"C{modifiers_match}").alteration()
        instance = cls(number=degree_number, alteration=alteration)
        return instance


class IntervalQuality:
    @classmethod
    @abstractmethod
    def from_interval_class(
        cls, spelled_interval_class: SpelledIntervalClass
    ) -> typing.Self:
        pass


@dataclass
class P(IntervalQuality):
    """
    This is a type class for perfect intervals
    alt_steps = 0 means e.g. perfect unison, perfect fifths
    alt_steps = 1 means aug 1 (A1), aug 5(A5), aug 4...
    alt_steps = -1 means dim 1 (D1) ...
    """

    alt_steps: int

    @classmethod
    def from_interval_class(
        cls, spelled_interval_class: SpelledIntervalClass
    ) -> typing.Self:  # TODO: fill in
        pass


@dataclass
class IP(IntervalQuality):
    """
    This is a type class for Imperfect intervals
    alt_steps = 1 means M2, M3, M6 ...
    alt_steps = 2 means A2, A3, A6 ...
    alt_steps = 3 means AA2, AA3, ...
    alt_steps = -1 means m2, m3, m6 ...
    """

    alt_steps: int

    def __post_init__(self):
        if self.alt_steps == 0:
            raise ValueError(f"{self.alt_steps=}")
        self.alt_steps = self.alt_steps

    @classmethod
    def from_interval_class(
        cls, spelled_interval_class: SpelledIntervalClass
    ) -> typing.Self:  # TODO: fill in
        pass


@dataclass
class HarmonyQuality:
    """Examples:
    stack_of_thirds_3_Mm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(1),IP(-1)]) # Major triad
    stack_of_thirds_4_mmm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(-1),IP(-1),IP(-1)]) # fully diminished seventh chord
    stack_of_fifth_4_PPP = HarmonyQuality(stacksize=5,ic_quality_list=[P(0),P(0),P(0)])
    stack_of_fifth_3_ADP = HarmonyQuality(stacksize=5,ic_quality_list=[P(-1),P(1),P(0)])
    """

    stack_size: int
    interval_class_quality_list: typing.List[P | IP]

    _map_to_ic_qualities_3 = {
        (True, "+"): [1, 1],  # augmented triad
        (True, ""): [1, -1],  # major triad
        (False, ""): [-1, 1],  # minor triad
        (False, "o"): [-1, -1],
    }  # diminished triad

    _map_to_ic_qualities_4 = {
        (True, "+M"): [1, 1, 1],  # aug major 7th
        (False, "M"): [-1, 1, 1],  # minor major 7th
        (True, "M"): [1, -1, 1],  # major major 7th
        (True, "+"): [1, 1, -1],  # aug minor 7th
        (False, "%"): [-1, -1, 1],  # half dim 7th
        (False, ""): [-1, 1, -1],  # minor minor 7th
        (True, ""): [1, -1, -1],  # dominant 7th
        (False, "o"): [-1, -1, -1],
    }  # fully dim 7th

    _which_dict = {3: _map_to_ic_qualities_3, 4: _map_to_ic_qualities_4}

    @classmethod
    def smart_init(
        cls,
        n_chord_tones: int,
        upper: bool,
        form_symbol: typing.Literal["o", "+", "%", "M", "+M", ""],
    ) -> typing.Self:
        dict_to_check = cls._which_dict[n_chord_tones]
        ic_quality_list = [IP(x) for x in dict_to_check[(upper, form_symbol)]]
        instance = cls(stack_size=3, interval_class_quality_list=ic_quality_list)
        return instance


@dataclass
class FiguredBass:
    degrees: typing.List[Degree]

    _figbass_degree_dict = {
        "7": [1, 3, 5, 7],
        "65": [3, 5, 7, 1],
        "43": [5, 7, 1, 3],
        "42": [7, 1, 3, 5],
        "2": [7, 1, 3, 5],
        "64": [5, 1, 3],
        "6": [3, 5, 1],
    }  # assume the first number x is: root + x = bass , 1 := unison

    @classmethod
    def parse(
        cls, figbass_str: str
    ) -> typing.Self:  # TODO: revise with alteration, add a dict to parse
        match = regex_spm.fullmatch_in(figbass_str)
        match match:
            case HarmonyRegexes.figuredbass_regex:
                instance = cls(
                    degrees=[
                        Degree.parse(degree_str=str(x))
                        for x in FiguredBass._figbass_degree_dict.get(figbass_str)
                    ]
                )
            case _:  # otherwise, assume root position triad
                instance = cls(
                    degrees=[
                        Degree(number=1, alteration=0),
                        Degree(number=3, alteration=0),
                        Degree(number=5, alteration=0),
                    ]
                )
        return instance

    def n_chord_tones(self) -> int:
        return len(self.degrees)


@dataclass
class AddedTones:
    degrees: typing.List[Degree]

    @classmethod
    def parse(cls, added_tones_str: str) -> typing.Self:
        match = regex_spm.match_in(added_tones_str)

        match match:
            case HarmonyRegexes.added_tone_regex:
                match_iter = re.finditer(
                    HarmonyRegexes.added_tone_regex, string=added_tones_str
                )
                degrees = [
                    Degree.parse(degree_str=match[0].replace("+", ""))
                    for match in match_iter
                ]
                instance = cls(degrees=degrees)
                return instance
            case _:
                return None


@dataclass
class ReplacementTones:
    degrees: typing.List[Degree]

    _regex_pattern = re.compile(r"([\^v][#b]*\d*)")

    @classmethod
    def parse(cls, replacement_tones_str: str) -> typing.Self:
        """
        V7(9) , V7(b9) , V7(v#9), V7(^9) , V7(^b9) , V7(#9),
        """
        match = regex_spm.match_in(replacement_tones_str)
        match match:
            case cls._regex_pattern:
                match_iter = re.finditer(
                    cls._regex_pattern, string=replacement_tones_str
                )
            case _:
                return None
        replacement_tones = [
            Degree.parse(degree_str=match[0].replace("^", "").replace("v", ""))
            for match in match_iter
        ]  # TODO: current version do not consider replaced from above or below (i.e., 'v', '^')
        instance = cls(degrees=replacement_tones)
        return instance


@dataclass
class SingleNumeralParts:
    roman_numeral: str
    modifiers: str
    form: str
    figbass: str
    added_tones: str
    replacement_tones: str

    # the regular expression conforms with the DCML annotation standards
    _sn_regex = re.compile(
        "^(?P<modifiers>(b*)|(#*))"  # accidentals
        "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
        "(?P<form>(%|o|\+|M|\+M))?"  # quality form
        "(?P<figbass>(7|65|43|42|2|64|6))?"  # figured bass
        "(?P<added_tones>(\(\+[#b]*\d*)\))?"  # added tones, non-chord tones added within parentheses and preceded by a "+"
        "(?P<replacement_tones>(\([\^v][#b]*\d*)\))?$"
    )  # replaced chord tones expressed through intervals <= 8

    @classmethod
    def parse(cls, numeral_str: str) -> typing.Self:
        # match with regex
        s_numeral_match = SingleNumeralParts._sn_regex.match(numeral_str)
        if s_numeral_match is None:
            raise ValueError(
                f"could not match '{numeral_str}' with regex: '{SingleNumeralParts._sn_regex.pattern}'"
            )

        roman_numeral = s_numeral_match["roman_numeral"]
        modifiers = s_numeral_match["modifiers"] if s_numeral_match["modifiers"] else ""
        form = s_numeral_match["form"] if s_numeral_match["form"] else ""
        figbass = s_numeral_match["figbass"] if s_numeral_match["figbass"] else ""
        added_tones = (
            s_numeral_match["added_tones"].replace("(", "").replace(")", "")
            if s_numeral_match["added_tones"]
            else ""
        )
        replacement_tones = (
            s_numeral_match["replacement_tones"].replace("(", "").replace(")", "")
            if s_numeral_match["replacement_tones"]
            else ""
        )

        instance = cls(
            modifiers=modifiers,
            roman_numeral=roman_numeral,
            form=form,
            figbass=figbass,
            added_tones=added_tones,
            replacement_tones=replacement_tones,
        )
        return instance


class SnpParsable(typing.Protocol):
    @classmethod
    @abstractmethod
    def parse_snp(cls, snp: SingleNumeralParts) -> typing.Self:
        pass


@dataclass(frozen=True)
class SingleNumeral(ProtocolHarmony):
    key: Key
    degree: Degree
    quality: HarmonyQuality
    figbass: FiguredBass
    added_tones: AddedTones
    replacement_tones: ReplacementTones

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> typing.Self:
        snp = SingleNumeralParts.parse(numeral_str=numeral_str)

        # parse key:
        if not isinstance(key_str, Key):
            key = Key.parse(key_str=key_str)
        else:
            key = key_str

        # parse degree:
        degree = Degree.parse(degree_str=snp.modifiers + snp.roman_numeral)

        # parse added_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        added_tones = AddedTones.parse(added_tones_str=snp.added_tones)

        # replacement_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        replacement_tones = ReplacementTones.parse(
            replacement_tones_str=snp.replacement_tones
        )

        # parse figbass:
        figbass = FiguredBass.parse(figbass_str=snp.figbass)

        # parse quality, in stack of thirds:
        quality = HarmonyQuality.smart_init(
            n_chord_tones=figbass.n_chord_tones(),
            upper=snp.roman_numeral.isupper(),
            form_symbol=snp.form,
        )

        # create class instance:
        instance = cls(
            key=key,
            degree=degree,
            quality=quality,
            figbass=figbass,
            added_tones=added_tones,
            replacement_tones=replacement_tones,
        )
        return instance

    def root(self) -> SpelledPitchClass:
        root = self.key.find_pc(self.degree)
        return root

    def key_if_tonicized(self) -> Key:
        """
        Make the current numeral as the tonic, return the spelled pitch class of the root as Key.
        """
        major_minor_key_mode = (
            "M"
            if (
                self.quality.interval_class_quality_list[0] == IP(1)
                and self.quality.interval_class_quality_list[1] == IP(-1)
            )
            else "m"
        )
        key = Key(root=self.root(), mode=major_minor_key_mode)  # TODO: better version?
        return key

    def bass_degree(self) -> Degree:
        bass_degree = self.degree + self.figbass.degrees[0]
        return bass_degree

    def bass_pc(self) -> SpelledPitchClass:
        pc = self.key.find_pc(self.bass_degree())
        return pc

    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        raise NotImplementedError

    def pc_set(self) -> typing.List[SpelledPitchClass]:
        raise NotImplementedError


@dataclass
class Chain(typing.Generic[T]):
    head: T
    tail: typing.Optional[Chain[T]]


class Numeral(Chain[SingleNumeral]):
    @classmethod
    def parse(cls, key_str: str, numeral_str: str) -> typing.Self:
        # numeral_str examples: "#ii/V", "##III/bIV/V", "bV", "IV(+6)", "vii%7/IV"

        if "/" in numeral_str:
            L_numeral_str, R_numeral_str = numeral_str.split("/", maxsplit=1)
            R = cls.parse(key_str=key_str, numeral_str=R_numeral_str)
            L = SingleNumeral.parse(
                key_str=R.head.key_if_tonicized(), numeral_str=L_numeral_str
            )

        else:
            L = SingleNumeral.parse(key_str=key_str, numeral_str=numeral_str)
            R = None

        instance = cls(head=L, tail=R)
        return instance


@dataclass(frozen=True)
class TonalHarmony(ProtocolHarmony):
    globalkey: Key
    numeral: Numeral
    bookeeping: typing.Dict[str, str]  # for bookkeeping

    @classmethod
    def parse(
        cls, globalkey_str: str, localkey_numeral_str: str, chord_str: str
    ) -> typing.Self:
        # chord_str examples: "IV(+6)", "vii%7/IV", "ii64"
        globalkey = Key.parse(key_str=globalkey_str)
        localkey = Numeral.parse(
            key_str=globalkey_str, numeral_str=localkey_numeral_str
        ).head.key_if_tonicized()
        compound_numeral = Numeral.parse(
            key_str=localkey.to_str(), numeral_str=chord_str
        )
        instance = cls(
            globalkey=globalkey,
            numeral=compound_numeral,
            bookeeping={
                "globalkey_str": globalkey_str,
                "localkey_numeral_str": localkey_numeral_str,
                "chord_str": chord_str,
            },
        )
        return instance

    def pc_set(self) -> typing.List[SpelledPitchClass]:
        pitchclass = self.chord_tones() | self.added_tones()
        return pitchclass

    def root(self) -> SpelledPitchClass:
        pass

    def key_if_tonicized(self) -> Key:
        pass

    def to_numeral(self) -> Numeral:
        pass

    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        pass

    def added_tones(self) -> typing.List[SpelledPitchClass]:
        pass


if __name__ == "__main__":
    # result = Numeral.parse(key_str='C', numeral_str='V7(v#9)/IV/III')
    result = TonalHarmony.parse(
        globalkey_str="C", localkey_numeral_str="I", chord_str="V"
    ).root()

    print(f"{result=}")
