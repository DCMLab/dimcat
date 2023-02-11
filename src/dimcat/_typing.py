from typing import NamedTuple, Tuple, TypeAlias, Union

import pandas as pd


class PieceID(NamedTuple):
    corpus: str
    piece: str


Pandas: TypeAlias = Union[pd.Series, pd.DataFrame]
GroupID: TypeAlias = tuple
SliceID: TypeAlias = Tuple[str, str, pd.Interval]
ID: TypeAlias = Union[PieceID, SliceID]
