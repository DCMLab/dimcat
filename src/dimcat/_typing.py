from typing import Tuple, TypeAlias, Union

import pandas as pd

Pandas: TypeAlias = Union[pd.Series, pd.DataFrame]
PieceID: TypeAlias = Tuple[str, str]
GroupID: TypeAlias = tuple
SliceID: TypeAlias = Tuple[str, str, pd.Interval]
ID: TypeAlias = Union[PieceID, SliceID]
