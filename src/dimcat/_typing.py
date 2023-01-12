from typing import Tuple, TypeAlias, Union

import pandas as pd

Pandas: TypeAlias = Union[pd.Series, pd.DataFrame]
GroupID: TypeAlias = tuple
Index: TypeAlias = Union[Tuple[str, str], Tuple[str, str, pd.Interval]]
