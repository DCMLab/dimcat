from __future__ import annotations

import logging
from typing import Iterable, Optional

import ms3
import pandas as pd
from dimcat.base import FriendlyEnum
from plotly import express as px
from plotly import graph_objs as go

logger = logging.getLogger(__name__)

COLOR_SCALE_SETTINGS = dict(
    color_continuous_scale="RdBu_r", color_continuous_midpoint=2
)
TRACES_SETTINGS = dict(marker_line_color="black")
Y_AXIS = dict()
X_AXIS = dict()


class GroupMode(FriendlyEnum):
    AXIS = "AXIS"
    COLOR = "COLOR"
    COLUMNS = "COLUMNS"
    ROWS = "ROWS"


GROUPMODE2BAR_PLOT_SETTING = {
    GroupMode.AXIS: "x",
    GroupMode.COLOR: "color",
    GroupMode.COLUMNS: "facet_col",
    GroupMode.ROWS: "facet_row",
}


def make_bar_plot(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_cols: Optional[str | Iterable[str]] = None,
    group_modes: Iterable[GroupMode] = (
        GroupMode.AXIS,
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    ),
    layout: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    """

    Args:
        layout: Keyword arguments passed to fig.update_layout()
        **kwargs: Keyword arguments passed to the Plotly plotting function.

    Returns:
        A Plotly Figure object.
    """
    df = df.reset_index()
    if y_col is None:
        y_col = df.columns[-1]
    plot_settings = dict(y=y_col, hover_data=["corpus", "piece"], height=500)
    if group_cols is not None:
        for group_col, group_mode in zip(group_cols, group_modes):
            setting_key = GROUPMODE2BAR_PLOT_SETTING[group_mode]
            if setting_key in plot_settings:
                raise ValueError(
                    f"Trying to set {setting_key!r} to {group_col!r} but it is already set to "
                    f"{plot_settings[setting_key]!r}."
                )
            plot_settings[setting_key] = group_col
    if "x" not in plot_settings:
        if x_col is None:
            plot_settings["x"] = df.columns[-2]
        else:
            plot_settings["x"] = x_col
    fig = px.bar(
        df,
        **plot_settings,
        **kwargs,
    )
    figure_layout = dict(
        xaxis_type="category",  # prevent Plotly from interpreting values as dates
    )
    if layout is not None:
        figure_layout.update(layout)
    fig.update_layout(figure_layout)
    return fig


def tpc_bubbles(
    df,
    width=1200,
    height=1500,
    title="Pitch class durations",
    x_axis=None,
    y_axis=None,
    output=None,
    flip=False,
    layout: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    """Expecting a long format DataFrame with two index levels where the first level groups pitch class distributions.

    Args:
        df:
        width:
        height:
        title:
        x_axis:
        y_axis:
        output:
        flip:

    Returns:

    """
    if layout is None:
        layout = dict()
    if flip:
        *_, x, y = df.index.names
        color_col = y
        x_axis, y_axis = y_axis, x_axis
        layout.update(
            dict(
                width=height,
                height=width,
                xaxis_type="category",  # prevent Plotly from interpreting values as dates
            )
        )
    else:
        *_, y, x = df.index.names
        color_col = x
        layout.update(
            dict(
                height=height,
                width=width,
                yaxis_type="category",  # prevent Plotly from interpreting values as dates
            )
        )
    df = df.reset_index()
    df["pitch class"] = ms3.fifths2name(df.tpc)
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size="duration in ♩",
        color=color_col,
        hover_data=["pitch class", "corpus"],
        **COLOR_SCALE_SETTINGS,
        title=title,
        **kwargs,
    )
    fig.update_traces(TRACES_SETTINGS)

    xaxis = dict(X_AXIS)
    yaxis = dict(Y_AXIS)
    if not flip:
        yaxis["autorange"] = "reversed"
    if x_axis is not None:
        xaxis.update(x_axis)
    if y_axis is not None:
        yaxis.update(y_axis)
    fig.update_layout(xaxis=xaxis, yaxis=yaxis, **layout)
    fig.update_coloraxes(showscale=False)
    if output is not None:
        fig.write_image(output)
    return fig
