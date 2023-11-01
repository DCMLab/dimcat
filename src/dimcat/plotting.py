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
STD_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin={"l": 40, "r": 0, "b": 0, "t": 80, "pad": 0},
    font={"size": 25},
)
TRACES_SETTINGS = dict(marker_line_color="black")
Y_AXIS = dict(gridcolor="lightgrey")
X_AXIS = dict(gridcolor="lightgrey")


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


def fifths_bar_plot(
    bar_data,
    x_col="tpc",
    y_col="duration_qb",
    labels=None,
    title="Pitch-class distribution",
    fifth_transform=ms3.fifths2name,
    shift_color_midpoint=2,
    showlegend=False,
    width=1500,
    height=400,
    output=None,
    **kwargs,
):
    """bar_data with x_col ('tpc'), y_col ('duration_qb')"""

    color_values = list(bar_data[x_col])
    if labels is None:
        labels = {str(x_col): "Tonal pitch class", str(y_col): "Duration in ♩"}
    fig = px.bar(
        bar_data,
        x=x_col,
        y=y_col,
        title=title,
        labels=labels,
        color=color_values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=shift_color_midpoint,
        width=width,
        height=height,
        **kwargs,
    )
    x_values = list(set(color_values))
    x_names = list(map(fifth_transform, x_values))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(**STD_LAYOUT, showlegend=showlegend)
    fig.update_yaxes(gridcolor="lightgrey")
    fig.update_xaxes(
        gridcolor="lightgrey",
        zerolinecolor="grey",
        tickmode="array",
        tickvals=x_values,
        ticktext=x_names,
        dtick=1,
        ticks="outside",
        tickcolor="black",
        minor=dict(dtick=6, gridcolor="grey", showgrid=True),
    )
    if output is not None:
        fig.write_image(output)
    return fig


def get_pitch_class_distribution(
    df,
    pitch_column="tpc",
    duration_column="duration_qb",
):
    return (
        df.groupby(pitch_column)[duration_column].sum().to_frame(name=duration_column)
    )


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


def plot_pitch_class_distribution(
    df: pd.DataFrame,
    pitch_column="tpc",
    duration_column="duration_qb",
    title="Pitch class distribution",
    fifths_transform=ms3.fifths2name,
    width=1500,
    height=500,
    labels=None,
    modin=True,
    output=None,
):
    bar_data = get_pitch_class_distribution(
        df=df,
        pitch_column=pitch_column,
        duration_column=duration_column,
    ).reset_index()
    if modin:
        x_col, y_col = 0, 1
    else:
        x_col, y_col = pitch_column, duration_column
    return fifths_bar_plot(
        bar_data=bar_data,
        x_col=x_col,
        y_col=y_col,
        labels=labels,
        title=title,
        fifth_transform=fifths_transform,
        width=width,
        height=height,
        output=output,
    )


def tpc_bubbles(
    df: pd.Series | pd.DataFrame,
    normalize=True,
    width=1200,
    height=1500,
    title="Pitch class durations",
    duration_column="duration_qb",
    x_axis=None,
    y_axis=None,
    labels=None,
    output=None,
    flip=False,
    modin=False,
    layout: Optional[dict] = None,
    **kwargs,
):
    """
    Expecting a long format DataFrame/Series with two index levels where the first level groups pitch class
    distributions: Pitch classes are the second index level and the distribution values are contained in the Series
    or the first column. Additional columns may serve, e.g. to add more hover_data fields (by passing the column name(s)
    as keyword argument 'hover_data'.
    """
    if layout is None:
        layout = dict(STD_LAYOUT)

    xaxis_settings, yaxis_settings = dict(Y_AXIS), dict(X_AXIS)
    if flip:
        if modin:
            x, y = 1, 2
        else:
            *_, x, y = df.index.names
        color_col = y
        x_axis, y_axis = y_axis, x_axis
        layout.update(dict(width=height, height=width, xaxis_type="category"))
    else:
        if modin:
            x, y = 2, 1
        else:
            *_, y, x = df.index.names
        color_col = x
        layout.update(dict(height=height, width=width, yaxis_type="category"))
    if normalize:
        if isinstance(df, pd.Series):
            df = df.groupby(level=0, group_keys=False).apply(lambda S: S / S.sum())
        else:
            df.iloc[:, 0] = (
                df.iloc[:, 0]
                .groupby(level=0, group_keys=False)
                .apply(lambda S: S / S.sum())
            )
        title = "Normalized " + title
    df = df.reset_index()
    if modin:
        size_column = 2
    else:
        size_column = duration_column
    tpc_names = ms3.fifths2name(list(df.tpc))
    df["pitch class"] = tpc_names
    hover_data = kwargs.pop("hover_data", [])
    if isinstance(hover_data, str):
        hover_data = [hover_data]
    hover_data += ["pitch class"]
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size_column,
        color=color_col,
        **COLOR_SCALE_SETTINGS,
        labels=labels,
        title=title,
        hover_data=hover_data,
        **kwargs,
    )
    fig.update_traces(TRACES_SETTINGS)

    if not flip:
        yaxis_settings["autorange"] = "reversed"
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    if y_axis is not None:
        yaxis_settings.update(y_axis)
    fig.update_layout(xaxis=xaxis_settings, yaxis=yaxis_settings, **layout)
    fig.update_coloraxes(showscale=False)
    if output is not None:
        fig.write_image(output)
    return fig
