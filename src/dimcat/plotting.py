from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import ms3
import pandas as pd
from dimcat.base import FriendlyEnum, get_setting
from dimcat.utils import resolve_path
from kaleido.scopes.plotly import PlotlyScope
from plotly import express as px
from plotly import graph_objects as go

AVAILABLE_FIGURE_FORMATS = PlotlyScope._all_formats


logger = logging.getLogger(__name__)

STD_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin={"l": 40, "r": 0, "b": 0, "t": 80, "pad": 0},
    font={"size": 20},
)
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


def get_pitch_class_distribution(
    df,
    pitch_column="tpc",
    duration_column="duration_qb",
):
    return (
        df.groupby(pitch_column)[duration_column].sum().to_frame(name=duration_column)
    )


def make_plot_settings(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_cols: Optional[str | Iterable[str]] = None,
    group_modes: Iterable[GroupMode] = (
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    ),
    title: Optional[str] = None,
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    if x_col is None:
        x_col = df.columns[-2]
    if y_col is None:
        y_col = df.columns[-1]
    plot_settings = dict(
        title=title,
        y=y_col,
        labels=labels,
        hover_data=hover_data,
        height=height,
        width=width,
    )
    if group_cols is not None:
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        for group_col, group_mode in zip(group_cols, group_modes):
            setting_key = GROUPMODE2BAR_PLOT_SETTING[group_mode]
            if setting_key in plot_settings:
                raise ValueError(
                    f"Trying to set {setting_key!r} to {group_col!r} but it is already set to "
                    f"{plot_settings[setting_key]!r}."
                )
            plot_settings[setting_key] = group_col
    if "x" not in plot_settings:
        plot_settings["x"] = x_col
    return plot_settings


def make_bar_plot(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_cols: Optional[str | Iterable[str]] = None,
    group_modes: Iterable[GroupMode] = (
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    ),
    title: Optional[str] = None,
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
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
    plot_settings = make_plot_settings(
        df=df,
        x_col=x_col,
        y_col=y_col,
        group_cols=group_cols,
        group_modes=group_modes,
        title=title,
        labels=labels,
        hover_data=hover_data,
        height=height,
        width=width,
    )
    fig = px.bar(
        df,
        **plot_settings,
        **kwargs,
    )
    update_figure_layout(
        fig=fig,
        layout=layout,
        x_axis=x_axis,
        y_axis=y_axis,
        color_axis=color_axis,
        traces_settings=traces_settings,
        xaxis_type="category",
    )
    if output is not None:
        write_image(fig=fig, filename=output, width=width, height=height)
    return fig


def make_bubble_plot(
    df: pd.Series | pd.DataFrame,
    normalize: bool = True,
    flip: bool = False,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    duration_column: str = "duration_qb",
    title: Optional[str] = None,
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 1500,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
    **kwargs,
):
    """
    Expecting a long format DataFrame/Series with two index levels where the first level groups pitch class
    distributions: Pitch classes are the second index level and the distribution values are contained in the Series
    or the first column. Additional columns may serve, e.g. to add more hover_data fields (by passing the column name(s)
    as keyword argument 'hover_data'.

    Args:
        x_col:
        y_col:
    """
    xaxis_settings, yaxis_settings = dict(X_AXIS), dict(Y_AXIS)
    if flip:
        x_axis, y_axis = y_axis, x_axis
        xaxis_settings, yaxis_settings = yaxis_settings, xaxis_settings
        figure_layout = dict(width=height, height=width)
        if y_col == "piece":
            figure_layout["xaxis_type"] = "category"
        else:
            figure_layout["xaxis_type"] = "linear"
    else:
        figure_layout = dict(height=height, width=width)
        if y_col == "piece":
            figure_layout["yaxis_type"] = "category"
        else:
            figure_layout["yaxis_type"] = "linear"
    if layout is not None:
        figure_layout.update(layout)
    if normalize:
        if isinstance(df, pd.Series):
            df = df.groupby(y_col, group_keys=False).apply(lambda S: S / S.sum())
        else:
            df.loc[:, duration_column] = df.groupby(y_col, group_keys=False)[
                duration_column
            ].apply(lambda S: S / S.sum())
    traces = dict(marker_line_color="black")
    if traces_settings is not None:
        traces.update(traces_settings)
    if not flip:
        yaxis_settings["autorange"] = "reversed"
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    if y_axis is not None:
        yaxis_settings.update(y_axis)
    c_axis = dict(showscale=False)
    if color_axis is not None:
        c_axis.update(color_axis)
    return make_scatter_plot(
        df=df,
        x_col=x_col,
        y_col=y_col,
        title=title,
        labels=labels,
        hover_data=hover_data,
        height=height,
        width=width,
        layout=figure_layout,
        x_axis=xaxis_settings,
        y_axis=yaxis_settings,
        color_axis=c_axis,
        traces_settings=traces,
        output=output,
        # **kwargs:
        size=duration_column,
        **kwargs,
    )


def make_lof_bar_plot(
    df: pd.DataFrame,
    fifth_transform=ms3.fifths2name,
    shift_color_midpoint=2,
    x_col="tpc",
    y_col="duration_qb",
    title=None,
    labels=None,
    hover_data: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
    **kwargs,
):
    """Like :func:`make_bar_plot` but coloring the bars along the Line of Fifths.
    bar_data with x_col ('tpc'), y_col ('duration_qb')"""
    if title is None:
        title = "Pitch-class distribution"
    if labels is None:
        labels = {str(x_col): "Tonal pitch class", str(y_col): "Duration in ♩"}
    df = df.reset_index()
    color_values = df[x_col].to_list()
    x_values = sorted(set(color_values))
    x_names = list(map(fifth_transform, x_values))
    figure_layout = dict(showlegend=False)
    if layout is not None:
        figure_layout.update(layout)
    xaxis_settings = dict(
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
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    c_axis = dict(showscale=False)
    if color_axis is not None:
        c_axis.update(color_axis)
    return make_bar_plot(
        df,
        x_col=x_col,
        y_col=y_col,
        title=title,
        labels=labels,
        hover_data=hover_data,
        height=height,
        width=width,
        layout=figure_layout,
        x_axis=xaxis_settings,
        y_axis=y_axis,
        color_axis=c_axis,
        traces_settings=traces_settings,
        output=output,
        # **kwargs:
        color=color_values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=shift_color_midpoint,
        **kwargs,
    )


def make_lof_bubble_plot(
    df: pd.Series | pd.DataFrame,
    normalize: bool = False,
    flip: bool = False,
    x_col: Optional[str] = "tpc",
    y_col: Optional[str] = "piece",
    duration_column: str = "duration_qb",
    title="Pitch class durations",
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 1500,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
    **kwargs,
):
    """Like :func:`make_bubble_plot` but coloring the bubbles along the Line of Fifths.
    Expecting a long format DataFrame/Series with two index levels where the first level groups pitch class
    distributions: Pitch classes are the second index level and the distribution values are contained in the Series
    or the first column. Additional columns may serve, e.g. to add more hover_data fields (by passing the column name(s)
    as keyword argument 'hover_data'.
    """
    df = df.reset_index()
    fifths = df[x_col].to_list()
    x_vals = sorted(set(fifths))
    x_names = ms3.fifths2name(x_vals)
    xaxis_settings = dict(tickvals=x_vals, ticktext=x_names)
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    df["pitch class"] = ms3.fifths2name(fifths)
    if hover_data is None:
        hover_data = []
    elif isinstance(hover_data, str):
        hover_data = [hover_data]
    hover_data.append("pitch class")
    color_col = x_col
    return make_bubble_plot(
        df=df,
        normalize=normalize,
        flip=flip,
        x_col=x_col,
        y_col=y_col,
        duration_column=duration_column,
        title=title,
        labels=labels,
        hover_data=hover_data,
        width=width,
        height=height,
        layout=layout,
        x_axis=xaxis_settings,
        y_axis=y_axis,
        color_axis=color_axis,
        traces_settings=traces_settings,
        output=output,
        # **kwargs:
        color=color_col,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=2,
        **kwargs,
    )


def make_scatter_plot(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_cols: Optional[str | Iterable[str]] = None,
    group_modes: Iterable[GroupMode] = (
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    ),
    title: Optional[str] = None,
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
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
    plot_settings = make_plot_settings(
        df=df,
        x_col=x_col,
        y_col=y_col,
        group_cols=group_cols,
        group_modes=group_modes,
        title=title,
        labels=labels,
        hover_data=hover_data,
        height=height,
        width=width,
    )
    fig = px.scatter(
        df,
        **plot_settings,
        **kwargs,
    )
    update_figure_layout(
        fig=fig,
        layout=layout,
        x_axis=x_axis,
        y_axis=y_axis,
        color_axis=color_axis,
        traces_settings=traces_settings,
    )
    if output is not None:
        write_image(fig=fig, filename=output, width=width, height=height)
    return fig


def plot_pitch_class_distribution(
    df: pd.DataFrame,
    pitch_column="tpc",
    duration_column="duration_qb",
    title="Pitch class distribution",
    fifths_transform=ms3.fifths2name,
    width=None,
    height=None,
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
    return make_lof_bar_plot(
        df=bar_data,
        x_col=x_col,
        y_col=y_col,
        labels=labels,
        title=title,
        fifth_transform=fifths_transform,
        width=width,
        height=height,
        output=output,
    )


def update_figure_layout(
    fig: go.Figure,
    layout: Optional[dict] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    **kwargs,
):
    figure_layout = dict(STD_LAYOUT)
    if layout is not None:
        figure_layout.update(layout)
    if len(kwargs) > 0:
        figure_layout.update(kwargs)
    fig.update_layout(**figure_layout)
    xaxis_settings = dict(Y_AXIS)
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    fig.update_xaxes(**xaxis_settings)

    yaxis_settings = dict(X_AXIS)
    if y_axis is not None:
        yaxis_settings.update(y_axis)
    fig.update_yaxes(**yaxis_settings)

    if color_axis is not None:
        fig.update_coloraxes(color_axis)

    if traces_settings is not None:
        fig.update_traces(traces_settings)


def write_image(
    fig: go.Figure,
    filename: str,
    directory: Optional[str] = None,
    format=None,
    scale=None,
    width=None,
    height=None,
    validate=True,
):
    """
    Convert a figure to a static image and write it to a file.

    Args:
        fig:
            Figure object or dict representing a figure

        file: str or writeable
            A string representing a local file path or a writeable object
            (e.g. a pathlib.Path object or an open file descriptor)

        format: str or None
            The desired image format. One of
              - 'png'
              - 'jpg' or 'jpeg'
              - 'webp'
              - 'svg'
              - 'pdf'
              - 'eps' (Requires the poppler library to be installed and on the PATH)

            If not specified and `file` is a string then this will default to the
            file extension. If not specified and `file` is not a string then this
            will default to:
                - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"
                - `plotly.io.orca.config.default_format` if engine is "orca"

        width: int or None
            The width of the exported image in layout pixels. If the `scale`
            property is 1.0, this will also be the width of the exported image
            in physical pixels.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"
                - `plotly.io.orca.config.default_width` if engine is "orca"

        height: int or None
            The height of the exported image in layout pixels. If the `scale`
            property is 1.0, this will also be the height of the exported image
            in physical pixels.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"
                - `plotly.io.orca.config.default_height` if engine is "orca"

        scale: int or float or None
            The scale factor to use when exporting the figure. A scale factor
            larger than 1.0 will increase the image resolution with respect
            to the figure's layout pixel dimensions. Whereas as scale factor of
            less than 1.0 will decrease the image resolution.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"
                - `plotly.io.orca.config.default_scale` if engine is "orca"

        validate: bool
            True if the figure should be validated before being converted to
            an image, False otherwise.
    """
    fname, fext = os.path.splitext(filename)
    has_allowed_extension = fext.lstrip(".") in AVAILABLE_FIGURE_FORMATS
    if format is None and has_allowed_extension:
        output_filename = filename
    else:
        if format is None:
            format = get_setting("default_figure_format")
        output_filename = f"{filename}.{format.lstrip('.')}"
    if directory is None:
        folder, filename = os.path.split(output_filename)
        if not folder:
            folder = get_setting("default_figure_path")
        folder = resolve_path(folder)
        output_filepath = os.path.join(folder, output_filename)
    else:
        output_filepath = os.path.join(directory, output_filename)
    if width is None:
        width = get_setting("default_figure_width")
    if height is None:
        height = get_setting("default_figure_height")
    fig.write_image(
        file=output_filepath,
        width=width,
        height=height,
        scale=scale,
        validate=validate,
    )
