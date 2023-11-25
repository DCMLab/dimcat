from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, List, Optional

import ms3
import numpy as np
import pandas as pd
import seaborn as sns
from dimcat.base import FriendlyEnum, get_setting
from dimcat.utils import resolve_path
from kaleido.scopes.plotly import PlotlyScope
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import express as px
from plotly import graph_objects as go
from scipy.stats import entropy

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


def clean_axis_labels(*labels: str) -> dict:
    """Clean axis labels for Plotly plots by removing all underscores ('_') with spaces.

    Args:
        *labels: Labels to clean.

    Returns:
        A dictionary mapping the original to the cleaned labels.
    """
    default_labels = {"duration_qb": "duration in 𝅘𝅥"}
    result = {}
    for label in labels:
        if pd.isnull(label):
            continue
        if label in default_labels:
            cleaned_label = default_labels[label]
        else:
            cleaned_label = label.replace("_", " ")
        result[label] = cleaned_label
    return result


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
        hover_data=hover_data,
        height=height,
        width=width,
    )
    if group_cols is not None:
        update_plot_grouping_settings(plot_settings, group_cols, group_modes)
    if "x" not in plot_settings:
        plot_settings["x"] = x_col
    label_settings = clean_axis_labels(*df.columns)
    if labels is not None:
        label_settings.update(labels)
    plot_settings["labels"] = label_settings
    return plot_settings


def update_plot_grouping_settings(
    plot_settings: dict,
    group_cols: Optional[str | Iterable[str]] = None,
    group_modes: Iterable[GroupMode] = (
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    ),
):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(group_modes, str):
        group_modes = [group_modes]
    for group_col, group_mode in zip(group_cols, group_modes):
        setting_key = GROUPMODE2BAR_PLOT_SETTING[group_mode]
        if setting_key in plot_settings:
            raise ValueError(
                f"Trying to set {setting_key!r} to {group_col!r} but it is already set to "
                f"{plot_settings[setting_key]!r}."
            )
        plot_settings[setting_key] = group_col


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
    if "barmode" not in kwargs:
        kwargs[
            "barmode"
        ] = "group"  # Plotly's default: "relative" (meaning stacked); other option: "overlay"]
    if "text" not in kwargs and "proportion_%" in df.columns:
        kwargs["text"] = "proportion_%"
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
    df = df.reset_index()
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
    x_col="tpc",
    y_col="duration_qb",
    fifths_transform: Optional[Callable[[int], Any]] = ms3.fifths2name,
    x_names_col: Optional[str] = None,
    title=None,
    labels=None,
    hover_data: Optional[List[str]] = None,
    shift_color_midpoint: int = 0,
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
    df = df.reset_index()
    fifths = df[x_col].to_list()
    xaxis_settings = dict(
        gridcolor="lightgrey",
        zerolinecolor="grey",
        dtick=1,
        ticks="outside",
        tickcolor="black",
        minor=dict(dtick=6, gridcolor="grey", showgrid=True),
    )
    if x_names_col is not None:
        x_names = df[x_names_col].values
        xaxis_settings = dict(tickmode="array", tickvals=fifths, ticktext=x_names)
    elif fifths_transform is not None:
        x_values = sorted(set(fifths))
        x_names = list(map(fifths_transform, x_values))
        xaxis_settings = dict(tickmode="array", tickvals=x_values, ticktext=x_names)
    figure_layout = dict(showlegend=False)
    if layout is not None:
        figure_layout.update(layout)
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
        color=fifths,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=shift_color_midpoint,
        **kwargs,
    )


def make_lof_bubble_plot(
    df: pd.Series | pd.DataFrame,
    normalize: bool = False,
    flip: bool = False,
    fifths_col: Optional[str] = "tpc",
    y_col: Optional[str] = "piece",
    duration_column: str = "duration_qb",
    fifths_transform: Optional[Callable[[int], Any]] = ms3.fifths2name,
    x_names_col: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[dict] = None,
    hover_data: Optional[List[str]] = None,
    shift_color_midpoint: int = 0,
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
    fifths = df[fifths_col].to_list()
    xaxis_settings = dict()
    if x_names_col is not None:
        x_names = df[x_names_col].values
        xaxis_settings = dict(tickmode="array", tickvals=fifths, ticktext=x_names)
    elif fifths_transform is not None:
        x_values = sorted(set(fifths))
        x_names = list(map(fifths_transform, x_values))
        xaxis_settings = dict(tickmode="array", tickvals=x_values, ticktext=x_names)
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    if hover_data is None:
        hover_data = []
    elif isinstance(hover_data, str):
        hover_data = [hover_data]
    # df["pitch class"] = ms3.fifths2name(fifths)
    # hover_data.append("pitch class")
    color_col = fifths_col
    return make_bubble_plot(
        df=df,
        normalize=normalize,
        flip=flip,
        x_col=fifths_col,
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
        color_continuous_midpoint=shift_color_midpoint,
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


def make_transition_heatmap_plots(
    left_transition_matrix: pd.DataFrame,
    left_unigrams: pd.Series,
    right_transition_matrix: Optional[pd.DataFrame] = None,
    right_unigrams: Optional[pd.Series] = None,
    top: int = 30,
    two_col_width=12,
    frequencies: bool = False,
    fontsize=8,
    labelsize=12,
    top_margin=0.99,
    bottom_margin=0.10,
    right_margin=0.005,
    left_margin=0.085,
) -> MatplotlibFigure:
    """
    Adapted from https://zenodo.org/records/2764889/files/reproduce_ABC.ipynb?download=1 which is the Jupyter notebook
    accompanying Moss FC, Neuwirth M, Harasim D, Rohrmeier M (2019) Statistical characteristics of tonal harmony: A
    corpus study of Beethoven’s string quartets. PLOS ONE 14(6): e0217242. https://doi.org/10.1371/journal.pone.0217242

    Args:
        left_unigrams:
        right_unigrams:
        left_transition_matrix:
        right_transition_matrix:
        top:
        two_col_width:
        frequencies: If set to True, the values of the unigram Series are interpreted as normalized frequencies and
            are multiplied with 100 for display on the y-axis.

    """
    # set custom context for this plot
    with plt.rc_context(
        {
            # disable spines for entropy bars
            "axes.spines.top": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        }
    ):

        def make_gridspec(
            left,
            right,
        ):
            gridspec_ratio = [0.25, 2.0]
            hspace = None
            wspace = 0.0
            gs = gridspec.GridSpec(1, 2, width_ratios=gridspec_ratio)
            gs.update(
                left=left,
                right=right,
                wspace=wspace,
                hspace=hspace,
                bottom=bottom_margin,
                top=top_margin,
            )
            return gs

        def add_entropy_bars(
            unigrams,
            bigrams,
            axis,
        ):
            # settings for margins etc.
            barsize = [0.0, 0.7]
            s_min = pd.Series(
                (
                    bigrams.apply(lambda x: entropy(x, base=2), axis=1)
                    / np.log2(bigrams.shape[0])
                )[:top].values,
                index=[
                    i + f" ({str(round(fr * 100, 1))})" if frequencies else i
                    for i, fr in zip(bigrams.index, unigrams[:top].values)
                ],
            )
            ax = s_min.plot(kind="barh", ax=axis, color="k")

            # create a list to collect the plt.patches data
            totals_min = []

            # find the values and append to list
            for i in ax.patches:
                totals_min.append(round(i.get_width(), 2))

            for i, p in enumerate(ax.patches):
                axis.text(
                    totals_min[i] - 0.01,
                    p.get_y() + 0.3,
                    f"${totals_min[i]}$",
                    color="w",
                    fontsize=fontsize,
                    verticalalignment="center",
                    horizontalalignment="left",
                )
            axis.set_xlim(barsize)

            axis.invert_yaxis()
            axis.invert_xaxis()
            axis.set_xticklabels([])
            axis.tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,
                bottom=False,
                labelleft=True,
                labelsize=labelsize,
            )

        def add_heatmap(transition_value_matrix, axis, colormap):
            sns.heatmap(
                transition_value_matrix,
                annot=True,
                fmt=".1f",
                cmap=colormap,
                ax=axis,
                # vmin=vmin,
                # vmax=vmax,
                annot_kws={"fontsize": fontsize, "rotation": 60},
                cbar=False,
            )
            axis.set_yticks([])
            axis.tick_params(bottom=False)

        single_col_width = two_col_width / 2
        plot_two_sides = right_transition_matrix is not None
        if plot_two_sides:
            assert (
                right_unigrams is not None
            ), "right_unigrams must be provided if right_bigrams is provided"
            fig = plt.figure(figsize=(two_col_width, single_col_width))
            gs1 = make_gridspec(
                left=left_margin,
                right=0.5 - right_margin,
            )
        else:
            fig = plt.figure(figsize=(single_col_width, single_col_width))
            gs1 = make_gridspec(
                left=left_margin,
                right=1.0 - right_margin,
            )

        # LEFT-HAND SIDE

        ax1 = plt.subplot(gs1[0, 0])

        add_entropy_bars(
            left_unigrams,
            left_transition_matrix,
            ax1,
        )

        ax2 = plt.subplot(gs1[0, 1])

        add_heatmap(
            left_transition_matrix[left_transition_matrix > 0].iloc[
                :top, :top
            ],  # only display non-zero values
            axis=ax2,
            colormap="Blues",
        )

        # RIGHT-HAND SIDE

        plot_two_sides = right_transition_matrix is not None
        if plot_two_sides:
            assert (
                right_unigrams is not None
            ), "right_unigrams must be provided if right_bigrams is provided"

            gs2 = make_gridspec(
                left=0.5 + left_margin,
                right=1.0 - right_margin,
            )

            ax3 = plt.subplot(gs2[0, 0])
            add_entropy_bars(
                right_unigrams,
                right_transition_matrix,
                ax3,
            )

            ax4 = plt.subplot(gs2[0, 1])
            add_heatmap(
                right_transition_matrix[right_transition_matrix > 0].iloc[:top, :top],
                axis=ax4,
                colormap="Reds",
            )

        fig.align_labels()
    return fig


def plot_pitch_class_distribution(
    df: pd.DataFrame,
    pitch_column="tpc",
    duration_column="duration_qb",
    title="Pitch class distribution",
    fifths_transform=ms3.fifths2name,
    shift_color_midpoint: int = 2,
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
        fifths_transform=fifths_transform,
        shift_color_midpoint=shift_color_midpoint,
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
