from __future__ import annotations

import logging
from typing import Optional

import ms3
import plotly.express as px
import plotly.graph_objs as go

logger = logging.getLogger(__name__)

COLOR_SCALE_SETTINGS = dict(
    color_continuous_scale="RdBu_r", color_continuous_midpoint=2
)
TRACES_SETTINGS = dict(marker_line_color="black")
Y_AXIS = dict()
X_AXIS = dict()


def tpc_bubbles(
    df,
    normalize=True,
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
        normalize:
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
    if normalize:
        df = (
            df.groupby(level=0, group_keys=False)
            .apply(lambda S: S / S.sum())
            .reset_index()
        )
        title = "Normalized " + title
    else:
        df = df.reset_index()
    df["pitch class"] = ms3.fifths2name(df.tpc)
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size="duration in ♩",
        color=color_col,
        hover_data=["pitch class"],
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
