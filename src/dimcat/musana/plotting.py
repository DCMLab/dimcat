import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def transition_matrix_heatmap(
    transition_matrix: pd.DataFrame, ax: plt.Axes, view_top_n: int, cmap: str
) -> None:
    if transition_matrix.size > 0:
        # view the top n rank chord symbol
        transition_matrix_to_view = transition_matrix.iloc[:view_top_n, :view_top_n]

        sns.heatmap(
            data=100 * transition_matrix_to_view,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
            cmap=cmap,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": "xx-small"},
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", rotation=45)
