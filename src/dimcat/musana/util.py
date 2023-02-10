from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns

# ===================================
# generic n-gram                     |
# ===================================


def get_n_grams(sequence: Sequence[object], n: int) -> np.ndarray:
    """
    Transform a sequence (list of objs) to a list of n-grams.
    :param sequence:
    :param n:
    :return:
    """
    transitions = np.array(
        [
            ["_".join(map(str, sequence[i - (n - 1) : i])), str(sequence[i])]
            for i in range(n - 1, len(sequence))
        ]
    )
    return transitions


def get_transition_matrix(n_grams: np.ndarray) -> pd.DataFrame:
    """
    Transform the n-gram np-array to a transition matrix dataframe
    :param n_grams:
    :return:
    """
    contexts, targets = np.unique(n_grams[:, 0]), np.unique(n_grams[:, 1])
    transition_matrix = pd.DataFrame(0, columns=targets, index=contexts)
    for i, n_gram in enumerate(n_grams):
        context, target = n_gram[0], n_gram[1]
        transition_matrix.loc[context, target] += 1
        # print(transition_matrix)
    return transition_matrix


def determine_era_based_on_year(year) -> str:
    if 0 < year < 1650:
        return "Renaissance"

    elif 1649 < year < 1759:
        return "Baroque"

    elif 1758 < year < 1819:
        return "Classical"

    elif 1819 < year < 1931:
        return "Romantic"


MAJOR_MINOR_KEYS_Dict = {
    "A": "major",
    "B": "major",
    "C": "major",
    "D": "major",
    "E": "major",
    "F": "major",
    "G": "major",
    "A#": "major",
    "B#": "major",
    "C#": "major",
    "D#": "major",
    "E#": "major",
    "F#": "major",
    "G#": "major",
    "Ab": "major",
    "Bb": "major",
    "Cb": "major",
    "Db": "major",
    "Eb": "major",
    "Fb": "major",
    "Gb": "major",
    "a": "minor",
    "b": "minor",
    "c": "minor",
    "d": "minor",
    "e": "minor",
    "f": "minor",
    "g": "minor",
    "a#": "minor",
    "b#": "minor",
    "c#": "minor",
    "d#": "minor",
    "e#": "minor",
    "f#": "minor",
    "g#": "minor",
    "ab": "minor",
    "bb": "minor",
    "cb": "minor",
    "db": "minor",
    "eb": "minor",
    "fb": "minor",
    "gb": "minor",
}


# ===================================
# seaborn plot                          |
# ===================================


def set_palette_6():
    sns.set()
    sns.set_style("white")
    platte = sns.set_palette(
        sns.color_palette(
            ["#046586", "#28A91A", "#C9A77C", "#F4A016", "#F6BBC6", "#E71F19"]
        )
    )

    return platte


def set_plot_style_palette_4():
    sns.set()
    sns.set_style("white")
    sns.set_palette(sns.color_palette(["#046586", "#28A9A1", "#F4A016", "#F6BBC6"]))


def set_platte_10():
    sns.set()
    sns.set_style("white")
    platte = sns.set_palette(
        sns.color_palette(
            [
                "#255459",
                "#4D8C8C",
                "#BFAA3F",
                "#FFB419",
                "#BF491F",
                "#BF847E",
                "#F2AF88",
                "#F2E3B6",
                "#0F4459",
                "#141A26",
            ]
        )
    )
    return platte


def set_platte_25():
    sns.set()
    sns.set_style("white")
    platte = sns.set_palette(
        sns.color_palette(
            [
                "#BF4417",
                "#D97C2B",
                "#F2B544",
                "#223A59",
                "#0B2559",
                "#6393A6",
                "#BF785E",
                "#F2CBBD",
                "#A65B4B",
                "#733B36",
                "#59202B",
                "#A68195",
                "#F2B138",
                "#F29E38",
                "#F24E29",
                "#BF491F",
                "#FFB419",
                "#BFAA3F",
                "#4D8C8C",
                "#255459",
                "#707317",
                "#3A4010",
                "#B3E0F2",
                "#35648C",
                "#1E3859",
            ]
        )
    )
    return platte
