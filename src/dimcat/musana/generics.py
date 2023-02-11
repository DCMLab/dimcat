from __future__ import annotations

import collections
from dataclasses import dataclass

import pandas as pd
from dimcat._typing import ST, Sequential

# set typing, Sequential of tuples:


@dataclass
class TransitionMatrix:
    n_grams: ST

    def __repr__(self) -> str:
        return f"TransitionMatrix(n_grams={self.n_grams})"

    # def get_label_counts(self) -> collections.Counter[str]:
    #     label_counts = collections.Counter()
    #     for n_gram in self.n_grams:
    #         source = '_'.join(n_gram[:-1])
    #         target = n_gram[-1:]
    #         label_counts[source] += 1
    #         label_counts[target] += 1
    #     return label_counts

    def get_source_label_counts(self) -> collections.Counter[str]:
        label_counts = collections.Counter()
        for n_gram in self.n_grams:
            source = "_".join(n_gram[:-1])
            label_counts[source] += 1
        return label_counts

    def get_target_label_counts(self) -> collections.Counter[str]:
        label_counts = collections.Counter()
        for n_gram in self.n_grams:
            target = "".join(n_gram[-1:])
            label_counts[target] += 1
        return label_counts

    def create_matrix(self, probability: bool = True) -> pd.DataFrame:
        if (len(n_gram) == 2 for n_gram in self.n_grams):
            unpack_tuples_list = list(map(lambda x: x[0], self.n_grams._seq[:])) + [
                self.n_grams._seq[-1][-1]
            ]
            count = collections.Counter(unpack_tuples_list)
            sorted_source_labels = [x for x, number in count.most_common()]
            sorted_target_labels = sorted_source_labels

        else:
            source_label_counts = self.get_source_label_counts()
            target_label_counts = self.get_target_label_counts()
            sorted_source_labels = [
                label for label, count in source_label_counts.most_common()
            ]
            sorted_target_labels = [
                label for label, count in target_label_counts.most_common()
            ]

        # Create an empty Pandas DataFrame with the sorted labels as the index and columns
        transition_matrix = pd.DataFrame(
            0, index=sorted_source_labels, columns=sorted_target_labels
        )

        for n_gram in self.n_grams:
            source = "_".join(n_gram[:-1])
            target = n_gram[-1:]
            transition_matrix.loc[source, target] += 1

        if probability:
            # normalize the DataFrame by dividing each element by the sum of its row
            transition_prob = transition_matrix.div(
                transition_matrix.sum(axis=1), axis=0
            )
            transition_prob = transition_prob.fillna(0)
            return transition_prob
        return transition_matrix


if __name__ == "__main__":
    test_seq = [1, 1, 1, 2, 2, 3, 4, 5, 6]
    sequential = Sequential.from_sequence(sequence=test_seq)
    # bigrams = sequential.get_n_grams(n=2)
    # print(bigrams)
    print(sequential[6])
