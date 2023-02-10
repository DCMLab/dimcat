from __future__ import annotations

import collections
import typing
from dataclasses import dataclass

import pandas as pd

# set typing, Sequential of tuples:
T = typing.TypeVar("T")


@dataclass
class Sequential(typing.Generic[T]):
    _seq: typing.Sequence[T]

    def __len__(self):
        return len(self._seq)

    def __getitem__(self, position_num):
        if isinstance(position_num, int):
            return self._seq[position_num]

    @classmethod
    def from_sequence(cls, sequence: typing.Sequence[T]) -> typing.Self:
        if len(sequence) > 0:
            first_object_type = type(sequence[0])
            type_check_pass = all((type(x) == first_object_type for x in sequence))
            if not type_check_pass:
                raise TypeError()
        return cls(_seq=sequence)

    @classmethod
    def join(cls, sequentials: typing.Sequence[Sequential]) -> Sequential:
        """
        A method to join multiple sequentials.
        :param sequentials:
        :return:
        """
        joined_seq = sum([x._seq for x in sequentials], [])
        sequential = cls.from_sequence(joined_seq)
        return sequential

    def map(
        self,
        operation: typing.Callable[
            [
                T,
            ],
            typing.Any,
        ],
    ) -> Sequential:
        new_seq = list(map(operation, self._seq))
        sequential = Sequential(_seq=new_seq)
        return sequential

    def filter_by_condition(
        self,
        condition: typing.Callable[
            [
                T,
            ],
            bool,
        ],
    ) -> Sequential:
        sequence = [x for x in self._seq if condition(x)]
        sequential = self.from_sequence(sequence=sequence)
        return sequential

    def get_n_grams(self, n: int) -> ST:
        """
        Returns a Squential type of object in which each transition is expressed as a tuple.
        :param n:
        :return:
        """
        length = len(self._seq)
        n_grams = [tuple(self._seq[i : i + n]) for i in range(length - n + 1)]
        n_grams = Sequential.from_sequence(sequence=n_grams)
        return n_grams

    def get_transition_matrix(self, probability: bool) -> pd.DataFrame:
        """
        Create a transition matrix
        :param n:
        :param probability:
        :return:
        """

        count = collections.Counter(self._seq)
        top_common_objects = [x for x, number in count.most_common()]

        transition_matrix = pd.DataFrame(
            0, columns=top_common_objects, index=top_common_objects
        )

        bigrams = self.get_n_grams(n=2)
        for bigram in bigrams._seq:
            source, target = bigram
            transition_matrix.loc[str(source), str(target)] += 1

        if probability:
            transition_prob = transition_matrix.divide(
                transition_matrix.sum(axis=1), axis=0
            )
            return transition_prob
        return transition_matrix


ST = Sequential[typing.Tuple]


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
