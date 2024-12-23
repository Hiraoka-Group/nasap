from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from nasap.ode_creation.reaction_class import Reaction

# n: number of assemblies
# m: number of reactions
# k: number of reaction kinds

def calc_particle_change(
        number_of_assems: int, reactions: Iterable[Reaction]
        ) -> npt.NDArray:  # shape (m, n)
    consumed_count = calc_consumed_count(number_of_assems, reactions)
    produced_count = calc_produced_count(number_of_assems, reactions)
    return produced_count - consumed_count


def calc_consumed_count(
        number_of_assems: int, reactions: Iterable[Reaction]
        ) -> npt.NDArray:  # shape (m, n)
    reactions = list(reactions)
    consumed_count = np.zeros(
        (len(reactions), number_of_assems), dtype=int)
    for i, reaction in enumerate(reactions):
        for assem in reaction.reactants:
            consumed_count[i, assem] += 1
    return consumed_count


def calc_produced_count(
        number_of_assems: int, reactions: Iterable[Reaction]
        ) -> npt.NDArray:  # shape (m, n)
    reactions = list(reactions)
    produced_count = np.zeros(
        (len(reactions), number_of_assems), dtype=int)
    for i, reaction in enumerate(reactions):
        for assem in reaction.products:
            produced_count[i, assem] += 1
    return produced_count
