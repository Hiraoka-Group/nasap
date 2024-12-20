from collections.abc import Iterable
from typing import Protocol

import numpy as np
import numpy.typing as npt

from nasap.ode_creation.reaction_class import Reaction

from .lib import calc_consumed_count, calc_produced_count


class OdeRhs(Protocol):
    def __call__(
            self, t: float, y: npt.NDArray, 
            log_k_of_rxn_kinds: npt.NDArray,
            ) -> npt.NDArray:
        ...


def create_ode_rhs(
        assems: Iterable[int], 
        reactions: Iterable[Reaction],
        reaction_kinds: Iterable[int],
        ) -> OdeRhs:
    # n: number of assemblies
    # m: number of reactions
    # k: number of reaction kinds
    assems = list(assems)
    reactions = list(reactions)
    reaction_kinds = list(reaction_kinds)
    
    # shape: (m, k), dtype: bool
    rxn_to_kind = np.array([
        [reaction.reaction_kind == kind for kind in reaction_kinds]
        for reaction in reactions])

    # shape: (m,)
    coefficients = np.array([
        reaction.duplicate_count for reaction in reactions])
    
    # shape: (m, n)
    consumed = calc_consumed_count(assems, reactions)  # shape: (m, n)
    produced = calc_produced_count(assems, reactions)  # shape: (m, n)
    change = produced - consumed  # shape: (m, n)
    
    # Note: `ode_rhs` should be as efficient as possible
    # to be used in `solve_ivp` for large-scale simulations
    def ode_rhs(
            t: float, y: npt.NDArray, log_k_of_rxn_kinds: npt.NDArray,
            ) -> npt.NDArray:  # shape: (n,)
        # t: time
        # y: concentrations of assemblies, shape: (n,)
        # log_k_of_rxn_kinds: log(k) of each reaction kind, shape: (k,)

        k_of_rxn_kinds = 10.0**log_k_of_rxn_kinds  # shape: (k,)
        ks = rxn_to_kind @ k_of_rxn_kinds  # shape: (m,)

        # First, determine the rate of the reaction occurrences.
        # Let's call it "event rate".
        # One value for each reaction. Thus, shape: (m,)
        # event_rate = coefficient * k * product(reactant**consumed_count)
        event_rates = coefficients * ks * np.prod(y**consumed, axis=1)

        # Then, calculate the change in the concentration of each assembly.
        # This is exactly the right-hand side of the ODE.
        # One value for each assembly. Thus, shape: (n,)
        # rhs = sum(event_rate * change)
        rhs = np.sum(event_rates[:, None] * change, axis=0)

        return rhs
    
    return ode_rhs
