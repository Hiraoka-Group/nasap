from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from .lib import calc_consumed_count, calc_produced_count
from .reaction_class import Reaction, convert_reaction_to_use_index

_T_co = TypeVar('_T_co', covariant=True)  # type of assembly
_S_co = TypeVar('_S_co', covariant=True)  # type of reaction kind


class OdeRhs(Protocol):
    def __call__(
            self, t: float, y: npt.NDArray, 
            log_k_of_rxn_kinds: npt.NDArray,
            ) -> npt.NDArray:
        ...


def create_ode_rhs(
        assemblies: Sequence[_T_co], 
        reaction_kinds: Sequence[_S_co],
        reactions: Iterable[Reaction[_T_co, _S_co]],
        ) -> OdeRhs:
    """Create a function that calculates the right-hand side of the ODE.

    Parameters
    ----------
    assemblies : Sequence[_T_co]
        Assembly IDs. Can be any hashable type. The order of the IDs
        will be used to determine the order of `y` parameter in the
        returned function.
    reaction_kinds : Sequence[_S_co]
        Reaction kind IDs. Can be any hashable type. The order of the IDs
        will be used to determine the order of `log_k_of_rxn_kinds`
        parameter in the returned function.
    reactions : Iterable[Reaction[_T_co, _S_co]]
        Reactions to be included in the ODE.
    """
    assem_id_to_index = {assem: i for i, assem in enumerate(assemblies)}
    reaction_kind_id_to_index = {
        kind: i for i, kind in enumerate(reaction_kinds)}
    reaction_by_index = [
        convert_reaction_to_use_index(
            reaction, assem_id_to_index, reaction_kind_id_to_index)
        for reaction in reactions]
    
    # n: number of assemblies
    # m: number of reactions
    # k: number of reaction kinds
    number_of_assems = len(assemblies)
    assem_indices = np.arange(number_of_assems)
    reaction_kind_indices = np.arange(len(reaction_kinds))
    
    # shape: (m, k), dtype: bool
    rxn_to_kind = np.array([
        [reaction.reaction_kind == kind for kind in reaction_kinds]
        for reaction in reaction_by_index])

    # shape: (m,)
    coefficients = np.array([
        reaction.duplicate_count for reaction in reaction_by_index])
    
    # shape: (m, n)
    consumed = calc_consumed_count(
        number_of_assems, reaction_by_index)  # shape: (m, n)
    produced = calc_produced_count(
        number_of_assems, reaction_by_index)  # shape: (m, n)
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
