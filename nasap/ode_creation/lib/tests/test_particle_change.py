import numpy as np
import pytest

from nasap.ode_creation import (calc_consumed_count, calc_particle_change,
                                calc_produced_count)
from nasap.ode_creation.reaction_class import Reaction


def test_1_to_1():
    # A -> B  (k)
    assemblies = [0, 1]
    reaction = Reaction(
        reactants=[0], products=[1], reaction_kind=0, 
        duplicate_count=1)
    np.testing.assert_array_equal(
        calc_consumed_count(assemblies, [reaction]), [[1, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assemblies, [reaction]), [[0, 1]])
    np.testing.assert_array_equal(
        calc_particle_change(assemblies, [reaction]), [[-1, 1]])


def test_2_to_1():
    # A + A -> B  (k)
    assemblies = [0, 1]
    reaction = Reaction(
        reactants=[0, 0], products=[1], reaction_kind=0, 
        duplicate_count=2)
    np.testing.assert_array_equal(
        calc_consumed_count(assemblies, [reaction]), [[2, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assemblies, [reaction]), [[0, 1]])
    np.testing.assert_array_equal(
        calc_particle_change(assemblies, [reaction]), [[-2, 1]])


def test_1_to_2():
    # A -> B + B  (k)
    assemblies = [0, 1]
    reaction = Reaction(
        reactants=[0], products=[1, 1], reaction_kind=0, 
        duplicate_count=1)
    np.testing.assert_array_equal(
        calc_consumed_count(assemblies, [reaction]), [[1, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assemblies, [reaction]), [[0, 2]])
    np.testing.assert_array_equal(
        calc_particle_change(assemblies, [reaction]), [[-1, 2]])


def test_2_to_2():
    # A + A -> B + B  (k)
    assemblies = [0, 1]
    reaction = Reaction(
        reactants=[0, 0], products=[1, 1], reaction_kind=0, 
        duplicate_count=2)
    np.testing.assert_array_equal(
        calc_consumed_count(assemblies, [reaction]), [[2, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assemblies, [reaction]), [[0, 2]])
    np.testing.assert_array_equal(
        calc_particle_change(assemblies, [reaction]), [[-2, 2]])
    

def test_reversible():
    # A -> 2B  (k1)
    # 2B -> A  (k2)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0], products=[1, 1], reaction_kind=0, 
            duplicate_count=1),
        Reaction(
            reactants=[1, 1], products=[0], reaction_kind=1, 
            duplicate_count=2)
        ]
    np.testing.assert_array_equal(
        calc_consumed_count(assemblies, reactions), [[1, 0], [0, 2]])
    np.testing.assert_array_equal(
        calc_produced_count(assemblies, reactions), [[0, 2], [1, 0]])
    np.testing.assert_array_equal(
        calc_particle_change(assemblies, reactions), [[-1, 2], [1, -2]])


if __name__ == '__main__':
    pytest.main(['-v', __file__])
