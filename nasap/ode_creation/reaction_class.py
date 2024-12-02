from collections.abc import Iterable


class Reaction:
    def __init__(
            self, reactants: Iterable[int], products: Iterable[int],
            reaction_kind: int, duplicate_count: int
            ) -> None:
        # If two A assemblies are consumed, then two As should be in the list.
        self._reactants = tuple(reactants)
        # If two B assemblies are produced, then two Bs should be in the list.
        self._products = tuple(products)
        self._duplicate_count = duplicate_count
        self._reaction_kind = reaction_kind

    @property
    def reactants(self) -> tuple[int, ...]:
        return self._reactants

    @property
    def products(self) -> tuple[int, ...]:
        return self._products

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def reaction_kind(self) -> int:
        return self._reaction_kind

    def __hash__(self) -> int:
        return hash((
            self._reactants, self._products, self._duplicate_count,
            self._reaction_kind))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reaction):
            return False
        return (
            self._reactants == other._reactants and
            self._products == other._products and
            self._duplicate_count == other._duplicate_count and
            self._reaction_kind == other._reaction_kind
            )