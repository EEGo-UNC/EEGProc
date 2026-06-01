from collections.abc import Sequence

def _ensure_tuple(value, n_items: int, name: str):
    if isinstance(value, Sequence) and not isinstance(value, str):
        value = tuple(value)
        if len(value) != n_items:
            raise ValueError(
                f"{name} must have length {n_items}, got length {len(value)}."
            )
        return value

    return tuple(value for _ in range(n_items))


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result