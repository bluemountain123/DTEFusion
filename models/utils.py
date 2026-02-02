from typing import Iterator, Sequence, TypeVar

T = TypeVar('T', bound=Sequence)

def slice_window(it: T, size: int) -> Iterator[T]:
    for i in range(len(it) - size + 1):
        yield it[i : i + size]
