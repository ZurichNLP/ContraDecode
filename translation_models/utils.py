from typing import List


def batch(input: List, batch_size: int):
    l = len(input)
    for ndx in range(0, l, batch_size):
        yield input[ndx:min(ndx + batch_size, l)]
