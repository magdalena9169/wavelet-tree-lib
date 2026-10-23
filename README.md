# wavelet-tree-lib

Stores a sequence of non-negative integers and answers `rank`, `select`, and
range quantile queries in O(log |Σ|) time using a Wavelet Tree over a balanced
binary hierarchy of bit vectors.

```python
from wavelet_tree_lib import WaveletTree

seq = [3, 1, 4, 1, 5, 9, 2, 6]
alphabet = sorted(set(seq))
tree = WaveletTree(seq, alphabet)

tree.rank(1, 8)            # 2  — count of 1s in seq[0:8]
tree.select(1, 2)           # 3  — 0-indexed position of the 2nd occurrence of 1
tree.range_quantile(0, 8, 0)  # 1  — smallest value in seq[0:8]
```

## Why this exists

You have a static sequence over a moderate integer alphabet and need to answer
rank (how many times does symbol `s` appear in a prefix), select (where is the
k-th occurrence of `s`), and range quantile (what is the k-th smallest value in
a sub-range) — all without scanning the sequence per query. A wavelet tree
gives all three at O(log |Σ|) per query at the cost of O(n log |Σ|) space.

The trade-off taken here: the alphabet must be supplied explicitly at
construction (it is not inferred from the data). This makes the symbol order
and the tree shape deterministic and inspectable via `tree.alphabet`, at the
cost of requiring the caller to know the alphabet up front.

## Edge cases you will hit

- `rank` of a symbol not in the alphabet returns `0` rather than raising. This
  is the natural meaning of "count of a thing that does not appear."
- `select` for a symbol not in the alphabet, or for a `k` larger than the
  number of occurrences, returns `-1`. This lets callers loop without catching
  exceptions.
- `range_quantile(l, r, k)` uses **0-indexed `k`** and a half-open range
  `[l, r)`. `k == 0` is the minimum. `k` must satisfy `0 <= k < r - l`.
- `bool` is rejected in both the sequence and the alphabet, even though
  `isinstance(True, int)` is `True` in Python. Mixing `True`/`False` with
  integers in a bit-packed structure is a source of silent bugs, so the
  boundary check rejects them early.
- An empty sequence with an empty alphabet constructs an empty tree; all
  quantile queries on it raise `ValueError`.
