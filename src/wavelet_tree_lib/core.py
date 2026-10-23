"""Wavelet Tree over an integer alphabet.

Design decisions
----------------
- The alphabet is a set of non-negative integers provided at construction time.
  We do not auto-discover it from the sequence so that ``len(tree.alphabet)``
  and the symbol order are deterministic and visible to the caller.
- The tree is balanced on the *sorted* alphabet. Each internal node splits
  its alphabet range at the midpoint; symbols in the lower half go to the
  left child, the rest to the right. This gives O(log |Σ|) depth regardless of
  symbol frequency, which is the property we need for bounded query cost.
- Ranks are materialized as a Python ``list[int]`` of cumulative counts in the
  node's bit vector. We use the convention ``rank(node, b, i)`` = number of
  1-bits in ``b[0:i]`` (0-indexed, half-open). The zero entry at index 0 is
  the identity element and keeps indexing arithmetic simple.
- ``select`` walks from a leaf up to the root, translating a rank in a child
  subtree into a rank at the parent. This is the standard formulation and lets
  us reuse the precomputed rank arrays at each level.
- ``range_quantile`` is recursive: at each node it counts 0-bits in the query
  sub-range; if the desired order statistic falls within that count it recurses
  left, otherwise it subtracts and recurses right.
- Nodes store their alphabet range as ``(lo, hi)`` inclusive. A leaf has
  ``lo == hi``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class BitVector:
    """A bit vector with precomputed prefix ranks for O(1) rank queries.

    We store bits as a list of 0/1 ints and a parallel ``ranks`` list where
    ``ranks[i]`` is the count of 1-bits in ``bits[0:i]``. The extra integer
    field per slot is the classic time/space trade-off; for the sequence sizes
    this library targets (thousands to low millions of symbols) it is faster
    in CPython than recomputing or using ``int.bit_count`` on slices.
    """

    __slots__ = ("bits", "ranks", "_ones")

    def __init__(self, bits: List[int]) -> None:
        self.bits = bits
        ranks = [0] * (len(bits) + 1)
        c = 0
        for i, b in enumerate(bits):
            if b:
                c += 1
            ranks[i + 1] = c
        self.ranks = ranks
        self._ones = c

    def rank1(self, i: int) -> int:
        """Count of 1-bits in ``bits[0:i]`` (half-open, 0-indexed)."""
        if i < 0:
            raise ValueError(f"index must be non-negative, got {i}")
        if i > len(self.bits):
            raise ValueError(
                f"index {i} out of range for length {len(self.bits)}"
            )
        return self.ranks[i]

    def rank0(self, i: int) -> int:
        """Count of 0-bits in ``bits[0:i]`` (half-open, 0-indexed)."""
        return i - self.rank1(i)

    def select1(self, k: int) -> int:
        """Position of the k-th 1-bit (1-indexed), or -1 if it does not exist."""
        if k < 1:
            raise ValueError(f"select1 requires k >= 1, got {k}")
        if k > self._ones:
            return -1
        # Binary search over ranks for the smallest index i with ranks[i] == k
        # and bits[i-1] == 1.
        lo, hi = 0, len(self.bits)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ranks[mid] >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo - 1

    def select0(self, k: int) -> int:
        """Position of the k-th 0-bit (1-indexed), or -1 if it does not exist."""
        if k < 1:
            raise ValueError(f"select0 requires k >= 1, got {k}")
        zeros = len(self.bits) - self._ones
        if k > zeros:
            return -1
        lo, hi = 0, len(self.bits)
        while lo < hi:
            mid = (lo + hi) // 2
            # number of zeros in bits[0:mid]
            z = mid - self.ranks[mid]
            if z >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo - 1

    def __len__(self) -> int:
        return len(self.bits)


class _Node:
    """Internal node or leaf of the wavelet tree."""

    __slots__ = ("lo", "hi", "bv", "left", "right")

    def __init__(
        self,
        lo: int,
        hi: int,
        bv: Optional[BitVector],
        left: Optional["_Node"],
        right: Optional["_Node"],
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.bv = bv
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.lo == self.hi


class WaveletTree:
    """A wavelet tree over a sequence of non-negative integers.

    The tree answers:
    - ``rank(symbol, i)``: count of ``symbol`` in ``seq[0:i]``.
    - ``select(symbol, k)``: index of the k-th occurrence of ``symbol`` (0-indexed),
      or -1 if there are fewer than ``k`` occurrences.
    - ``range_quantile(l, r, k)``: the k-th smallest value in ``seq[l:r]``
      (0-indexed, half-open; ``k`` is 0-indexed).

    Construction is O(n log |Σ|) time and space. Queries are O(log |Σ|) time.
    """

    __slots__ = ("_root", "_alphabet", "_n")

    def __init__(self, sequence: List[int], alphabet: List[int]) -> None:
        if len(alphabet) == 0:
            if len(sequence) != 0:
                raise ValueError(
                    "empty alphabet cannot index a non-empty sequence"
                )
            self._root = None
            self._alphabet = []
            self._n = 0
            return

        # Validate once, at the boundary, so internal code can assume validity.
        seen = set()
        for a in alphabet:
            if not isinstance(a, int) or isinstance(a, bool):
                raise TypeError(f"alphabet entries must be int, got {type(a)!r}")
            if a < 0:
                raise ValueError(f"alphabet entries must be non-negative, got {a}")
            if a in seen:
                raise ValueError(f"duplicate alphabet symbol {a}")
            seen.add(a)

        alpha_sorted = sorted(alphabet)
        alpha_set = seen
        for s in sequence:
            if not isinstance(s, int) or isinstance(s, bool):
                raise TypeError(f"sequence entries must be int, got {type(s)!r}")
            if s not in alpha_set:
                raise ValueError(f"symbol {s} not in alphabet")

        self._alphabet = alpha_sorted
        self._n = len(sequence)
        lo = alpha_sorted[0]
        hi = alpha_sorted[-1]
        self._root = self._build(sequence, lo, hi, alpha_sorted)

    def _build(
        self,
        sequence: List[int],
        lo: int,
        hi: int,
        alpha_sorted: List[int],
    ) -> Optional[_Node]:
        if lo > hi:
            return None
        if lo == hi:
            # Leaf: no bit vector needed (all symbols identical).
            return _Node(lo, hi, None, None, None)

        mid = (lo + hi) // 2
        bits: List[int] = [0] * len(sequence)
        left_seq: List[int] = []
        right_seq: List[int] = []
        for i, s in enumerate(sequence):
            if s <= mid:
                left_seq.append(s)
            else:
                bits[i] = 1
                right_seq.append(s)

        bv = BitVector(bits)
        left = self._build(left_seq, lo, mid, alpha_sorted)
        right = self._build(right_seq, mid + 1, hi, alpha_sorted)
        return _Node(lo, hi, bv, left, right)

    @property
    def alphabet(self) -> List[int]:
        """Sorted copy of the alphabet the tree was built with."""
        return list(self._alphabet)

    def __len__(self) -> int:
        return self._n

    def rank(self, symbol: int, i: int) -> int:
        """Count of ``symbol`` in ``seq[0:i]`` (half-open, 0-indexed)."""
        if i < 0:
            raise ValueError(f"i must be non-negative, got {i}")
        if i > self._n:
            raise ValueError(f"i out of range: {i} > {self._n}")
        if symbol not in set(self._alphabet):
            return 0
        node = self._root
        if node is None:
            return 0
        while node is not None and not node.is_leaf:
            mid = (node.lo + node.hi) // 2
            if symbol <= mid:
                i = node.bv.rank0(i)
                node = node.left
            else:
                i = node.bv.rank1(i)
                node = node.right
        return i

    def select(self, symbol: int, k: int) -> int:
        """Index of the k-th occurrence of ``symbol`` (1-indexed k), or -1."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if symbol not in set(self._alphabet):
            return -1

        # First, descend to the leaf for symbol to get the starting rank.
        node = self._root
        if node is None:
            return -1
        path: List[Tuple[_Node, bool]] = []  # (node, went_right)
        while node is not None and not node.is_leaf:
            mid = (node.lo + node.hi) // 2
            if symbol <= mid:
                path.append((node, False))
                node = node.left
            else:
                path.append((node, True))
                node = node.right
        if node is None:
            return -1

        # Walk back up, translating rank in child to rank in parent.
        for parent, went_right in reversed(path):
            if went_right:
                pos = parent.bv.select1(k)
            else:
                pos = parent.bv.select0(k)
            if pos == -1:
                return -1
            k = pos + 1  # rank among the child's symbols up to and including pos

        return k - 1  # convert rank (1-indexed) to position (0-indexed)

    def range_quantile(self, l: int, r: int, k: int) -> int:
        """The k-th smallest value in ``seq[l:r]`` (0-indexed k, half-open range)."""
        if l < 0 or r < 0:
            raise ValueError(f"l and r must be non-negative, got l={l}, r={r}")
        if l > r:
            raise ValueError(f"l must be <= r, got l={l}, r={r}")
        if r > self._n:
            raise ValueError(f"r out of range: {r} > {self._n}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k >= r - l:
            raise ValueError(
                f"k={k} out of range for {r - l} elements in [{l}, {r})"
            )
        if self._root is None:
            raise ValueError("cannot query an empty tree")
        return self._range_quantile(self._root, l, r, k)

    def _range_quantile(
        self, node: _Node, l: int, r: int, k: int
    ) -> int:
        if node.is_leaf:
            return node.lo
        bv = node.bv
        zeros_in_range = bv.rank0(r) - bv.rank0(l)
        if k < zeros_in_range:
            new_l = bv.rank0(l)
            new_r = bv.rank0(r)
            return self._range_quantile(node.left, new_l, new_r, k)
        else:
            new_k = k - zeros_in_range
            new_l = bv.rank1(l)
            new_r = bv.rank1(r)
            return self._range_quantile(node.right, new_l, new_r, new_k)
