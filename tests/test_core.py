import unittest

from wavelet_tree_lib import WaveletTree, BitVector


class TestBitVector(unittest.TestCase):
    def test_rank1_basic(self):
        bv = BitVector([1, 0, 1, 1, 0, 1])
        self.assertEqual(bv.rank1(0), 0)
        self.assertEqual(bv.rank1(1), 1)
        self.assertEqual(bv.rank1(2), 1)
        self.assertEqual(bv.rank1(3), 2)
        self.assertEqual(bv.rank1(6), 4)

    def test_rank0(self):
        bv = BitVector([1, 0, 1, 1, 0, 1])
        self.assertEqual(bv.rank0(0), 0)
        self.assertEqual(bv.rank0(2), 1)
        self.assertEqual(bv.rank0(6), 2)

    def test_rank_at_full_length(self):
        bv = BitVector([0, 1, 1, 0])
        self.assertEqual(bv.rank1(4), 2)
        self.assertEqual(bv.rank0(4), 2)

    def test_select1(self):
        bv = BitVector([1, 0, 1, 1, 0, 1])
        self.assertEqual(bv.select1(1), 0)
        self.assertEqual(bv.select1(2), 2)
        self.assertEqual(bv.select1(3), 3)
        self.assertEqual(bv.select1(4), 5)
        self.assertEqual(bv.select1(5), -1)

    def test_select0(self):
        bv = BitVector([1, 0, 1, 1, 0, 1])
        self.assertEqual(bv.select0(1), 1)
        self.assertEqual(bv.select0(2), 4)
        self.assertEqual(bv.select0(3), -1)

    def test_empty_bitvector(self):
        bv = BitVector([])
        self.assertEqual(len(bv), 0)
        self.assertEqual(bv.rank1(0), 0)
        self.assertEqual(bv.select1(1), -1)

    def test_rank_out_of_range(self):
        bv = BitVector([1, 0])
        with self.assertRaises(ValueError):
            bv.rank1(3)
        with self.assertRaises(ValueError):
            bv.rank1(-1)


class TestWaveletTreeConstruction(unittest.TestCase):
    def test_empty_sequence_and_alphabet(self):
        t = WaveletTree([], [])
        self.assertEqual(len(t), 0)
        self.assertEqual(t.alphabet, [])

    def test_non_empty_sequence_empty_alphabet_raises(self):
        with self.assertRaises(ValueError):
            WaveletTree([1], [])

    def test_symbol_not_in_alphabet_raises(self):
        with self.assertRaises(ValueError):
            WaveletTree([1, 2], [1])

    def test_duplicate_alphabet_raises(self):
        with self.assertRaises(ValueError):
            WaveletTree([1], [1, 1])

    def test_negative_symbol_in_alphabet_raises(self):
        with self.assertRaises(ValueError):
            WaveletTree([0], [-1])

    def test_bool_rejected_in_alphabet(self):
        with self.assertRaises(TypeError):
            WaveletTree([0], [True])


class TestRank(unittest.TestCase):
    def test_rank_single_symbol(self):
        t = WaveletTree([5, 5, 5], [5])
        self.assertEqual(t.rank(5, 0), 0)
        self.assertEqual(t.rank(5, 1), 1)
        self.assertEqual(t.rank(5, 3), 3)

    def test_rank_missing_symbol(self):
        t = WaveletTree([1, 2, 3], [1, 2, 3])
        self.assertEqual(t.rank(99, 3), 0)

    def test_rank_full_sequence(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        t = WaveletTree(seq, sorted(set(seq)))
        for s in set(seq):
            self.assertEqual(t.rank(s, len(seq)), seq.count(s))

    def test_rank_boundaries(self):
        seq = [1, 2, 1, 2, 1]
        t = WaveletTree(seq, [1, 2])
        self.assertEqual(t.rank(1, 0), 0)
        self.assertEqual(t.rank(1, 1), 1)
        self.assertEqual(t.rank(1, 2), 1)
        self.assertEqual(t.rank(1, 3), 2)

    def test_rank_i_out_of_range(self):
        t = WaveletTree([1, 2], [1, 2])
        with self.assertRaises(ValueError):
            t.rank(1, 3)
        with self.assertRaises(ValueError):
            t.rank(1, -1)


class TestSelect(unittest.TestCase):
    def test_select_basic(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        t = WaveletTree(seq, sorted(set(seq)))
        # Positions of 1: indices 1, 3
        self.assertEqual(t.select(1, 1), 1)
        self.assertEqual(t.select(1, 2), 3)
        self.assertEqual(t.select(1, 3), -1)

    def test_select_missing_symbol(self):
        t = WaveletTree([1, 2], [1, 2])
        self.assertEqual(t.select(99, 1), -1)

    def test_select_single_occurrence(self):
        seq = [5, 5, 3, 5]
        t = WaveletTree(seq, [3, 5])
        self.assertEqual(t.select(3, 1), 2)
        self.assertEqual(t.select(3, 2), -1)

    def test_select_invalid_k(self):
        t = WaveletTree([1, 2], [1, 2])
        with self.assertRaises(ValueError):
            t.select(1, 0)


class TestRangeQuantile(unittest.TestCase):
    def test_range_quantile_basic(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        t = WaveletTree(seq, sorted(set(seq)))
        # seq[0:4] = [3,1,4,1], sorted = [1,1,3,4]
        self.assertEqual(t.range_quantile(0, 4, 0), 1)
        self.assertEqual(t.range_quantile(0, 4, 1), 1)
        self.assertEqual(t.range_quantile(0, 4, 2), 3)
        self.assertEqual(t.range_quantile(0, 4, 3), 4)

    def test_range_quantile_full_range(self):
        seq = [5, 2, 8, 2, 5]
        t = WaveletTree(seq, sorted(set(seq)))
        sorted_seq = sorted(seq)
        for k in range(len(seq)):
            self.assertEqual(t.range_quantile(0, len(seq), k), sorted_seq[k])

    def test_range_quantile_single_element(self):
        seq = [5, 2, 8, 2, 5]
        t = WaveletTree(seq, sorted(set(seq)))
        self.assertEqual(t.range_quantile(2, 3, 0), 8)

    def test_range_quantile_k_out_of_range(self):
        t = WaveletTree([1, 2, 3], [1, 2, 3])
        with self.assertRaises(ValueError):
            t.range_quantile(0, 3, 3)
        with self.assertRaises(ValueError):
            t.range_quantile(0, 3, -1)

    def test_range_quantile_invalid_range(self):
        t = WaveletTree([1, 2, 3], [1, 2, 3])
        with self.assertRaises(ValueError):
            t.range_quantile(2, 1, 0)
        with self.assertRaises(ValueError):
            t.range_quantile(-1, 2, 0)


class TestIntegration(unittest.TestCase):
    def test_rank_select_inverse(self):
        seq = [4, 2, 4, 2, 4, 2, 1]
        t = WaveletTree(seq, [1, 2, 4])
        for s in [1, 2, 4]:
            positions = [i for i, v in enumerate(seq) if v == s]
            for idx, pos in enumerate(positions, start=1):
                self.assertEqual(t.select(s, idx), pos)
                self.assertEqual(t.rank(s, pos + 1), idx)

    def test_large_alphabet_with_gaps(self):
        seq = [10, 20, 10, 30, 20, 10]
        t = WaveletTree(seq, [10, 20, 30])
        self.assertEqual(t.rank(10, 6), 3)
        self.assertEqual(t.rank(20, 6), 2)
        self.assertEqual(t.rank(30, 6), 1)
        self.assertEqual(t.select(30, 1), 3)
        self.assertEqual(t.range_quantile(0, 6, 0), 10)
        self.assertEqual(t.range_quantile(0, 6, 5), 30)


if __name__ == "__main__":
    unittest.main()
