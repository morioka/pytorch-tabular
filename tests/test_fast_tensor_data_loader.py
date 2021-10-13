import numpy as np
import torch

import unittest
from fast_tensor_data_loader import FastTensorDataLoader

# https://qiita.com/aomidro/items/3e3449fde924893f18ca

class TestFastTensorDataLoader(unittest.TestCase):
    """test class of fast_tensor_data_loader.py
    """

    def setUp(self):
        self.x = np.arange(100)
        self.y = np.arange(100)

    def test_batch_compatible(self):
        loader = FastTensorDataLoader(self.x, self.y, batch_size=5)
        self.assertEqual(20,    loader.n_batches)
        self.assertEqual(5,     loader.batch_size)
        self.assertEqual(100,   loader.dataset_len)
        self.assertEqual(False, loader.shuffle)
        self.assertEqual(False, loader.drop_last)
        for i, j in loader:
            pass
        self.assertEqual(loader.batch_size, len(i))

    def test_batch_incompatible(self):
        loader = FastTensorDataLoader(self.x, self.y, batch_size=9)
        self.assertEqual(12,    loader.n_batches)
        self.assertEqual(9,     loader.batch_size)
        self.assertEqual(100,   loader.dataset_len)
        self.assertEqual(False, loader.shuffle)
        self.assertEqual(False, loader.drop_last)
        for i, j in loader:
            pass
        self.assertNotEqual(loader.batch_size, len(i))

    def test_batch_incompatible_droplast(self):
        loader = FastTensorDataLoader(self.x, self.y, batch_size=9, drop_last=True)
        self.assertEqual(11,    loader.n_batches)
        self.assertEqual(9,     loader.batch_size)
        self.assertEqual(100,   loader.dataset_len)
        self.assertEqual(False, loader.shuffle)
        self.assertEqual(True,  loader.drop_last)
        for i, j in loader:
            pass
            # print(i)
        self.assertNotEqual(99, i[-1])
        self.assertEqual(98, i[-1])
        self.assertEqual(loader.batch_size, len(i))

    def test_batch_incompatible_droplast1(self):
        loader = FastTensorDataLoader(self.x, self.y, batch_size=5, drop_last=True)
        self.assertEqual(20,    loader.n_batches)
        self.assertEqual(5,     loader.batch_size)
        self.assertEqual(100,   loader.dataset_len)
        self.assertEqual(False, loader.shuffle)
        self.assertEqual(True,  loader.drop_last)
        for i, j in loader:
            pass
            # print(i)
        self.assertEqual(self.x[-1], i[-1])
        self.assertEqual(loader.batch_size, len(i))

    def test_batch_shuffle(self):
        loader = FastTensorDataLoader(self.x, self.y, batch_size=5, shuffle=True)
        self.assertEqual(20,    loader.n_batches)
        self.assertEqual(5,     loader.batch_size)
        self.assertEqual(100,   loader.dataset_len)
        self.assertEqual(True,  loader.shuffle)
        self.assertEqual(False, loader.drop_last)
        for i, j in loader:
            pass
        self.assertEqual(loader.batch_size, len(i))
        self.assertNotEqual(self.x[-1], i[-1])

    def tearDown(self):
        del self.x
        del self.y

if __name__ == "__main__":
    unittest.main()

