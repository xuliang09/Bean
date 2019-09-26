import unittest


class TestSet(unittest.TestCase):
    def test_set(self):
        self.assertRaises(TypeError, {[0, 1], [1, 2]})