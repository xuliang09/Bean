from random import *
import unittest


class TestRandom(unittest.TestCase):
    def test_choice(self):
        foo = ['a', 'b', 'c', 'd', 'e']
        self.assertTrue(choice(foo) in foo)


if __name__ == '__main__':
    unittest.main()