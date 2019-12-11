import unittest


class TestFor(unittest.TestCase):
    def test_for_break(self):
        for i in range(10):
            if i < 11:
                continue
            if i == 11:
                break
        else:
            print('1')

        for i in range(10):
            if i < 3:
                continue
            if i == 5:
                break
        else:
            print('2')