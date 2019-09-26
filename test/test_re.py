import re
import unittest


class TestRE(unittest.TestCase):
    def test_findall(self):
        res = re.findall('\d\d', '1234')
        self.assertTrue(isinstance(res, list))
        self.assertTrue(res[0] == '12' and res[1] == '34')
        res = re.findall('\d\d\d', '1234')
        self.assertEqual(res[0], '123')


if __name__ == '__main__':
    unittest.main()
