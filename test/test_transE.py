import sys

sys.path.append('..')

import unittest

from model.transE import TransE
from util.print_func import print


class TestTransE(unittest.TestCase):
    def test_transE(self):
        transE = TransE(ent_total=3, rel_total=2, dim=128, margin=2.0, norm=2, iters=500, lr=0.001)
        transE.train([[0, 1, 2]])
        score = transE.predict([[0, 1, 2], [0, 0, 2]])
        print(score)

    def test_transE_FB13(self):
        from data.data_transformer import read_file
        data_dataset = '../data/FB13_small'
        id2ent, ent_total = read_file('ent', data_dataset)
        id2rel, rel_total = read_file('rel', data_dataset)
        train_list, train_total = read_file('train', data_dataset)
        test_list, test_total = read_file('test', data_dataset)
        valid_list, valid_total = read_file('valid', data_dataset)

        transE = TransE(ent_total=ent_total, rel_total=rel_total, dim=128, margin=1.0, norm=2, iters=1000, lr=0.005)
        print('start training')
        transE.train(train_list)
        print('start predicting')
        head_hit10, tail_hit10, avg_hit10 = transE.link_prediction(test_list)
        print(head_hit10)
        print(tail_hit10)
        print(avg_hit10)

    def test_transE_FB15K(self):
        from data.data_transformer import read_file
        data_dataset = '../data/FB15K_small'
        id2ent, ent_total = read_file('ent', data_dataset)
        id2rel, rel_total = read_file('rel', data_dataset)
        train_list, train_total = read_file('train', data_dataset)
        test_list, test_total = read_file('test', data_dataset)
        valid_list, valid_total = read_file('valid', data_dataset)

        transE = TransE(ent_total=ent_total, rel_total=rel_total, dim=128, margin=1.0, norm=2, iters=200, lr=0.005)
        print('start training')
        transE.train(train_list)
        print('start predicting')
        head_hit10, tail_hit10, avg_hit10 = transE.link_prediction(test_list)
        print(head_hit10)
        print(tail_hit10)
        print(avg_hit10)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestTransE('test_transE'))
    # suite.addTest(TestTransE('test_transE_FB13'))
    suite.addTest(TestTransE('test_transE_FB15K'))

    # suite =  unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
