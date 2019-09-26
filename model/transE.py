import random

import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np

from util.print_func import print


class TransEModel(nn.Module):
    def __init__(self, ent_total, rel_total, dim, margin=1.0, norm=2):
        super(TransEModel, self).__init__()

        self.ent_total = ent_total
        self.rel_total = rel_total
        self.dim = dim
        self.margin = margin
        self.norm = norm

        self.ent_embeddings = nn.Embedding(self.ent_total, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.norm, -1)

    def forward(self, pos_triple, neg_triple):
        pos_h, pos_r, pos_t = pos_triple
        neg_h, neg_r, neg_t = neg_triple

        pos_h_vec = self.ent_embeddings(pos_h)
        pos_t_vec = self.ent_embeddings(pos_t)
        pos_r_vec = self.rel_embeddings(pos_r)

        neg_h_vec = self.ent_embeddings(neg_h)
        neg_t_vec = self.ent_embeddings(neg_t)
        neg_r_vec = self.rel_embeddings(neg_r)

        p_score = self._calc(pos_h_vec, pos_t_vec, pos_r_vec)
        n_score = self._calc(neg_h_vec, neg_t_vec, neg_r_vec)

        return p_score, n_score

    def predict(self, triple):
        batch_h, batch_r, batch_t = triple
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()


class TransE:
    def __init__(self, ent_total, rel_total, dim=100, margin=1.0, norm=2, iters=100, lr=0.001):
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.iters = iters
        self.lr = lr

        self.transE = TransEModel(ent_total, rel_total, dim, margin, norm)
        self.optimizer = optimizer.SGD(self.transE.parameters(), lr=self.lr)
        self.criterion = nn.MarginRankingLoss(margin, reduction='sum')

        self.triple = None

    def _preprocess(self, triple):
        h, r, t = list(), list(), list()

        for item in triple:
            h.append(item[0])
            r.append(item[1])
            t.append(item[2])

        return torch.tensor([h, r, t])

    def _neg_sampling(self, triple):
        neg_triple = []
        for item in triple:
            h, r, t = item
            for i in range(10):
                ent = random.randint(0, self.ent_total - 1)
                rel = random.randint(0, self.rel_total - 1)

                rand = random.choice(['h', 'r', 't'])
                if rand == 'h':
                    h = ent
                elif rand == 't':
                    t = ent
                elif rand == 'r':
                    r = rel

                if (h, r, t) not in self.triple_set:
                    neg_triple.append((h, r, t))
                    break
            else:
                neg_triple.append((h, r, t))
        return neg_triple

    def train(self, triple):
        self.triple = triple
        self.triple_set = {tuple(item) for item in self.triple}

        for epoch in range(self.iters):
            neg_triple = self._neg_sampling(triple)
            triple_ = self._preprocess(triple)
            neg_triple_ = self._preprocess(neg_triple)

            self.optimizer.zero_grad()
            p_score, n_score = self.transE(triple_, neg_triple_)
            loss = self.criterion(p_score, n_score, torch.Tensor([-1]))
            print('training {}th epoch, loss={}'.format(epoch, loss.item()))

            loss.backward()
            self.optimizer.step()

    def predict(self, triple):
        triple_ = self._preprocess(triple)
        score = self.transE.predict(triple_)
        return score

    def tail_link_prediction(self, triple):
        hit10 = 0.0
        for i, item in enumerate(triple):
            h, r, t = item
            tail_link_data = []
            for tail in range(self.ent_total):
                if tail != t and (h, r, tail) in self.triple_set:
                    continue
                tail_link_data.append((h, r, tail))
                if tail == t:
                    pos = len(tail_link_data) - 1
            tail_link_data_ = self._preprocess(tail_link_data)
            score = self.transE.predict(tail_link_data_)
            hit = list(np.argsort(score)).index(pos)
            hit10 += 1 if hit <= 10 else 0
            print('predicting {}th=({},{},{}) hit={}'.format(i, h, r, t, hit))
        hit10 /= len(triple)
        return hit10

    def head_link_prediction(self, triple):
        hit10 = 0.0
        for i, item in enumerate(triple):
            h, r, t = item
            head_link_data = []
            for head in range(self.ent_total):
                if head != h and (head, r, t) in self.triple_set:
                    continue
                head_link_data.append((head, r, t))
                if head == h:
                    pos = len(head_link_data) - 1
            head_link_data_ = self._preprocess(head_link_data)
            score = self.transE.predict(head_link_data_)
            hit = list(np.argsort(score)).index(pos)
            hit10 += 1 if hit <= 10 else 0
            print('predicting {}th=({},{},{}) hit={}'.format(i, h, r, t, hit))
        hit10 /= len(triple)
        return hit10

    def link_prediction(self, triple):
        head_hit10 = self.head_link_prediction(triple)
        tail_hit10 = self.tail_link_prediction(triple)
        return head_hit10, tail_hit10, (head_hit10 + tail_hit10) / 2
