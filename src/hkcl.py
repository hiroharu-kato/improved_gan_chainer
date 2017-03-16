#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import chainer
import chainer.functions as cf
import cupy


class Adam(chainer.optimizers.Adam):
    def update_one_cpu(self, param, state):
        raise NotImplementedError

    def update_one_gpu(self, param, state):
        chainer.cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''
            m += one_minus_beta1 * (grad - m);
            v += one_minus_beta2 * (grad * grad - v);
            param -= lr * m / (sqrt(v + eps));
            ''',
            'adam',
        )(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2, self.eps, param.data, state['m'], state['v'])


class GaussianNoiseLayer(chainer.Link):
    def __init__(self, sigma):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma

    def __call__(self, x, test=False):
        if not test:
            x = x + self.sigma * cupy.random.normal(size=x.shape, dtype='float32')
        return x


class WeightNormalizedLinear(chainer.Link):
    def __init__(self, in_size, out_size, train_scale=False, init_theta=None, init_stdv=1.0):
        super(WeightNormalizedLinear, self).__init__()

        self.init_stdv = init_stdv
        if init_theta is None:
            init_theta = chainer.initializers.Normal(0.1)
        self.add_param('theta', (out_size, in_size), initializer=init_theta, dtype='float32')

        init_b = chainer.initializers.Constant(0)
        self.add_param('b', out_size, initializer=init_b, dtype='float32')

        if train_scale:
            init_scale = chainer.initializers.Constant(1)
            self.add_param('scale', out_size, initializer=init_scale, dtype='float32')
        else:
            self.scale = chainer.Variable(cupy.ones((out_size,), dtype='float32'))

    def __call__(self, x, init=False):
        # eps = 1e-6
        eps = 0
        if init:
            self.scale.data = self.scale.data * 0 + 1
            self.b.data = self.b.data * 0
        s = self.scale / cf.sqrt(eps + cf.sum(cf.square(self.theta), axis=1))
        w = cf.basic_math.mul(*cf.broadcast(self.theta, s[:, None]))
        x = cf.linear(x, w, self.b)

        if init:
            xd = x.data
            mean = xd.mean(0)
            std = xd.std(0)
            inv_stdv = self.init_stdv / std

            self.scale.data[:] = self.scale.data * inv_stdv
            self.b.data[:] = -mean * inv_stdv
            x = (x - mean[None, :]) * inv_stdv[None, :]

        return x


class SemiSupervisedIterator(chainer.iterators.SerialIterator):
    def __init__(
            self, dataset, batch_size, repeat=True, num_labeled_samples_per_class=10, num_unlabeled_sets=1, seed=None):
        super(SemiSupervisedIterator, self).__init__(dataset, batch_size, repeat, shuffle=False)

        self.num_unlabeled_sets = num_unlabeled_sets
        self.images = dataset._datasets[0]
        self.labels = dataset._datasets[1]
        num_classes = int(np.max(self.labels) + 1)

        # select samples
        rng = np.random.RandomState(seed)
        indices = [np.nonzero(self.labels == i)[0] for i in range(num_classes)]
        [rng.shuffle(i) for i in indices]
        indices = np.array([i[:num_labeled_samples_per_class] for i in indices]).flatten()
        self.indices_labeled = indices

        # shuffle
        self.shuffle()

    def shuffle(self):
        sets = []
        for _ in range(self.num_unlabeled_sets):
            sets.append(np.copy(self.images[np.random.permutation(self.images.shape[0])]))
        indices = np.tile(self.indices_labeled, self.images.shape[0] / self.indices_labeled.size + 1)
        indices = indices[np.random.permutation(indices.shape[0])][:self.images.shape[0]]
        sets.append(np.copy(self.images[indices]))
        sets.append(np.copy(self.labels[indices]))
        self.dataset = chainer.datasets.TupleDataset(*sets)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        batch = self.dataset[i:i_end]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                self.shuffle()
                if rest > 0:
                    batch.extend(self.dataset[:rest])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__
