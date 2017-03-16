#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

import chainer
import chainer.training.extensions as cte
import cupy as cp

import hkcl
import lib


def preprocess_mnist(dataset):
    pass


def preprocess_cifar(dataset):
    dataset._datasets[0][:] = 2 * dataset._datasets[0][:] - 1


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_data', type=int, default=0)
    args = parser.parse_args()

    gpu = 0
    dim_noise = 100
    batch_size = 100
    if args.dataset == 'mnist':
        out = '../data/mnist'
        epoch = 300
        learning_rate = 0.003
        num_labeled_samples_per_class = 10
        get_dataset = chainer.datasets.get_mnist
        Model = lib.MNISTModel
        preprocess = preprocess_mnist
    elif args.dataset == 'cifar':
        out = '../data/cifar'
        epoch = 1200
        learning_rate = 0.0003
        num_labeled_samples_per_class = 400
        get_dataset = chainer.datasets.get_cifar10
        Model = lib.CIFARModel
        preprocess = preprocess_cifar

    # init xp
    chainer.cuda.get_device(gpu).use()
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    # create generator & discriminator
    model = Model()
    model.to_gpu()

    # setup optimizers
    optimizers = {'generator': hkcl.Adam(alpha=learning_rate, beta1=0.5),
                  'discriminator': hkcl.Adam(alpha=learning_rate, beta1=0.5)}
    optimizers['generator'].setup(model.generator)
    optimizers['discriminator'].setup(model.discriminator)

    # load dataset
    train, test = get_dataset()
    preprocess(train)
    preprocess(test)
    train_iter = hkcl.SemiSupervisedIterator(train, batch_size, num_unlabeled_sets=2, seed=args.seed_data,
                                             num_labeled_samples_per_class=num_labeled_samples_per_class)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

    def noise_generator(num):
        return cp.random.uniform(0., 1., size=(num, dim_noise), dtype='float32')

    # init
    image_init = cp.array(train._datasets[0][:500])
    noise_init = noise_generator(image_init.shape[0])
    model.init_weight(noise_init, image_init)

    # setup trainer
    updater = lib.Updater(train_iter, noise_generator, model, optimizers, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(cte.PrintReport(['epoch', 'validation/main/error', 'main/loss_classifier',
                                    'main/loss_discriminator', 'main/loss_generator', 'elapsed_time']))
    trainer.extend(cte.ProgressBar())
    trainer.extend(cte.LogReport())
    trainer.extend(cte.Evaluator(test_iter, model, device=gpu))
    trainer.extend(lib.ImageWriter(model.generator, noise_generator, out), trigger=(1, 'epoch'))
    trainer.reporter.add_observer('main', model)

    # run
    trainer.run()


if __name__ == '__main__':
    run()
