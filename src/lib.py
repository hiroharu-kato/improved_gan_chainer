#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as cf
import chainer.links as cl
import matplotlib
import numpy as np
import os
import pickle

import hkcl

matplotlib.use('Agg')
import pylab


class Model(chainer.Chain):
    def __init__(self, generator, discriminator, averaged_discriminator, feature_matching_metric='l2'):
        self.feature_matching_metric = feature_matching_metric
        super(Model, self).__init__(generator=generator, discriminator=discriminator)
        self.averaged_discriminator = None

    def init_weight(self, noise, image):
        self.generator(noise, init=True)
        self.discriminator(image, init=True)
        self.averaged_discriminator = pickle.loads(pickle.dumps(self.discriminator))
        for p in self.averaged_discriminator.params():
            p.data *= 0

    def compute_discriminator_loss(self, image_real, image_fake, image_labeled, label):
        # predict
        prediction_real = self.discriminator(image_real)
        prediction_fake = self.discriminator(image_fake)
        prediction_labeled = self.discriminator(image_labeled)

        # discriminator loss
        prediction_real_lse = cf.logsumexp(prediction_real, axis=1)
        prediction_fake_lse = cf.logsumexp(prediction_fake, axis=1)
        loss_discriminator = (
            0.5 * cf.sum(cf.softplus(prediction_real_lse)) / prediction_real_lse.size +
            0.5 * cf.sum(-prediction_real_lse) / prediction_real_lse.size +
            0.5 * cf.sum(cf.softplus(prediction_fake_lse)) / prediction_fake_lse.size)

        # classifier loss
        loss_classifier = cf.softmax_cross_entropy(prediction_labeled, label)

        loss = loss_discriminator + loss_classifier
        chainer.reporter.report({'loss_discriminator': loss_discriminator, 'loss_classifier': loss_classifier}, self)
        return loss

    def compute_generator_loss(self, image_real, image_fake):
        # feature matching
        feature_real = self.discriminator(image_real, is_feature=True)
        feature_fake = self.discriminator(image_fake, is_feature=True)
        feature_real = cf.sum(feature_real, axis=0) / feature_real.shape[0]
        feature_fake = cf.sum(feature_fake, axis=0) / feature_fake.shape[0]
        diff = feature_real - feature_fake
        if self.feature_matching_metric == 'l2':
            loss = cf.sum(cf.square(diff)) / diff.size
        elif self.feature_matching_metric == 'l1':
            loss = cf.sum(cf.absolute(diff)) / diff.size

        chainer.reporter.report({'loss_generator': loss}, self)
        return loss

    def __call__(self, image, label):
        # act as classifier
        image.volatile = 'off'
        label.volatile = 'off'
        prediction = self.averaged_discriminator(image, test=True)
        accuracy = cf.accuracy(prediction, label)
        chainer.reporter.report({'accuracy': accuracy, 'error': 1 - accuracy}, self)
        return accuracy


class MNISTGenerator(chainer.Chain):
    def __init__(self):
        init = chainer.initializers.GlorotUniform()
        super(MNISTGenerator, self).__init__(
            linear1=cl.Linear(100, 500, nobias=True, initialW=init),
            linear1_bn=cl.BatchNormalization(500, use_gamma=False),
            linear2=cl.Linear(500, 500, nobias=True, initialW=init),
            linear2_bn=cl.BatchNormalization(500, use_gamma=False),
            linear3=hkcl.WeightNormalizedLinear(500, 28 ** 2, train_scale=True, init_theta=init))

    def __call__(self, x, test=False, init=False):
        x = cf.softplus(self.linear1_bn((self.linear1(x)), test=test))
        x = cf.softplus(self.linear2_bn((self.linear2(x)), test=test))
        x = cf.sigmoid(self.linear3(x, init=False))
        return x


class MNISTDiscriminator(chainer.Chain):
    def __init__(self):
        init = chainer.initializers.Normal(0.1)
        super(MNISTDiscriminator, self).__init__(
            noise1=hkcl.GaussianNoiseLayer(0.3),
            linear1=hkcl.WeightNormalizedLinear(28 ** 2, 1000, init_theta=init),
            noise2=hkcl.GaussianNoiseLayer(0.5),
            linear2=hkcl.WeightNormalizedLinear(1000, 500, init_theta=init),
            noise3=hkcl.GaussianNoiseLayer(0.5),
            linear3=hkcl.WeightNormalizedLinear(500, 250, init_theta=init),
            noise4=hkcl.GaussianNoiseLayer(0.5),
            linear4=hkcl.WeightNormalizedLinear(250, 250, init_theta=init),
            noise5=hkcl.GaussianNoiseLayer(0.5),
            linear5=hkcl.WeightNormalizedLinear(250, 250, init_theta=init),
            noise6=hkcl.GaussianNoiseLayer(0.5),
            linear6=hkcl.WeightNormalizedLinear(250, 10, init_theta=init, train_scale=True))

    def __call__(self, x, test=False, init=False, is_feature=False):
        x = self.noise1(x, test=test)
        x = cf.relu((self.linear1(x, init=init)))
        x = self.noise2(x, test=test)
        x = cf.relu((self.linear2(x, init=init)))
        x = self.noise3(x, test=test)
        x = cf.relu((self.linear3(x, init=init)))
        x = self.noise4(x, test=test)
        x = cf.relu((self.linear4(x, init=init)))
        x = self.noise5(x, test=test)
        x = cf.relu((self.linear5(x, init=init)))
        if is_feature:
            return x
        x = self.noise6(x, test=test)
        x = self.linear6(x, init=init)
        return x


class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__(
            generator=MNISTGenerator(),
            discriminator=MNISTDiscriminator(),
            averaged_discriminator=MNISTDiscriminator(),
            feature_matching_metric='l2')


class Updater(chainer.training.StandardUpdater):
    def __init__(self, iterator, noise_generator, model, optimizers, converter=chainer.dataset.convert.concat_examples,
                 device=None):
        super(Updater, self).__init__(iterator, optimizers, converter, device)
        self.iterator = iterator
        self.noise_generator = noise_generator
        self.model = model

    def update_core(self):
        # get data
        batch = self.iterator.next()
        image_unlabeled1, image_unlabeled2, image_labeled, label = self.converter(batch, self.device)
        image_fake1 = self.model.generator(self.noise_generator(label.size))
        image_fake2 = self.model.generator(self.noise_generator(label.size))

        # discriminator
        self._optimizers['discriminator'].target.cleargrads()
        self.model.compute_discriminator_loss(image_unlabeled1, image_fake1, image_labeled, label).backward()
        self._optimizers['discriminator'].update()

        # generator
        self._optimizers['generator'].target.cleargrads()
        self.model.compute_generator_loss(image_unlabeled2, image_fake2).backward()
        self._optimizers['generator'].update()

        # parameter averaging
        for p1, p2 in zip(self.model.discriminator.params(), self.model.averaged_discriminator.params()):
            p2.data += 0.0001 * (p1.data - p2.data)


class ImageWriter(chainer.training.extension.Extension):
    def __init__(self, generator, noise_generator, out, num_images=8):
        super(ImageWriter, self).__init__()
        self.generator = generator
        self.noise = noise_generator(num_images ** 2)
        self.out = out
        self.num_images = num_images

    def __call__(self, trainer=None):
        images = self.generator(self.noise).data.get()
        if images.ndim == 2:
            # mnist
            image_size = int(np.sqrt(images.shape[-1]))
            images = images.reshape((-1, image_size, image_size))
        else:
            # cifar
            images = images.swapaxes(1, 2)
            images = images.swapaxes(2, 3)
        images = images[:self.num_images ** 2]
        for i, image in enumerate(images):
            pylab.subplot(self.num_images, self.num_images, i + 1)
            pylab.imshow(image)
        pylab.savefig(os.path.join(self.out, 'generated.png'))
        pylab.close()
