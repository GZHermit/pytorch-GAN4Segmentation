# coding:utf-8

from trainval import fcns, pspnet, refinenet

models = {'fcn32s': fcns,
          'fcn16s': fcns,
          'fcn8s': fcns,
          'pspnet': pspnet,
          'refinenet': refinenet}


def get_models():
    return models
