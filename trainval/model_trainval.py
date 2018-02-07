# coding:utf-8

from trainval import fcns, pspnet, refinenet

models = {'fcns': fcns,
          'pspnet': pspnet,
          'refinenet': refinenet}


def get_models():
    return models
