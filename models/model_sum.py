# coding:utf-8

from models import FCNs, PSPNet, RefineNet

Models = {
    'fcn32s': FCNs.FCN32s,
    'fcn16s': FCNs.FCN16s,
    'fcn8s': FCNs.FCN8s,
    'pspnet': PSPNet.PSPNet,
    'refinenet': RefineNet.RefineNet
}


def get_models():
    return Models
