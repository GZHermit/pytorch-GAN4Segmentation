# coding:utf-8
from datasets import CityScapes, VOC

datasets = {
    'voc': VOC.VOC2012,
    'cityscapes': CityScapes.CityScapes
}


def get_datasets():
    return datasets
