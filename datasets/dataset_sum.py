# coding:utf-8
from datasets import CityScapes, VOC

datasets = {
    'voc': VOC,
    'cityscapes': CityScapes
}


def get_datasets():
    return datasets
