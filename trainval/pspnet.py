# coding:utf-8

import torch
from models import model_sum
from datasets import dataset_sum

classes = model_sum.get_models()
datasets = dataset_sum.get_datasets()


def train(args):
    net = classes[args.dataset].cuda()


def validate():
    pass
