# coding:utf-8

import torch
from utils import transforms
from models import model_sum
from datasets import dataset_sum


models = model_sum.get_models()
datasets = dataset_sum.get_datasets()


def train(args):

    joint_transform = transforms.Compose([
        transforms.RandomScale(),
        transforms.Mirror(),
        transforms.RandomCrop()
    ])
    trainset = datasets[args.dataset](mode=args.mode, root=args.dataset_root)

    net = models[args.g]


def validate():
    pass
