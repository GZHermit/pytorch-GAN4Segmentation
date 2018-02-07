# coding:utf-8

import torch
import argparse
import os
import time
import random
from trainval import model_trainval


def start(args):
    models = model_trainval.get_models()
    model = models[args.model]
    if args.train:
        model.train(args)
    else:
        model.validate(args)


def get_arguments():
    pass


if __name__ == '__main__':
    args = get_arguments()
    start(args)
