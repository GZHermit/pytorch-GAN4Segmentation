# coding:utf-8

import torch
import argparse
import os
import time
import random
from trainval import trainval_sum


def start(args):
    models = trainval_sum.get_models()
    if args.is_gan:
        pass
    else:
        model = models[args.g]
        if args.mode == 'train':
            model.train(args)
        else:
            model.validate(args)


def get_arguments():
    BATCH_SIZE = 1
    G = 'fcn32s'
    D = 'vgg16'
    MODE = 'train'
    IS_GAN = False
    IS_MULTITASK = False
    NUM_STEPS = 1e5
    DATASET = 'cityscapes'
    DATASET_ROOT = '/home/gzh/Workspace/Dataset/Cityscapes_DATA'

    parser = argparse.ArgumentParser(description="GAN for Semantic Segmentation")

    parser.add_argument("--d", type=str, default=D,
                        help="which d_model can be choosed")
    parser.add_argument("--g", type=str, default=G,
                        help="which g_model can be choosed")
    parser.add_argument("--mode", type=str, default=MODE,
                        help="the mode of operate model")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="the kind of the dataset")
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT,
                        help="the root_path of the dataset")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--is_gan", type=bool, default=IS_GAN,
                        help="whether to train model by gan")
    parser.add_argument("--is_multitask", type=bool, default=IS_MULTITASK,
                        help="whether train gan by using the multitask")

    # parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
    #                     help="Random seed to have reproducible results.")
    # parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
    #                     help="How many images to save.")
    # parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
    #                     help="Save summaries and checkpoint every often.")
    # parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
    #                     help="Base learning rate for training with polynomial decay.")
    # parser.add_argument("--lambd", type=float, default=LAMBD,
    #                     help="a constant for constrainting the D-model loss")
    # parser.add_argument("--momentum", type=float, default=MOMENTUM,
    #                     help="Momentum component of the optimiser.")
    # parser.add_argument("--power", type=float, default=POWER,
    #                     help="Decay parameter to compute the learning rate.")
    # parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
    #                     help="Regularisation parameter for L2-loss.")
    # parser.add_argument("--is_training", type=bool, default=False,
    #                     help="Whether to updates the running means and variances during the training.")

    # parser.add_argument("--random_mirror", type=bool, default=True,
    #                     help="Whether to randomly mirror the inputs during the training.")
    # parser.add_argument("--random_scale", type=bool, default=True,
    #                     help="Whether to randomly scale the inputs during the training.")
    # parser.add_argument("--random_crop", type=bool, default=True,
    #                     help="Whether to randomly scale the inputs during the training.")
    # parser.add_argument("--data_dir", type=list, default=DATA_DIRECTORY,
    #                     help="Path to the directory containing the PASCAL VOC dataset.")
    # parser.add_argument("--img_size", type=tuple, default=IMG_SIZE,
    #                     help="Comma_separated string with height and width of images.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    start(args)
