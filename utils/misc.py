# coding:utf-8

import os
import sys


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
