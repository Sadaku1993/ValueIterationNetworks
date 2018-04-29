from __future__ import print_function
import argparse
import pickle
import numpy as np


import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

def main():
    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--data', '-d', type=str, default='./map_data.pkl',
                        help='Path to map data generated with scripts_make_data.py')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default="",
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of uints')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

if __name__ == "__main__":
    main()
