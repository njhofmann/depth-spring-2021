import argparse as ap
from typing import Tuple
import sys


def parse_channels(arg: str) -> Tuple[bool, bool]:
    # TODO move this logic?
    if arg == 'rgbd':
        return True, True
    elif arg == 'rgb':
        return True, False
    elif arg == 'd':
        return False, True
    raise ValueError(f'{arg} is not a valid set of channels')


def init_arg_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('--model', '-m', default='vgg', choices=['vgg', 'alexnet', 'resnet', 'dense'],
                        help='model to use as a feature extractor')
    parser.add_argument('--epochs', '-e', type=int, default=None, help='number of epochs to train on')
    parser.add_argument('--iterations', '-i', type=int, default=None)
    parser.add_argument('--channels', '-c', type=parse_channels, default='rgbd',
                        help='type of input data to use; rgb, depth, or rgbd')
    parser.add_argument('--seg', '-s', action='store_true', help='train for task of semantic segmentation')
    parser.add_argument('--box', '-b', action='store_true', help='train for task of object detection')
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--iter_eval', '-ie', type=int, default=50,
                        help='prints and saves the status of training (loss, etc.) every given amount of intervals, if '
                             'negative displays & saves nothing')
    parser.add_argument('--model_name', '-mn', type=str, default=None, help='what to call the model being trained')
    parser.add_argument('--no_augment', '-na', action='store_false', help='to not augment dataset')
    parser.add_argument('--depth_conv_option', '-dco', type=str, choices=['all', 'front', 'back'], default=None,
                        help='selection of convolutional operators to replace with depth aware conv operators, '
                             'model dependent')
    parser.add_argument('--set_device', '-sd', default=0, type=int, help='set the default GPU for PyTorch to use')
    parser.add_argument('--alpha', '-a', default=8.3, type=int, help='value of alpha for depth convolutional operators')
    # TODO dataset,
    return parser


def get_user_args() -> ap.Namespace:
    user_args = init_arg_parser().parse_args(sys.argv[1:])

    if user_args.seg == user_args.box:
        raise ValueError('can only select segmentation or bounding box')

    if (user_args.epochs is None) == (user_args.iterations is None):
        raise ValueError('can only select one of epochs and iterations')

    if user_args.channels != (True, True) and user_args.depth_conv_option is not None:
        raise ValueError(f'can only use rgbd channels with depth convolutions')

    return user_args


if __name__ == '__main__':
    print(init_arg_parser().parse_args([]).no_augment)
