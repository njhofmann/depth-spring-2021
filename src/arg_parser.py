import argparse as ap
from typing import Tuple
import sys


def parse_channels(arg: str) -> Tuple[bool, bool]:
    if arg == 'rgbd':
        return True, True
    elif arg == 'rgb':
        return True, False
    elif arg == 'd':
        return False, True
    raise ValueError(f'{arg} is not a valid set of channels')


def init_arg_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('--model', '-m', default='vgg')
    parser.add_argument('--epochs', '-e', type=int, default=None)
    parser.add_argument('--iterations', '-i', type=int, default=None)
    parser.add_argument('--channels', '-c', type=parse_channels, default='rgbd')
    parser.add_argument('--seg', '-s', action='store_true', default=True)
    parser.add_argument('--box', '-b', action='store_true')
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--iter_eval', '-ie', type=int, default=50,
                        help='prints and saves the status of training (loss, etc.) every given amount of intervals, if '
                             'negative displays & saves nothing')
    parser.add_argument('--model_name', '-mn', type=str, default=None)
    parser.add_argument('--no_augment', '-na', action='store_false', help='to not augment dataset')
    # TODO dataset,
    return parser


def get_user_args() -> ap.Namespace:
    user_args = init_arg_parser().parse_args(sys.argv[1:])

    if user_args.seg == user_args.box:
        raise ValueError('can only select segmentation or bounding box')

    if (user_args.epochs is None) == (user_args.iterations is None):
        raise ValueError('can only select epochs and iterations')

    return user_args


if __name__ == '__main__':
    print(init_arg_parser().parse_args([]).no_augment)
