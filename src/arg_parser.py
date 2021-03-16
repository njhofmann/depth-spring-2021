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
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--channels', '-c', type=parse_channels)
    parser.add_argument('--seg', '-s', action='store_true')
    parser.add_argument('--box', '-b', action='store_true')
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    # TODO dataset,
    return parser


def get_user_args() -> ap.Namespace:
    user_args = init_arg_parser().parse_args(sys.argv[1:])

    if user_args.seg == user_args.box:
        raise ValueError('can only select segmentation or bounding box')

    return user_args
