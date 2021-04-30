#!/usr/bin/env python

import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description='Compute the perfect recall.')
    parser.add_argument('name', help='Path to sdk json file.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(f'hello {args.name}!')
    if args.name == 'lizz':
        warnings.warn('I have a secret!')


if __name__ == '__main__':
    main()
