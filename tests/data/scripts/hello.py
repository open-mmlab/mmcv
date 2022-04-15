# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python

import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello.')
    parser.add_argument('name', help='To whom.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(f'hello {args.name}!')
    if args.name == 'agent':
        warnings.warn('I have a secret!')


if __name__ == '__main__':
    main()
