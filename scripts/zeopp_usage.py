#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to run zeo++ functions on a CIF.

Author: Andrew Tarzia

Date Created: 28 Jul 2019

"""

import sys
from glob import glob


def main():
    if (not len(sys.argv) == 4):
        print("""
Usage: zeopp_usage.py script CIF

    """)
        sys.exit()
    else:
        script = sys.argv[1]
        glob_pattern = sys.argv[2]
        warn = sys.argv[3].lower()


if __name__ == "__main__":
    main()
