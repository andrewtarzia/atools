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
Usage: zeopp_usage.py job CIF

    job (str) : target job to be run.
    structure (str) : input structure to run job on.
        (include '*' if to be run on pattern.) See Zeo++ manual for
        possible structures.

    Available jobs:
        - zsa_n2: runs surface area analysis (-ha -sa)

    """)
        sys.exit()
    else:
        script = sys.argv[1]
        glob_pattern = sys.argv[2]
        warn = sys.argv[3].lower()


if __name__ == "__main__":
    main()
