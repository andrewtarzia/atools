#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for zeo++ usage

Author: Andrew Tarzia

Date Created: 28 Jul 2019
"""


def convert_zsa_to_xyz(file):
    """
    Convert .zsa coordinates into XYZ file for visualisation.

    """
    with open(file, 'r') as f:
        data = f.readlines()

    for i, j in enumerate(data):
        if 'color red' in j:
            red_mention = i

    greens = data[1:red_mention]
    reds = data[red_mention+1:]

    n_atoms = len(greens) + len(reds)
    xyz_file = file.replace('.zsa', '_z.xyz')

    with open(xyz_file, 'w') as f:
        f.write(f'{n_atoms}\nWritten by Andrew Tarzia!\n')
        for g in greens:
            id = 'H'
            D = g.rstrip().replace('{', '').replace('}', '')
            x, y, z = [
                i for i in D.replace('point', '').split(' ') if i
            ]
            f.write(f'{id} {x} {y} {z}\n')
        for g in reds:
            id = 'P'
            D = g.rstrip().replace('{', '').replace('}', '')
            x, y, z = [
                i for i in D.replace('point', '').split(' ') if i
            ]
            f.write(f'{id} {x} {y} {z}\n')
