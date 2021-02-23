#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions for SHAPE usage.

Shape: http://www.ee.ub.edu/index.php?option=com_content&view=article&id=575:shape-available&catid=80:news&Itemid=466

Author: Andrew Tarzia

Date Created: 23 Feb 2021

"""

import subprocess as sp


def ref_shape_dict():
    return {
        'cube': {
            'vertices': '8',
            'label': 'CU-8',
            'code': '4',
        },
        'octagon': {
            'vertices': '8',
            'label': 'OP-8',
            'code': '1',
        },
    }

def write_shape_input_file(
    input_file,
    name,
    structure,
    num_vertices,
    central_atom_id,
    ref_shapes,
):
    """
    Write input file for shape.

    """

    title = '$shape run by atools\n'
    size_of_poly = f'{num_vertices} {central_atom_id}\n'
    codes = ' '.join(ref_shapes)+'\n'

    structure_string = f'{name}\n'
    pos_mat = structure.get_position_matrix()
    for atom in structure.get_atoms():
        ele = atom.__class__.__name__
        x, y, z = pos_mat[atom.get_id()]
        structure_string += f'{ele} {x} {y} {z}\n'

    string = title+size_of_poly+codes+structure_string

    with open(input_file, 'w') as f:
        f.write(string)



def run_shape(input_file, shape_path, std_out):
    """
    Run input file for shape.

    """

    cmd = (
        f'{shape_path} {input_file}'
    )

    with open(std_out, 'w') as f:
        # Note that sp.call will hold the program until completion
        # of the calculation.
        sp.call(
            cmd,
            stdin=sp.PIPE,
            stdout=f,
            stderr=sp.PIPE,
            # Shell is required to run complex arguments.
            shell=True
        )


def collect_all_shape_values(output_file):
    """
    Collect shape values from output.

    """

    with open(output_file, 'r') as f:
        lines = f.readlines()

    label_idx_map = {}
    for line in reversed(lines):
        if 'Structure' in line:
            line = [
                i.strip()
                for i in line.rstrip().split(']')[1].split(' ')
                if i.strip()
            ]
            for idx, symb in enumerate(line):
                label_idx_map[symb] = idx
            break
        line = [i.strip() for i in line.rstrip().split(',')]
        values = line

    shapes = {
        i: float(values[1+label_idx_map[i]]) for i in label_idx_map
    }

    return shapes