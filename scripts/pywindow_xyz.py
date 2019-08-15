#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to run pywindow on a series of XYZ files.

Author: Andrew Tarzia

Date Created: 15 Aug 2019

"""

import sys
import glob
import pywindow as pw


def main():
    if (not len(sys.argv) == 2):
        print("""
Usage: pywindow_xyz.py glob_pattern

    """)
        sys.exit()
    else:
        glob_pattern = sys.argv[1]

    files = glob.glob(f'*{glob_pattern}')
    for file in files:
        print(f'doing {file}')
        json_file = file.replace('.xyz', '.json')
        pw_out = file.replace('.xyz', '_com.xyz')
        molsys = pw.MolecularSystem.load_file(file)
        mol = molsys.system_to_molecule()
        mol.full_analysis()
        # Dump pyWindow properties into JSON and cage into xyz
        mol.dump_properties_json(json_file)
        mol.dump_molecule(pw_out, include_coms=True)
        print(f'done {file}')


if __name__ == "__main__":
    main()
