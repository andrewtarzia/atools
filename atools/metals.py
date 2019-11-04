#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that define properties of metals.

Author: Andrew Tarzia

Date Created: 04 Nov 2019
"""

import sys


def get_electron_properties(symbol, oxidation_state):
    """
    Extract the charge and unpaired electrons of a metal.

    See for some basic definitions:
    https://en.wikipedia.org/wiki/Spin_states_(d_electrons)

    Parameters
    ----------
    symbol : :class:`str`
        Element symbol of metal.

    oxidation_state : :class:`int`
        Oxidation state of metal.

    Returns
    -------
    charge : :class:`int`
        Charge on metal.

    unpaired_e : :class:`int`
        Number of unpaired electrons.

    spin : :class:`int`
        Spin on metal == 2*unpaired_e + 1

    """
    _data = {
        ('Fe', '3'): (5, 1),
        ('Fe', '2'): (4, 0),
        ('Pd', '2'): (0)
    }

    charge = oxidation_state
    try:
        unpaired_es = _data[(symbol, oxidation_state)]
    except KeyError:
        sys.exit(f'{symbol} and {oxidation_state} not defined!')

    spins = [2*i + 1 for i in unpaired_es]
    return charge, unpaired_es, spins
