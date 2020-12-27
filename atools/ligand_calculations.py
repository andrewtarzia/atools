#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions for calculating ligand properties.

Author: Andrew Tarzia

Date Created: 29 May 2020

"""

import numpy as np
from itertools import combinations
from scipy.spatial.distance import euclidean

from .calculations import angle_between
from .stk_f import AromaticCNC, AromaticCNN, get_center_of_mass


def calculate_NN_distance(bb):
    """
    Calculate the N-N distance of ditopic building block.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC and metals + AromaticCNN.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    Returns
    -------
    NN_distance : :class:`float`
        Distance(s) between [angstrom] N atoms in functional groups.

    """

    fg_counts = 0
    N_positions = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC) or isinstance(fg, AromaticCNN):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            N_position, = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            N_positions.append(N_position)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have 2 AromaticCNC or AromaticCNN '
            'functional groups.'
        )

    # Calculate the angle between the two vectors.
    NN_distance = np.linalg.norm(N_positions[0] - N_positions[1])
    return NN_distance


def calculate_bite_angle(bb):
    """
    Calculate the bite angle of a ditopic building block.

    Here the bite angle is defined `visually` as in:
        https://doi.org/10.1016/j.ccr.2018.06.010

    It is calculated using the angles between binding vectors and the
    N to N vector.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC and metals + AromaticCNN.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        Stk molecule to analyse.

    Returns
    -------
    bite_angle : :class:`float`
        Angle between two bonding vectors of molecule.

    """

    fg_counts = 0
    fg_vectors = []
    N_positions = []

    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC) or isinstance(fg, AromaticCNN):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position.
            N_position, = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            N_positions.append(N_position)
            # Get centroid of neighbouring C atom positions.
            if isinstance(fg, AromaticCNC):
                CC_MP = bb.get_centroid(
                    atom_ids=(
                        fg.get_carbon1().get_id(),
                        fg.get_carbon2().get_id()
                    )
                )
            elif isinstance(fg, AromaticCNN):
                CC_MP = bb.get_centroid(
                    atom_ids=(
                        fg.get_carbon().get_id(),
                        fg.get_nitrogen2().get_id()
                    )
                )
            # Get vector between COM and N position.
            v = N_position - CC_MP
            fg_vectors.append(v)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have exactly 2 AromaticCNC or AromaticCNN '
            'functional groups.'
        )

    # Get N to N vector.
    NN_vec = N_positions[1] - N_positions[0]

    # Calculate the angle between the two vectors.
    angle_1 = np.degrees(angle_between(
        fg_vectors[0], NN_vec
    ))
    angle_2 = np.degrees(angle_between(
        fg_vectors[1], -NN_vec
    ))
    bite_angle = (angle_1 - 90) + (angle_2 - 90)
    return bite_angle


def get_furthest_pair_FGs(stk_mol):
    """
    Returns the pair of functional groups that are furthest apart.

    """

    if stk_mol.get_num_functional_groups() == 2:
        return tuple(i for i in stk_mol.get_functional_groups())
    elif stk_mol.get_num_functional_groups() < 2:
        raise ValueError(f'{stk_mol} does not have at least 2 FGs')

    fg_centroids = [
        (fg, stk_mol.get_centroid(atom_ids=fg.get_placer_ids()))
        for fg in stk_mol.get_functional_groups()
    ]

    fg_dists = sorted(
        [
            (i[0], j[0], euclidean(i[1], j[1]))
            for i, j in combinations(fg_centroids, 2)
        ],
        key=lambda x: x[2],
        reverse=True
    )

    return (fg_dists[0][0], fg_dists[0][1])


def calculate_N_COM_N_angle(bb):
    """
    Calculate the N-COM-N angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC and metals + AromaticCNN.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    Returns
    -------
    angle : :class:`float`
        Angle between two bonding vectors of molecule.

    """

    fg_counts = 0
    fg_positions = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC) or isinstance(fg, AromaticCNN):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            N_position, = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            fg_positions.append(N_position)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have 2 AromaticCNC or AromaticCNN '
            'functional groups.'
        )

    # Get building block COM.
    COM_position = get_center_of_mass(bb)

    # Get vectors.
    fg_vectors = [i-COM_position for i in fg_positions]

    # Calculate the angle between the two vectors.
    angle = np.degrees(angle_between(*fg_vectors))
    return angle
