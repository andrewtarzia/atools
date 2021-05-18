#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for rdkit usage

Author: Andrew Tarzia

Date Created: 15 Mar 2019
"""

import numpy as np
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Descriptors3D import NPR1, NPR2, PMI1, PMI2, PMI3
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Geometry import rdGeometry
import pandas as pd


def calculate_all_MW(molecules):
    """
    Calculate the molecular weight of all molecules in DB dictionary.

    {name: SMILES}

    """
    for m, smile in molecules.items():
        # Read SMILES and add Hs
        mol = rdkit.AddHs(rdkit.MolFromSmiles(smile))
        MW = Descriptors.MolWt(mol)
        print(m, '---', smile, '---', 'MW =', MW, 'g/mol')


def draw_mol_to_svg(mol, filename):
    """
    Draw a single molecule to an SVG file with transparent BG.

    """
    # change BG to transperent
    # (https://sourceforge.net/p/rdkit/mailman/message/31637105/)
    o = DrawingOptions()
    o.bgColor = None
    # Use copy of molecule to avoid changing instance of mol.
    new_mol = rdkit.MolFromMolBlock(rdkit.MolToMolBlock(mol))
    rdkit.Compute2DCoords(new_mol)
    Draw.MolToFile(
        new_mol,
        filename,
        fitImage=True,
        imageType='svg',
        options=o
    )


def draw_smiles_to_svg(smiles, filename):
    """
    Draw a single molecule to an SVG file with transparent BG.

    """
    mol = rdkit.MolFromSmiles(smiles)
    draw_mol_to_svg(mol, filename)


def draw_and_save_grid(
    mol_list,
    names,
    subImgSize,
    mol_per_row,
    filename
):
    """
    Draw RDKit molecules and save SVG.

    """
    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=mol_per_row,
        subImgSize=subImgSize,
        legends=names,
        useSVG=True
    )
    save_svg(
        filename=filename,
        string=img
    )


def mol_list2grid(
    molecules,
    filename,
    mol_per_row,
    maxrows,
    subImgSize=(200, 200),
    names=None
):
    """
    Produce a grid of molecules in mol_list.

    molecules (list) - list of molecule SMILEs

    """

    if len(molecules) > mol_per_row * maxrows:
        # have to make multiple images
        new_mol_list = []
        new_names = []
        count = 1
        for i, mol in enumerate(molecules):
            new_mol_list.append(mol)
            if names is None:
                new_names = None
            else:
                new_names.append(names[i])
            # make image
            chk1 = len(new_mol_list) == mol_per_row * maxrows
            chk2 = i == len(molecules)-1
            if chk1 or chk2:
                draw_and_save_grid(
                    mol_list=new_mol_list,
                    mol_per_row=mol_per_row,
                    subImgSize=subImgSize,
                    names=new_names,
                    filename=f'{filename}_{count}.svg'
                )
                # img.save(filename + '_' + str(count) + '.png')
                new_mol_list = []
                new_names = []
                count += 1
    else:
        draw_and_save_grid(
            mol_list=molecules,
            mol_per_row=mol_per_row,
            subImgSize=subImgSize,
            names=names,
            filename=f'{filename}.svg'
        )


def save_svg(filename, string):
    """
    Save svg text to a file.

    """

    with open(filename, 'w') as f:
        f.write(string)


def read_mol_txt_file(filename):
    """
    Function to read molecule SMILES and information from txt file.

    """
    data = pd.read_table(filename, delimiter=':')
    molecules = {}
    diameters = {}
    for i, row in data.iterrows():
        # try:
        #     name, smile, radius = line.rstrip().split(':')
        # except ValueError:
        #     print(line, 'had : in there twice,
        #     fix this naming or SMILE')
        #     print('skipped')
        name = row['molecule']
        smile = row['smile']
        diameter = row['diameter']
        molecules[name] = smile
        diameters[name] = diameter
    return data, molecules, diameters


def get_inertial_prop(mol, cids):
    """
    Get inertial 3D descriptors for all conformers in mol.

    """
    # ratio 1 is I1/I3
    # ratio 2 is I2/I3
    sml_PMI, mid_PMI, lge_PMI = [], [], []
    ratio_1_, ratio_2_ = [], []
    for cid in cids:
        sml_PMI.append(PMI1(mol, confId=cid))
        mid_PMI.append(PMI2(mol, confId=cid))
        lge_PMI.append(PMI3(mol, confId=cid))
        ratio_1_.append(NPR1(mol, confId=cid))
        ratio_2_.append(NPR2(mol, confId=cid))

    return sml_PMI, mid_PMI, lge_PMI, ratio_1_, ratio_2_


def get_COMs(mol, cids):
    """
    Get COM of all conformers of mol.

    Code from:
    https://iwatobipen.wordpress.com/2016/08/16/
    scoring-3d-diversity-using-rdkit-rdkit/

    """
    coms = []
    numatoms = mol.GetNumAtoms()
    for confId in range(len(cids)):
        # print('conf:', confId)
        # print('number of atoms:', numatoms)
        conf = mol.GetConformer(confId)
        coords = np.array([
            list(
                conf.GetAtomPosition(atmidx))
            for atmidx in range(numatoms)
        ])
        # print('coords:')
        # print(coords)
        atoms = [atom for atom in mol.GetAtoms()]
        mass = Descriptors.MolWt(mol)
        # print('mass:', mass)
        centre_of_mass = np.array(
            np.sum(
                atoms[i].GetMass() * coords[i] for i in range(numatoms)
            )
        ) / mass
        # print(centre_of_mass)
        coms.append(centre_of_mass)

    return coms


def def_point(x, y, z):
    """
    Define a 3D point in RDKIT

    """
    point = rdGeometry.Point3D()
    point.x = x
    point.y = y
    point.z = z

    return point


def smiles2conformers(smiles, N=10, optimize=True):
    """
    Convert smiles string to N conformers.

    Keyword Arguments:
        smiles (str) - smiles string for molecule
        N (int) - number of conformers to generate using the ETKDG
            algorithm
        optimize (bool) - flag for UFF optimization (default=True)

    Returns:
        mol (RDKit molecule ::class::) - contains N conformers
    """
    # Read SMILES and add Hs
    mol = rdkit.MolFromSmiles(smiles)
    if mol is None:
        print('RDKit error for', smiles)
        return None
    mol = rdkit.AddHs(mol)
    # try based on RuntimeError from RDKit
    try:
        # 2D to 3D with multiple conformers
        cids = rdkit.EmbedMultipleConfs(
            mol=mol,
            numConfs=N,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
        )
        # quick UFF optimize
        for cid in cids:
            rdkit.UFFOptimizeMolecule(mol, confId=cid)
    except RuntimeError:
        print('RDKit error for', smiles)
        return None
    return mol


def get_query_atom_ids(query, rdkit_mol):
    """
    Yield the ids of atoms in `rdkit_mol` which match `query`.

    Multiple substructures in `rdkit_mol` can match `query` and
    therefore each set is yielded as a group.

    Parameters
    ----------
    query : :class:`str`
        A SMARTS string used to query atoms.

    rdkit_mol : :class:`rdkit.Mol`
        A molecule whose atoms should be queried.

    Yields
    ------
    :class:`tuple` of :class:`int`
        The ids of atoms in `molecule` which match `query`.

    """

    rdkit.SanitizeMol(rdkit_mol)
    yield from rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(query),
    )
