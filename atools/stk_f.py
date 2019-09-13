#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for stk usage.

Author: Andrew Tarzia

Date Created: 18 Mar 2019
"""

from glob import glob
import stk
import logging
import sys


def build_ABCBA(core, liga, link, flippedlink=False):
    """
    Build ABCBA ligand using linear stk polymer.

    Polymer structure:
        ligand -- linker -- core -- linker -- ligand

    Parameters
    ----------
    core : :class:`stk.BuildingBlock`
        The molecule to use as core.
    liga : :class:`stk.BuildingBlock`
        The molecule to use as ligand.
    link : :class:`stk.BuildingBlock`
        The molecule to use as linker.
    flippedlink : :class:`bool`
        `True` to flip the linker molecules. Defaults to `False`.

    Returns
    -------
    polymer : :class:`stk.ConstructedMolecule`
        Built molecule pre optimisation.

    """
    if flippedlink is False:
        orientation = (0, 0, 0, 1, 1)
    elif flippedlink is True:
        orientation = (0, 1, 0, 0, 1)
    polymer = stk.ConstructedMolecule(
        building_blocks=[liga, link, core],
        topology_graph=stk.polymer.Linear(
            repeating_unit='ABCBA',
            num_repeating_units=1,
            orientations=orientation,
            num_processes=1
        )
    )
    return polymer


def build_ABA(core, liga):
    """
    Build ABA ligand using linear stk.Polymer().

    Polymer structure:
        ligand -- core -- ligand

    Parameters
    ----------
    core : :class:`stk.BuildingBlock`
        The molecule to use as core.
    liga : :class:`stk.BuildingBlock`
        The molecule to use as ligand.

    Returns
    -------
    polymer : :class:`stk.ConstructedMolecule`
        Built molecule pre optimisation.

    """
    polymer = stk.ConstructedMolecule(
        building_blocks=[liga, core],
        topology_graph=stk.polymer.Linear(
            repeating_unit='ABA',
            num_repeating_units=1,
            orientations=(0, 0, 1),
            num_processes=1
        )
    )
    return polymer


def build_population(directory, fgs=None, suffix='.mol'):
    """
    Build population of BuilingBlocks from directory.

    Parameters
    ----------
    directory : :class:`str`
        Directory containing molecule files.
    fgs : :class:`list` of :class:`str`
        Functional groups of BuildingBlocks. Defaults to bromine.
    suffix : :class:`str`
        File suffix to use.

    Returns
    -------
    popn : :class:`stk.Population`
        Population of BuildingBlocks.

    """
    if fgs is None:
        fgs = ['bromine']

    mols = []
    for file in sorted(glob(directory + '*' + suffix)):
        mol = stk.BuildingBlock.init_from_file(
            path=file,
            functional_groups=fgs,
            use_cache=False
        )
        mol.name = file.rstrip(suffix).replace(directory, '')
        mols.append(mol)
    popn = stk.Population(*mols)
    return popn


def topo_2_property(topology, property):
    """Returns properties of a topology for a given topology name.

    Properties:
        'stk_func' - gives the stk topology function for building cages
        'stoich' - gives the stoichiometries of both building blocks
            assuming that the first building block has the larger
            number of functional groups.
        'noimines' - gives the number of imines formed to build that
            topology
        'expected_wind' - gives the number of windows expected

    Currently defined topologies:
        TwoPlusThree topologies
        ThreePlusThree topologies

    """
    properties = ['stk_func', 'stoich', 'noimines', 'expected_wind']
    if property not in properties:
        logging.info(f'{property} not defined')
        logging.info(f'possible properties: {properties}')
        sys.exit('exitting.')

    dict = {
        '2p3': {
            'stk_func': stk.TwoPlusThree(),
            'stoich': (2, 3),
            'noimines': 6,
            'expected_wind': 3,
        },
        '4p6': {
            'stk_func': stk.FourPlusSix(),
            'stoich': (4, 6),
            'noimines': 12,
            'expected_wind': 4,
        },
        '4p62': {
            'stk_func': stk.FourPlusSix2(),
            'stoich': (4, 6),
            'noimines': 12,
            'expected_wind': 4,
        },
        '6p9': {
            'stk_func': stk.SixPlusNine(),
            'stoich': (6, 9),
            'noimines': 18,
            'expected_wind': 5,
        },
        'dodec': {
            'stk_func': stk.Dodecahedron(),
            'stoich': (20, 30),
            'noimines': 60,
            'expected_wind': 12,
        },
        '8p12': {
            'stk_func': stk.EightPlusTwelve(),
            'stoich': (8, 12),
            'noimines': 24,
            'expected_wind': 6,
        },
        '1p1': {
            'stk_func': stk.OnePlusOne(
                # place bb1 on vertex (0), bb2 on vertex (1)
                bb_positions={0: [0], 1: [1]}),
            'stoich': (1, 1),
            'noimines': 3,
            'expected_wind': 3,
        },
        '4p4': {
            'stk_func': stk.FourPlusFour(
                # place bb1 on vertex (0, 2), bb2 on vertex (1, 3)
                bb_positions={0: [0, 3, 5, 6], 1: [1, 2, 4, 7]}),
            'stoich': (4, 4),
            'noimines': 12,
            'expected_wind': 6,
        },
    }
    if topology not in dict:
        logging.info(f'properties not defined for {topology}')
        sys.exit('exitting')
    return dict[topology][property]


def is_porous(pore_diameter, max_window_diameter):
    """
    Returns True if a cage is deemed to be porous.

    A porous cage is defined as having:
        (Computationally-inspired discovery of an unsymmetrical
        porous organic cage)
        1 - p_diam_opt > 3.4
        2 - max window_diameter > 2.8

    """
    if max_window_diameter > 2.8 and pore_diameter > 3.4:
        return True
    else:
        return False


def is_collapsed(topo, pore_diameter, no_window):
    """
    Returns True if a cage is deemed to be collapsed.

    A collapsed cage is defined as having:
        - pore_diam_opt < 2.8 Angstrom (H2 kinetic diameter)
        - number of windows != expected number based on topology.

    """
    expected_wind = topo_2_property(topo, property='expected_wind')
    if expected_wind != no_window:
        return True
    elif pore_diameter < 2.8:
        return True
    else:
        return False


def load_StructUnitX(file, X=0):
    """
    Load StructUnitX class with the cache turned off to avoid
    misreading of file.

    Keyword Arguments:
        file (str) - file to load in
        X (int) - order of struct unit to load. Defaults to 0

    X = 0 -> stk.StructUnit
    X = 2 -> stk.StructUnit2
    X = 3 -> stk.StructUnit3

    """
    stk.OPTIONS['cache'] = False  # turn caching off for loading
    if X == 0:
        struct = stk.StructUnit(file)
    elif X == 2:
        struct = stk.StructUnit2(file)
    elif X == 3:
        struct = stk.StructUnit3(file)
    else:
        logging.info('X must be 0, 2 or 3')
        sys.exit('exitting')
    stk.OPTIONS['cache'] = True  # turn caching back on
    return struct


def default_stk_MD_settings():
    """Default settings from stk source code as of 26/04/19.

    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 300,  # K
                'conformers': 50,
                'time_step': 1.0,  # fs
                'eq_time': 10,  # ps
                'simulation_time': 200,  # ps
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False
                }
    return Settings


def atarzia_short_MD_settings():
    """My default settings for short, crude cage optimizations in stk.


    Modified on 26/04/19.
    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 700,  # K
                'conformers': 50,
                'time_step': 1,  # fs
                'eq_time': 50,  # ps
                'simulation_time': 1000,  # ps -- 1 ns
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False}
    return Settings


def atarzia_long_MD_settings():
    """
    My default settings for rigorous cage optimizations in stk.

    Mimics: Computationally-inspired discovery of an unsymmetrical
    porous organic cage - DOI:10.1039/C8NR06868B

    Modified on 26/04/19.
    Modified on 06/06/19.
    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 700,  # K
                'conformers': 5000,  # change from 10000
                'time_step': 0.5,  # fs
                'eq_time': 100,  # ps
                # ps -- 50 ns changed from 100 ns
                'simulation_time': -500,
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False}
    return Settings


def optimize_structunit(infile, outfile, exec,
                        settings=None, method='OPLS'):
    """
    Read file into StructUnit and run optimization via method.

    """
    logging.info(f'loading in: {infile}')
    struct = load_StructUnitX(infile, X=0)
    if method == 'OPLS':
        # Use standard settings applied in andrew_marsh work if
        # md/settings is None.
        if settings is None:
            Settings = default_stk_MD_settings()
        else:
            Settings = settings
        logging.info(f'doing MD optimization of {infile}')
        # restricted=False optimization with OPLS forcefield by default
        ff = stk.MacroModelForceField(
            macromodel_path=exec, restricted=False
        )
        # MD process - run MD, collect N conformers, optimize each,
        # return lowest energy conformer
        md = stk.MacroModelMD(
            macromodel_path=exec,
            output_dir=Settings['output_dir'],
            timeout=Settings['timeout'],
            force_field=Settings['force_field'],
            temperature=Settings['temperature'],
            conformers=Settings['conformers'],
            time_step=Settings['time_step'],
            eq_time=Settings['eq_time'],
            simulation_time=Settings['simulation_time'],
            maximum_iterations=Settings['maximum_iterations'],
            minimum_gradient=Settings['minimum_gradient'],
            use_cache=Settings['use_cache']
        )
        macromodel = stk.OptimizerSequence(ff, md)
        macromodel.optimize(mol=struct)
        struct.write(outfile)
        logging.info('done')
    elif method == 'xtb':
        logging.info(f'doing xTB optimization of {infile}')
        xtb_opt = stk.XTB(
            xtb_path=exec,
            output_dir='xtb_opt',
            opt_level='tight',
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True
        )
        xtb_opt.optimize(struct)
        struct.write(outfile)
        logging.info('done')
    else:
        logging.info(f'{method} is not implemented yet.')
        sys.exit('exitting')


def build_and_opt_cage(prefix, BB1, BB2, topology, macromod_,
                       settings=None, pdb=None, output_dir=None):
    """

    Keyword Arguments:
        prefix (str) - output file name prefix
        BB1 (str) - name of building block 1 file
        BB2 (str) - name of building block 2 file
        topology (stk.topology) = cage toplogy object
        macromod_ (str) - location of macromodel
        settings (dict) - settings for MacroModel Opt
        pdb (bool) - output PDB file of optimized cage
            (default is None)
        output_dir (str) - directory to save MacroModel output to
            (default is None)
    """
    # use standard settings applied in andrew_marsh work if md/settings
    # is None
    if settings is None:
        Settings = default_stk_MD_settings()
    else:
        Settings = settings

    # use default output dir (which is CWD) is output_dir is not given
    if output_dir is None:
        output_dir = Settings['output_dir']

    # try:
    cage = stk.Cage([BB1, BB2], topology)
    cage.write(prefix + '.mol')
    cage.dump(prefix + '.json')
    # restricted=True optimization with OPLS forcefield by default
    ff = stk.MacroModelForceField(
        macromodel_path=macromod_,
        restricted=True,
        output_dir=output_dir
    )
    # MD process - run MD, collect N conformers, optimize each,
    # return lowest energy conformer
    # no restricted
    md = stk.MacroModelMD(
        macromodel_path=macromod_,
        output_dir=output_dir,
        timeout=Settings['timeout'],
        force_field=Settings['force_field'],
        temperature=Settings['temperature'],
        conformers=Settings['conformers'],
        time_step=Settings['time_step'],
        eq_time=Settings['eq_time'],
        simulation_time=Settings['simulation_time'],
        maximum_iterations=Settings['maximum_iterations'],
        minimum_gradient=Settings['minimum_gradient'],
        use_cache=Settings['use_cache']
    )
    macromodel = stk.OptimizerSequence(ff, md)
    macromodel.optimize(mol=cage)
    cage.write(prefix + '_opt.mol')
    cage.dump(prefix + '_opt.json')
    if pdb is True:
        cage.write(prefix + '_opt.pdb')
    return cage
    # except MacroMoleculeBuildError:
    #     print('build failed')
    #     pass
