#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for stk optimizer usage.

Author: Andrew Tarzia

Date Created: 10 Dec 2019
"""

import stk
import os
from os.path import exists, join
import glob
import matplotlib.pyplot as plt
import logging

from .plotting import scatter_plot


def default_stk_MD_settings():
    """
    Default settings from stk source code as of 26/04/19.

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
    struct = stk.BuildingBlock.init_from_file(infile)
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
        raise NotImplementedError(f'{method} is not implemented yet.')


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
    cage = stk.ConstructedMolecule([BB1, BB2], topology)
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


def MOC_rdkit_opt(cage, cage_name, do_long):
    """
    Perform RDKit optimisation of MOC.

    Parameters
    ----------
    cage : :class:`stk.ConstructedMolecule`
        Cage to be optimised.

    cage_name : :class:`str`
        Name of cage.

    Returns
    -------
    cage : :class:`stk.ConstructedMolecule`
        Optimised cage.

    """

    # TODO: Add more arguments and options.
    print('doing rdkit optimisation')
    optimizer = stk.MetalOptimizer(
        metal_binder_distance=1.9,
        metal_binder_fc=1.0e2,
        binder_ligand_fc=0.0,
        ignore_vdw=False,
        rel_distance=None,
        res_steps=50,
        restrict_bonds=True,
        restrict_angles=True,
        restrict_orientation=True,
        max_iterations=40,
        do_long_opt=do_long
    )

    optimizer.optimize(mol=cage)

    return cage


def MOC_uff_opt(cage, cage_name, metal_FFs):
    """
    Perform UFF4MOF optimisation of MOC.

    Parameters
    ----------
    cage : :class:`stk.ConstructedMolecule`
        Cage to be optimised.

    cage_name : :class:`str`
        Name of cage.

    Returns
    -------
    cage : :class:`stk.ConstructedMolecule`
        Optimised cage.

    """

    print('doing UFF4MOF optimisation')
    gulp_opt = stk.GulpMetalOptimizer(
        gulp_path='/home/atarzia/software/gulp-5.1/Src/gulp/gulp',
        metal_FF=metal_FFs,
        output_dir=f'cage_opt_{cage_name}_uff'
    )
    gulp_opt.assign_FF(cage)
    gulp_opt.optimize(mol=cage)

    return cage


def MOC_MD_opt(
    cage,
    cage_name,
    integrator,
    temperature,
    N,
    timestep,
    equib,
    production,
    opt_conf,
    metal_FFs,
    save_conf=False
):
    """
    Perform UFF4MOF molecular dynamics of MOC.

    Parameters
    ----------
    cage : :class:`stk.ConstructedMolecule`
        Cage to be optimised.

    cage_name : :class:`str`
        Name of cage.

    Returns
    -------
    cage : :class:`stk.ConstructedMolecule`
        Optimised cage.

    """

    print('doing UFF4MOF MD')
    gulp_MD = stk.GulpMDMetalOptimizer(
        gulp_path='/home/atarzia/software/gulp-5.1/Src/gulp/gulp',
        metal_FF=metal_FFs,
        output_dir=f'cage_opt_{cage_name}_MD',
        integrator=integrator,
        ensemble='nvt',
        temperature=temperature,
        equilbration=equib,
        production=production,
        timestep=timestep,
        N_conformers=N,
        opt_conformers=opt_conf,
        save_conformers=save_conf
    )
    gulp_MD.assign_FF(cage)
    gulp_MD.optimize(cage)

    return cage


def MOC_xtb_conformers(
    cage,
    cage_name,
    etemp,
    output_dir,
    conformer_dir,
    nc,
    free_e,
    charge,
    opt=False,
    opt_level=None,
    solvent=None,
    handle_failure=False
):
    """
    Perform GFN2-xTB conformer scan of MOC.

    Parameters
    ----------
    cage : :class:`stk.ConstructedMolecule`
        Cage to be optimised.

    cage_name : :class:`str`
        Name of cage.

    Returns
    -------
    cage : :class:`stk.ConstructedMolecule`
        Optimised cage.

    """

    if not exists(output_dir):
        os.mkdir(output_dir)

    if solvent is None:
        solvent_str = None
        solvent_grid = 'normal'
    else:
        solvent_str, solvent_grid = solvent

    print('doing XTB conformer sorting by energy')
    conformers = glob.glob(f'{conformer_dir}/conf_*.xyz')
    ids = []
    energies = []
    min_energy = 10E20
    for file in sorted(conformers):
        id = file.replace('.xyz', '').split('_')[-1]
        cage.update_from_file(file)
        opt_failed = False
        if opt:
            print(f'optimising conformer {id}')
            xtb_opt = stk.XTB(
                xtb_path='/home/atarzia/software/xtb-190806/bin/xtb',
                output_dir=f'opt_{cage_name}_{id}',
                gfn_version=2,
                num_cores=nc,
                opt_level=opt_level,
                charge=charge,
                num_unpaired_electrons=free_e,
                max_runs=1,
                electronic_temperature=etemp,
                calculate_hessian=False,
                unlimited_memory=True,
                solvent=solvent_str,
                solvent_grid=solvent_grid
            )
            try:
                xtb_opt.optimize(mol=cage)
                cage.write(join(f'{output_dir}', f'conf_{id}_opt.xyz'))
            except stk.XTBConvergenceError:
                if handle_failure:
                    opt_failed = True
                else:
                    raise stk.XTBConvergenceError()

        print(f'calculating energy of {id}')
        # Extract energy.
        xtb_energy = stk.XTBEnergy(
            xtb_path='/home/atarzia/software/xtb-190806/bin/xtb',
            output_dir=f'ey_{cage_name}_{id}',
            num_cores=nc,
            charge=charge,
            num_unpaired_electrons=free_e,
            electronic_temperature=etemp,
            unlimited_memory=True,
            solvent=solvent_str,
            solvent_grid=solvent_grid
        )
        if handle_failure and opt_failed:
            energy = 10E24
        else:
            energy = xtb_energy.get_energy(cage)
        if energy < min_energy:
            min_energy_conformer = file
            min_energy = energy
        ids.append(id)
        energies.append(energy)

    print('done', min_energy, min_energy_conformer)
    cage.update_from_file(min_energy_conformer)

    energies = [(i-min(energies))*2625.5 for i in energies]
    fig, ax = scatter_plot(
        X=ids, Y=energies,
        xtitle='conformer id',
        ytitle='rel. energy [kJ/mol]',
        xlim=(0, 201),
        ylim=(-5, 1000)
    )

    fig.tight_layout()
    fig.savefig(
        join(output_dir, f'{cage_name}_conf_energies.pdf'),
        dpi=720,
        bbox_inches='tight'
    )
    plt.close()


def MOC_xtb_opt(
    cage,
    cage_name,
    nc,
    opt_level,
    etemp,
    charge,
    free_e,
    solvent=None
):
    """
    Perform GFN2-xTB optimisation of MOC.

    Parameters
    ----------
    cage : :class:`stk.ConstructedMolecule`
        Cage to be optimised.

    cage_name : :class:`str`
        Name of cage.

    Returns
    -------
    cage : :class:`stk.ConstructedMolecule`
        Optimised cage.

    """

    if solvent is None:
        solvent_str = None
        solvent_grid = 'normal'
    else:
        solvent_str, solvent_grid = solvent

    print('doing XTB optimisation')
    xtb_opt = stk.XTB(
        xtb_path='/home/atarzia/software/xtb-190806/bin/xtb',
        output_dir=f'cage_opt_{cage_name}_xtb',
        gfn_version=2,
        num_cores=nc,
        opt_level=opt_level,
        charge=charge,
        num_unpaired_electrons=free_e,
        max_runs=1,
        electronic_temperature=etemp,
        calculate_hessian=False,
        unlimited_memory=True,
        solvent=solvent_str,
        solvent_grid=solvent_grid
    )
    xtb_opt.optimize(mol=cage)
