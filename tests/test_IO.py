import stk
from os.path import join
import os
import sys
import IO_tools
from ase.io import read
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


odir = 'IO_tests_output'
if not os.path.exists(odir):
    os.mkdir(odir)


def test_convert_MOL3000_2_PDB_XYZ():
    stk_mol = stk.BuildingBlock(
        smiles=f'[H]C(=O)C1=C([H])C([H])=C(C2=C([H])C([H])=C(C(C3=C'
        f'([H])C([H])=C(C4=C([H])C([H])=C(C([H])=O)C([H])=C4[H])C([H])'
        f'=C3[H])(C3=C([H])C([H])=C(C4=C([H])C([H])=C(C([H])=O)C([H])'
        f'=C4[H])C([H])=C3[H])C([H])([H])C([H])([H])C([H])([H])C([H])'
        f'([H])[H])C([H])=C2[H])C([H])=C1[H]'
    )
    no_atoms = len(stk_mol.atoms)
    stk_mol.write(join(odir, 'test.mol'))
    IO_tools.convert_MOL3000_2_PDB_XYZ(file=join(odir, 'test.mol'))
    # Check the number of atoms are the same.
    PDB = join(odir, 'test.pdb')
    XYZ = join(odir, 'test.xyz')

    pdb_mol = read(PDB)
    assert len(pdb_mol) == no_atoms
    xyz_mol = read(XYZ)
    assert len(xyz_mol) == no_atoms
