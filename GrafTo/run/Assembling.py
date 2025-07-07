#!/usr/bin/env python
# coding: utf-8
    
import numpy as np
import MDAnalysis as mda
from GrafTo.utils import WandR as wr
from GrafTo.utils import Building as bld

class Assembler(bld.Builder, wr.WriterAndReader):

    """
    The Assembler class is responsible for assembling molecular structures and performing various operations on them.

    Attributes:
    - universe: The universe object representing the molecular structure.
    - solvents: A dictionary containing information about solvents.
    - folder: The folder path where the files are located.
    - input_file_path: The path to the input file.

    Methods:
    - write_gro: Writes the molecular structure to a GRO file.
    - run_assembler: Runs the assembler to assemble the blocks and generate the final structure.
    - universe_from_df: Creates a universe object from a dataframe.
    - build_slab: Builds a slab structure using the given parameters.
    """

    def __init__(self, folder):
        self.universe = None
        self.solvents = {}
        self.folder = folder

    def write_gro(self, u, outName):
        """
        Writes the molecular structure to a GRO file.

        Parameters:
        - u: The universe object representing the molecular structure.
        - outName: The name of the output GRO file.

        Returns:
        - None
        """
        gro_writer = mda.coordinates.GRO.GROWriter(f"{self.folder}/{outName}", n_atoms=len(u.atoms))
        gro_writer.write(u)
        gro_writer.close()

    def run_assembler(self, topol=None, molnames=None, folder=None, inputs=None):
        """
        Run the assembler to assemble the blocks and generate the final structure.

        Parameters:
        - topol (str): The name of the topology file.
        - molnames (list): A list of molecule names.
        - folder (str): The path to the folder containing the blocks.
        - inputs (dict): A dictionary containing additional inputs.

        Returns:
        - None
        """

        if folder:
            self.folder = folder

        if inputs:
            self.folder = inputs["folder"]
            self.blocks = inputs["blocks"]
            self.positions = np.array(inputs["positions"])
            self.transforms = [None]*len(self.blocks) if not inputs["transforms"] else inputs["transforms"]
            self.name = inputs["name"]
            self.outName = inputs["out name"]
            self.box = inputs["box dimensions"]

        print("\n.*.*.*.*.*.*.*.*.*.*.*.\nStarting assembling\n\nname: ",self.name,"\nroot: ",folder,"\nblocks: ",self.blocks,"\npositions: ",self.positions,"\nbox: ",self.box,"\ntransformations: ",self.transforms,"\nout name: ",self.outName,"\n")

        atom_groups = []
        n_atoms = 0

        for block,ts,ps in zip(self.blocks,self.transforms,self.positions):

            u = mda.Universe(f"{self.folder}/{block}")
            original_atoms = u.select_atoms("all")
            positions = original_atoms.positions
            new_atoms = original_atoms.copy()

            if ts:
                coords_to_flip = [index for index, value in enumerate(ts) if value]
                for coord in coords_to_flip:
                    positions[:,coord] = positions[:,coord]*(-1)

            #mins = np.array([positions[:,0].min(), positions[:,1].min(), positions[:,2].min()])
            #positions = positions - mins
            positions = positions + ps
            new_atoms.positions = positions
            atom_groups.append(new_atoms)
            n_atoms += new_atoms.n_atoms

        final_atoms = mda.Merge(*atom_groups)
        final_atoms.dimensions = self.box

        self.write_gro(final_atoms, self.outName)
        self.universe = final_atoms

        if topol and molnames:
            for mol in molnames:
                n = final_atoms.select_atoms(f"resname {mol} or name {mol}").n_residues
                with open(f"{self.folder}/{topol}", 'a') as f:
                    f.write(f"{mol} {n}\n")

        print("Finished assembling (:\n.*.*.*.*.*.*.*.*.*.*.*.\n\n")

    def universe_from_df(self, df, box=None, convert_unit=1):
        """
        Creates a universe object from a dataframe.

        Parameters:
        - df: The dataframe.
        - box: The box dimensions.

        Returns:
        - The universe object.
        """
        n_atoms = len(df)
        n_residues = len(df)
        resindices = df.index
        u = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=resindices, trajectory=True)
        u.atoms.positions = df[["x","y","z"]].values*convert_unit
        box = np.array(box)*convert_unit
        u.dimensions = np.array([*box,90,90,90])
        u.add_TopologyAttr('name', df["bead"].values)
        u.add_TopologyAttr('resname', df["type"].values)
        u.add_TopologyAttr('resid', list(range(1, n_residues+1)))

        return u

    def build_slab(self, outName, atomNames=["WALL","N1P"], Ns= [ 50, 10, 3 ], a=0.47*2**(1/6)):
        """
        Builds a slab structure using the given parameters.

        Parameters:
        - outName (str): The name of the output file.
        - atomNames (list, optional): The names of the atoms in the slab. Default is ["WALL","N1P"].
        - Ns (list, optional): The number of unit cells in each direction. Default is [50, 10, 9].
        - a (float, optional): The lattice constant. Default is 0.47*2**(1/6).

        Returns:
        - None
        """
        nx,ny,nz = Ns
        nz = nz*2.7
        df,lx,ly,lz = self.build_fcc(a,nx,ny,nz)
        df["type"],df["bead"] = atomNames

        u = self.universe_from_df(df, box=[lx,ly,lz], convert_unit=10)
        self.dfs["bulk"] = df
        self.universe = u
        self.write_gro(u, outName)

        return df

