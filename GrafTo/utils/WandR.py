import numpy as np
import MDAnalysis as mda
import pandas as pd

class WriterAndReader:

    def __init__(self, folder):
        self.universe = None
        self.solvents = {}
        self.folder = folder
        
    def out_gro(self, name="initial_config.gro", u=None, outSizes=False, convert=1):
        if u:
            universe = u
        else:
            try:
                universe = self.universe
            except AttributeError:
                raise Exception("No universe found")
            
        filename = f"{self.folder}/{name}"
        universe.atoms.positions = np.array(universe.atoms.positions)*convert
        universe.dimensions[:2] = universe.dimensions[:2]*convert
        universe.dimensions[2] = universe.dimensions[2]+200*convert
        gro_writer = mda.coordinates.GRO.GROWriter(filename, n_atoms=len(universe.atoms))
        gro_writer.write(universe)
        gro_writer.close()

        if outSizes:
            out = open(f"{self.folder}/molSizes.dat", "w")
            for m in self.molSizes:
                out.write(f"{m}\n")
            out.close()

    def read_inputs_assembler(self, path_to_input_file):
        """
        Reads the inputs from a file or dictionary.

        Parameters:
        - path_to_input_file: The path to the input file or a dictionary.

        Raises:
        - Exception: If the input file is not a path to a .json file or a dictionary.
        """
        if isinstance(path_to_input_file, str):
            import json
            inputs = json.load(open(path_to_input_file,"r"))
        elif isinstance(path_to_input_file, dict):
            inputs = path_to_input_file
        else:
            raise Exception("Input file must be a path to a .json file or a dictionary")

        self.folder = inputs["folder"]
        self.blocks = inputs["blocks"]
        self.positions = np.array(inputs["positions"])
        self.transforms = [None]*len(self.blocks) if not inputs["transforms"] else inputs["transforms"]
        self.name = inputs["name"]
        self.outName = inputs["out name"]
        self.box = inputs["box dimensions"]

        print(f"\nname: {self.name}\nroot: {self.folder}\nblocks: {self.blocks}\npositions: {self.positions}\nbox: {self.box}\ntransformations: {self.transforms}\nout name: {self.outName}\n")

    def read_inputs_grafter(self,path_to_input_file):
        """
        Reads the inputs from a file or dictionary.

        Parameters:
        - path_to_input_file: The path to the input file or a dictionary.

        Raises:
        - Exception: If the input file is not a path to a .json file or a dictionary.
        """
        if isinstance(path_to_input_file, str):
            import json
            inputs = json.load(open(path_to_input_file,"r"))
        elif isinstance(path_to_input_file, dict):
            inputs = path_to_input_file
        else:
            raise Exception("Input file must be a path to a .json file or a dictionary")

        self.folder = inputs["folder"]
        self.surfDist = inputs["surface_distance"]*10
        self.grafDens = inputs["grafting_density"]
        self.matrix = ["file", inputs["matrix"]["file"]] if inputs["matrix"]["file"] else ["build", inputs["matrix"]["size"]]
        self.dispersity = ["mono", inputs["chain dispersity"]["monodisperse"]] if inputs["chain dispersity"]["monodisperse"] else ["poly", inputs["chain dispersity"]["polydisperse"]]
        self.surfGeometry = "cylindrical" if inputs["surface geometry"]["cylindrical"] else "flat"
        self.dictNames = inputs["atom names"]
        self.tiltMol = inputs["perturbation"]
        self.name = inputs["name"]
        self.graftingParams = [self.surfDist, self.grafDens, self.matrix, self.dispersity, self.surfGeometry, self.dictNames, self.tiltMol]

        print(f"\nname: {self.name}\nfolder: {self.folder}\nsurface distance: {self.surfDist*.1} nm\ngrafting density: {self.grafDens} gps/nm^3\nmatrix: {self.matrix}\ndispersity: {self.dispersity}\nsurface geometry: {self.surfGeometry}\natom names: {self.dictNames}\ntilt molecule: {self.tiltMol}\n")

        if self.matrix[0] == "file":
            self.universe = mda.Universe(self.matrix[1])
            self.dfs["bulk"] = pd.DataFrame(self.universe.atoms.positions, columns=["x","y","z"])

    def write_xyz(self,fname,type,df,nTot,mode="w"):
        fout = open(fname,mode)
        if mode == "w":
            fout.write("%d\n" % (nTot))
            fout.write("Atoms. Timestep: 0\n")
        for i,row in df.iterrows():
            fout.write("%1d %5f %5f %5f\n" % (type, row["x"], row["y"], row["z"]))
        fout.close()
        return

    def write_atoms(self,fname,label=None,mode="w",id=[],atomTp=0,rsdNm=0,atomNm=0,q=0,m=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        for i in id:
            fout.write("%5d%8s%8d%8s%8s%6d%9.1f%9.1f\n" % (i,atomTp,i,rsdNm,atomNm,i,q,m))
        fout.close()
        return

    def write_bonds(self,fname,label=None,mode="w",id=[],f=0,l_eq=0,k_eq=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        for i in id[:-1]:
            fout.write("%5d%6d%6d%10.3f%10.3f\n" % (i,i+1,f,l_eq,k_eq))
        fout.close()
        return

    def write_angles(self,fname,label=None,mode="w",id=[],f=0,angle=0,k_eq=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        fout.write("%5d%6d%6d%6d%10.3f%10.3f\n" % (id[0],id[1],id[2],f,angle,k_eq))
        fout.close()
        return

    def write_topol(self,fname,sysname,mols,includes,mode):
        fout = open(fname,mode)
        for i in includes:
            fout.write("%s\n" % i)
        if sysname:
            fout.write("\n[ system ]\n%s" % sysname)
            fout.write("\n\n[ molecules ]\n")
        if mols:
            for i in mols:      
                fout.write("%5s%10d\n" % (i[0],i[1]))
        fout.close()
        return