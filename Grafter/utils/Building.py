import numpy as np
import pandas as pd
import MDAnalysis as mda
import subprocess

class Builder:
    def __init__(self,folder):
        self.folder = folder

    def build_fcc(self, space,Lbox_x,Lbox_y,Lbox_z):
        a = 2*space/np.sqrt(2)

        d1 = a/np.sqrt(2)
        d2 = a/(np.sqrt(2)*2)
        d3 = a*np.sqrt(6)/4

        N_at_x = round(Lbox_x/d1)
        N_at_y = round(Lbox_y/d3)
        Nrow = Lbox_z

        #RIGHT BOX
        l_x = (N_at_x)*d1
        l_y = (N_at_y)*d3
        l_z = Nrow/2*a

        print(f'\nCreating fcc lattice')
        print(f'Dimensions: {l_x} {l_y} {l_z}')
        print(f'a:{a}, d1:{d1}, d2:{d2}, d3:{d3}')
        Ntot = N_at_x*N_at_y*Nrow
        N_layer1 = Ntot-N_at_x*N_at_y

        #fo = open("fcc.dat","w")
        #fo.write(f'Box: {l_x} {l_y} {l_z} \n Atoms (x,y,z): {N_at_x} {N_at_y} {Nrow} \n Ntot: {Ntot} \n')
        #fo.write(f'a={a}, d1={d1}, d2={d2}, d3={d3}')
        #fo.close()

        def three_layers(Nx,Ny,d1,d2,d3,zd,x,y,z):
            x0 = 0
            y0 = 0
            z0 = zd
            N_at_x = Nx
            N_at_y = Ny
            for j in range(0,N_at_y,1):
                if (j%2 != 0):
                    x0 += d2
                else:
                    x0 = 0
                for k in range(0,N_at_x,1):
                    x1 = x0+d1*k
                    if (x1<=l_x):
                        x.append(x1)
                    else:
                        x.append(x1-l_x)
                    if (y0<=l_y):
                        y.append(y0)
                    else:
                        y.append(y0-l_y)
                    z.append(z0)
                y0 += d3
            
            x1 = d1
            y1 = (2/3)*d3
            z1 = z0+a/np.sqrt(3)
            for j in range(0,N_at_y,1):
                if (j%2 != 0):
                    x1 += d2
                else:
                    x1 = d1
                for k in range(0,N_at_x,1):
                    x2 = x1+d1*k  
                    if (x2<=l_x):
                        x.append(x2)
                    else:
                        x.append(x2-l_x)
                    y.append(y1)
                    z.append(z1)
                y1 += d3

            x2 = d1+d2
            y2 = d3/3
            z2 = z0+a/np.sqrt(3)*2
            for j in range(0,N_at_y,1):
                if (j%2 != 0):
                    x2 += d2
                else:
                    x2 = d1+d2
                for k in range(0,N_at_x,1):
                    x3 = x2+d1*k
                    if (x3<=l_x):
                        x.append(x3)
                    else:
                        x.append(x3-l_x)
                    """
                    if (y2<=l_y):
                        y.append(y2)
                    else:
                        y.append(y2-l_y)
                    """
                    y.append(y2)
                    z.append(z2)
                y2 += d3

            return x,y,z

        x,y,z = [],[],[]
        layers = int(Nrow/3)
        for i in range(layers):
            z_d = (a/np.sqrt(3)*3)*i
            x,y,z = three_layers(N_at_x,N_at_y,d1,d2,d3,z_d,x,y,z)
        z_d = (a/np.sqrt(3)*3)*layers

        df = pd.DataFrame({"x":x, "y":y, "z":z})
        return df,l_x,l_y,l_z

    def solvate(self, gmxSource, cp, cs, out, top, pbcBox, maxN, folder=None):
        """
        solvates a system using GROMACS solvate command.

        Args:
            gmxSource (str): Path to the GROMACS source.
            cp (str): Path to the input coordinate file (.gro) of the system to be solvated.
            cs (str): Path to the solvent coordinate file (.gro).
            out (str): Path to the output solvated coordinate file (.gro).
            top (str): Path to the topology file (.top) of the system.
            pbcBox (tuple): Tuple containing the dimensions of the simulation box (x, y, z).
            maxN (int): Maximum number of solvent molecules to add.
            folder (str, optional): Path to the folder where the solvation will be performed. Defaults to None.

        Returns:
            None
        """
        import subprocess

        if not folder:
            folder = self.folder

        if top:
            app = f"-p {top}"
        else:
            app = " "

        command = f"mpirun -n 1 --bind-to none gmx_mpi solvate -cp {cp} -cs {cs} -o {out} -scale 1 -radius 0.2 " + app + f" -box {pbcBox[0]} {pbcBox[1]} {pbcBox[2]} -maxsol {maxN} > solvation.log 2>&1;"
        subprocess.run(f"cd {folder};" +
                       f"source {gmxSource} > /dev/null 2>&1;" +
                       command,
                       shell=True, executable="/bin/bash")

        self.universe = mda.Universe(f"{folder}/{out}")
        return

    @staticmethod
    def solv_box(folder,gmxSource,outName,solv,boxSize,positions,pbcBox,numMols,into=None): 
        """
        solvates a box with molecules using GROMACS.

        Parameters:
        - gmxSource (str): Path to the GROMACS source.
        - outName (str): Output file name.
        - solv (str): Path to the solvent molecule file.
        - boxSize (list): solvent box dimensions that goes inside simulation box [x, y, z].
        - positions (list): List of position ranges [[lower_x, upper_x], [lower_y, upper_y], [lower_z, upper_z]].
        - pbcBox (list): Simulation box coordinates [x, y, z].
        - numMols (int): Number of solvent molecules to insert.
        - folder (str, optional): Path to the folder where the commands will be executed. Defaults to None.

        Returns:
        - None
        """
        
        lower_x = positions[0][0]
        upper_x = positions[0][1]
        lower_y = positions[1][0]
        upper_y = positions[1][1]
        lower_z = positions[2][0]
        upper_z = positions[2][1]

        x = lower_x + (upper_x-lower_x)/2
        y = lower_y + (upper_y-lower_y)/2
        z = lower_z + (upper_z-lower_z)/2

        if into:
            extra = f"-f {into}"
        else:
            extra = " "

        subprocess.run(f"cd {folder};"+
                        f"source {gmxSource} > /dev/null 2>&1;"+
                        f"mpirun -n 1 --bind-to none gmx_mpi insert-molecules {extra} -ci {solv} -o {outName} -radius 0.2 -scale 1 -try {numMols*5} -box {boxSize[0]} {boxSize[1]} {boxSize[2]} -nmol {numMols} >> insert.log 2>&1;"+
                        f"mpirun -n 1 --bind-to none gmx_mpi editconf -f {outName} -center {x} {y} {z} -o centered_{outName} -box {pbcBox[0]} {pbcBox[1]} {pbcBox[2]} > centering.log 2>&1",
                        shell = True, executable="/bin/bash")
        return
