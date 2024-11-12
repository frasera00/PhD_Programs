import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd

class Plotter:
    def __init__(self, folder):
        self.folder = folder

    @staticmethod
    def stylize_plot(axis, xlab, ylab, labelsize=14, ticksize=12, hide=["right","top"], ticks = {"top":False,"bottom":True,"left":True,"right":False}):

        if hide:
            for h in hide:
                axis.spines[h].set_visible(False)
                ticks[h] = False
            
        axis.tick_params(which='minor', length=2, direction="in", top=ticks["top"], bottom=ticks["bottom"], left=ticks["left"], right=ticks["right"], labelsize=ticksize)
        axis.tick_params(which='major', length=4, direction="in", top=ticks["top"], bottom=ticks["bottom"], left=ticks["left"], right=ticks["right"], labelsize=ticksize)

        axis.set_ylabel(ylab,size=labelsize)
        axis.set_xlabel(xlab,size=labelsize)      
        return
    
    def plot_system(self, colors=None, names=None, outFile=None, universe=None, axes=None, s=2, zorders=None):
        dfs = {}
        
        if universe is None:
            try: 
                u = self.universe
            except AttributeError:
                raise Exception("No universe found")
        else:
            u = universe
        
        if not names:
            names = list(set(u.atoms.names))

        key_list = names
        for key in key_list:
            selection = u.select_atoms(f"name {key} or resname {key}")
            dfs[key] = pd.DataFrame(selection.positions, columns=["x","y","z"])
        
        if not colors:
            key_list = dfs.keys()
            color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = {k:c for k,c in zip(key_list,color_list)}
       
        from mpl_toolkits.mplot3d import Axes3D

        if axes is None:
            fig, axes = plt.subplots(1,3,figsize=(14,4),sharey=False)
        else:
            fig = axes[0].get_figure()
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        for key in names:
            if key != "unified" and key in dfs:  # Check if key exists in dfs
                i = 0
                for key in dfs:
                    if zorders is not None:
                        zorder = zorders[key]
                    else:
                        zorder = i
                    df = dfs[key]
                    ax1.scatter(df["x"], df["z"], marker='o', color=colors[key], s=s, zorder=zorder)
                    ax2.scatter(df["y"], df["z"], marker='o', color=colors[key], s=s, zorder=zorder)
                    ax3.scatter(df["x"], df["y"], marker='o', color=colors[key], s=s, zorder=zorder)
                    i+=1

        ax1.set_xlabel("x, $\AA$")
        ax1.set_ylabel("z, $\AA$")
        ax2.set_xlabel("y, $\AA$")
        ax2.set_ylabel("z, $\AA$")
        ax3.set_xlabel("x, $\AA$")
        ax3.set_ylabel("y, $\AA$")

        if outFile:    
            fig.savefig(outFile, dpi=350)

        return fig, axes

    def extract_sequences(self,arr):
        sequences = []
        current_sequence = []

        for i in range(len(arr)):
            if i > 0 and arr[i] != arr[i - 1] + 1:
                sequences.append(current_sequence)
                current_sequence = []

            current_sequence.append(arr[i])

        # Append the last sequence
        if current_sequence:
            sequences.append(current_sequence)

        return sequences

    def get_chains(self, polName, atomName="", axes=None, ax3D=None, frame=-1, colors=None, chains=None, outName=None, numOfChains=None, molSizes=None, plot3D=False, plot2D=False, cmap=False, alpha=1, pointSize=3):

        from MDAnalysis.analysis.distances import self_distance_array as dist
        from natsort import natsorted
        from glob import glob

        if self.universe.trajectory.n_frames == 0:
            try:
                filenames = natsorted(glob(f"{self.folder}/*part*.gro"))
                trajnames = natsorted(glob(f"{self.folder}/*nojump*.xtc"))
                u = mda.Universe(filenames[-1], trajnames[-1])
            except:
                raise Exception("No trajectory found")
        else:
            u = self.universe
        
        polys = u.select_atoms(f"name {atomName} or resname {polName}")
        resnames = list(set(polys.residues.resnames))

        if chains is None and numOfChains is None:
            raise Exception("No chains or number of chains specified")
        
        if numOfChains and chains is None:
            chains = np.random.choice(resnames, size=numOfChains, replace=False)
        
        print(f"Chains: {chains}")

        if not molSizes:
            molSizes = [int(s.split(polName[0])[1]) for s in resnames if s in chains]     
        
        if axes is None and (plot2D or plot3D):
            fig,axes = plt.subplots(1,2,figsize=(8,3),sharey=True)
        
        if plot2D:
            ax1 = axes[0]
            ax2 = axes[1]
            self.stylize_plot(ax1,"x ($\AA$)","z ($\AA$)")
            self.stylize_plot(ax2,"y ($\AA$)","z ($\AA$)")

        if plot3D:
            if not ax3D:
                fig3D = plt.figure(figsize=(10,5))
                ax3D = fig3D.add_subplot(111, projection='3d')
            ax3D.set_xlabel("x ($\AA$)")
            ax3D.set_ylabel("y ($\AA$)")
            ax3D.set_zlabel("z ($\AA$)")
        
        if colors is None:
            key_list = list(set(u.atoms.resnames))
            color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = {k:c for k,c in zip(resnames,color_list)}
        elif len(colors) == 1:
            colors = {k:colors[0] for k in resnames}
        
        ts = u.trajectory[frame]
        positions = {}

        for resname in resnames[:]:
            indices = u.select_atoms(f"resname {resname}").indices
            sequence = self.extract_sequences(indices)
            positions[resname] = []

            pos_seq = []
            for seq in sequence:
                monomers = u.select_atoms(f"index {seq[0]}:{seq[-1]}")
                pos = monomers.positions
                pos[:,2] = pos[:,2] - np.min(pos[:,2])
                pos_seq.append(pos)

            if len(pos_seq) > 1:
                idx = -1
            else:
                idx = 0
                
            positions[resname].append(pos_seq[idx])

        if plot3D or plot2D:
            for chain in chains:
                for p in positions[chain]:
                    if plot2D:
                        if cmap:
                            ax1.scatter(p[:,0], p[:,2], marker="o", s=3, c=p[:,2], cmap="viridis")
                            ax2.scatter(p[:,1], p[:,2], marker="o", s=3, c=p[:,2], cmap="viridis")
                            ax2_cb = plt.colorbar(ax2.scatter(p[:,1], p[:,2], marker="o", s=3, c=p[:,2], cmap="viridis"), ax=ax2)
                            ax2_cb.set_label("z ($\AA$)")
                        else:
                            ax1.scatter(p[:,0], p[:,2], marker="o", s=pointSize, c=colors[chain], alpha=alpha)
                            ax2.scatter(p[:,1], p[:,2], marker="o", s=pointSize, c=colors[chain], alpha=alpha)


                    elif plot3D:
                        if cmap:
                            ax3D.plot(p[:,0], p[:,1], p[:,2], ls="solid", marker="none", lw=0.8, alpha=0.7, color="xkcd:light gray")
                            ax3D.scatter(p[:,0], p[:,1], p[:,2], marker="o", s=10, c=p[:,2], cmap="viridis")
                            ax3D_cb = plt.colorbar(ax3D.scatter(p[:,0], p[:,1], p[:,2], marker="o", s=10, c=p[:,2], cmap="viridis"), ax=ax3D, pad=0.1)                    
                            ax3D_cb.set_label("z ($\AA$)")
                        else:
                            ax3D.plot(p[:,0], p[:,1], p[:,2], ls="solid", marker="none", lw=0.8, alpha=0.7, color="xkcd:light gray")
                            ax3D.scatter(p[:,0], p[:,1], p[:,2], marker="o", s=10, c=colors[chain])
                            ax3D_cb = plt.colorbar(ax3D.scatter(p[:,0], p[:,1], p[:,2], marker="o", s=10, c=colors[chain]), ax=ax3D, pad=0.1)                    
                            ax3D_cb.set_label("z ($\AA$)")


        if outName:
            fig.savefig(f"{self.folder}/{outName}.png", dpi=350)

        if plot2D and not plot3D:
            return [ax1,ax2],resnames,positions
        if plot3D and not plot2D:
            return ax3D,resnames,positions
        if plot2D and plot3D:
            return [ax1,ax2],ax3D,resnames,positions
        else:
            return resnames,positions