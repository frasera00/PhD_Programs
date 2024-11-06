import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from scipy.signal import find_peaks
import pandas as pd

class DensityProfile():
    def __init__(self, folder):
        self.folder = folder

    def normalize_curves(self, x1, x2, curve1, curve2):

        def find_plateau_points(curve, window_size=11, polyorder=3):
            from scipy.signal import savgol_filter
            #smoothed_curve = savgol_filter(curve, window_size, polyorder)
            smoothed_curve = curve
            derivative = np.gradient(smoothed_curve)
            left = np.where(derivative==np.max(derivative))[0]
            right = np.where(derivative==np.min(derivative))[0]

            return left,right

        df1 = pd.DataFrame(np.column_stack([x1,curve1]), columns=["z","vol_frac"])
        df2 = pd.DataFrame(np.column_stack([x2,curve2]), columns=["z","vol_frac"])
        df1_roll = df1.rolling(window=2).mean().dropna()
        df2_roll = df2.rolling(window=2).mean().dropna()
        curve1  = df1_roll["vol_frac"].values
        curve2  = df2_roll["vol_frac"].values
        x1 = df1_roll["z"].values
        x2 = df2_roll["z"].values

        _, idx1 = find_plateau_points(curve1)
        idx2,_ = find_plateau_points(curve2)

        c1 = (1 - curve2[idx1])/(curve1[idx1])
        curve1 = curve1*c1
        curve2 = curve2
        
        curve_sum = curve1 + curve2
        max_sum = np.max(curve_sum)

        curve1 = curve1/max_sum
        curve2 = curve2/max_sum
        curve2 = curve2/np.max(curve2)

        return x1, x2, curve1, curve2

    def calc_kde(self, z_SOLV=None, z_POLY=None, solvent=None, clip=(0,600),cut=600, outFolder="", system="System", save=False):
        
        fig_d,ax_d = plt.subplots()
        my_kde = sns.kdeplot(z_POLY, clip=clip, cut=cut, bw_adjust=1.5, ax=ax_d)
        line = my_kde.lines[0]
        x, y = line.get_data()

        y1 = 0
        if solvent != "vacuum":
            fig_d,ax_d = plt.subplots()
            my_kde1 = sns.kdeplot(z_SOLV, clip=clip, cut=cut, bw_adjust=1.5, ax=ax_d)
            line1 = my_kde1.lines[0]
            x1, y1 = line1.get_data()
            
        x, x1, y, y1 = self.normalize_curves(x, x1, y, y1)

        x = x*0.1  #to nm
        x1 = x1*0.1  #to nm
        Sum = y1+y

        fig_s, ax_s = plt.subplots()
        ax_s.plot(x,Sum,label="sum")
        ax_s.legend(frameon=False)

        kde1 = np.array(np.column_stack([x,y]))
        kde2 = np.array(np.column_stack([x1,y1]))
        if save:
            df = pd.DataFrame(kde1, columns=["z","vol_frac"])
            df.to_csv(f"{outFolder}/kde_{system}_poly.csv",index=False)
            df = pd.DataFrame(kde2, columns=["z","vol_frac"])
            df.to_csv(f"{outFolder}/kde_{system}_solv.csv",index=False)
        return kde1, kde2

    def plot_kde(self, ax, x, y, color, label="", fill=False, zorder=1, alpha=0.2, ls="solid"):
        if fill:
            ax.fill_between(x, y, alpha=alpha, color=color, zorder=zorder)
            ax.plot(x,y, c=color, ls=ls, lw=2, zorder=zorder, label=label)
        else:
            ax.plot(x,y, c=color, ls=ls, lw=2, label=label)

        return ax

    def density_profile(self, mol, nframes=10):
        from MDAnalysis.analysis.density import DensityAnalysis

        try:
            u = self.universe
        except AttributeError:
            raise AttributeError("No universe loaded. Create one instantiating Grafing.NewSystem() class.")

        lastframe = u.trajectory[-1]

        totalframes = len(u.trajectory)
        start = int( totalframes - nframes )
        if start < 0:
            start = totalframes
            nframes = totalframes
        print(f"{nframes} frames from {totalframes} total.")

        density, density_SOLV, den_side_x, den_side_y, den_up = [], [], [], [], []
        if mol != "solvent":
            LAY = u.select_atoms("name N1L1 or name N1L")
            SURF = np.max(LAY.positions[:,2]) + 1
            POLY = u.select_atoms(f"name {mol} and prop z > {SURF}", updating=True)

            print(f"{POLY.n_atoms} atoms of type {mol} selected.")
            D = DensityAnalysis(POLY, delta=1.0)
            D.run(start=start, verbose=True)
            DENS_edges = D.results.density.edges
            density = D.results.density.grid

            den_side_x = np.mean(density[:,:,:],axis=1)
            den_side_y = np.mean(density[:,:,:],axis=0)
            den_side_x = np.swapaxes(den_side_x,0,1)
            den_side_y = np.swapaxes(den_side_y,0,1)
            den_up = np.mean(density[:,:,:],axis=2)
            den_up = np.swapaxes(den_up,0,1)
        else:
            LAY = u.select_atoms("name N1L1 or name N1L")
            SURF = np.max(LAY.positions[:,2]) + 1
            SOLV = u.select_atoms(f"resname W or resname SW or resname TW or resname TOLU or resname OCT and prop z > {SURF}", updating=True)
            D_SOLV = DensityAnalysis(SOLV, delta=1.0)
            D_SOLV.run(start=start, verbose=True)
            DENS_edges_SOLV = D_SOLV.results.density.edges
            density_SOLV = D_SOLV.results.density.grid

        return  den_side_x, den_side_y, den_up, density, density_SOLV

    def position_hist(self,mol,nframes=10):  
        try:
            u = self.universe
        except AttributeError:
            raise AttributeError("No universe loaded. Create one instantiating Grafing.NewSystem() class.")

        LAY = u.select_atoms("name N1L1 or name N1L")
        SURF = np.max(LAY.positions[:,2]) + 1
        DMS = u.select_atoms(f"name {mol} and prop z > {SURF}", updating=True)
        SOLV = u.select_atoms(f"resname W or resname SW or resname TW or resname TOLU or resname OCT and prop z > {SURF}", updating=True)

        z_SOLV, z_POLY = [], []
        for t in tqdm(u.trajectory[-nframes:]):
            if SOLV:
                z_SOLV.extend(SOLV.positions[:,2])
            z_POLY.extend(DMS.positions[:,2])

        return z_SOLV,z_POLY