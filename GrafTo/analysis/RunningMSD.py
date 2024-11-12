import pandas as pd
import numpy as np
from MDAnalysis.transformations import wrap

class MSD:
    def __init__(self, folder, universe=None):
        self.folder = folder
        self.universe = universe

    def diffusivity(self, lagtimes, msd, start_time, d):
        from scipy.stats import linregress
        linear_model = linregress(lagtimes[start_time:],msd[start_time:])
        slope = linear_model.slope
        error = linear_model.stderr
        D = slope * 1/(2*d)
        return D

    def calc_MSD(self, polyname, nframes=None, frange=None, outNames=None, distFromLayer=None, timestep=20):
        import MDAnalysis.analysis.msd as msd
        from scipy.interpolate import interp1d  

        try:
            u = self.universe
        except AttributeError:
            raise Exception("No universe found")

        if not nframes:
            nframes = len(u.trajectory)
            
        start = len(u.trajectory)-nframes

        # MSD all monomers
        monomers = u.select_atoms(f"name {polyname} or resname {polyname}")
        df_pos_all = pd.DataFrame(monomers.positions, columns=["x","y","z"])

        if not frange:
            frange = [df_pos_all["x"].min(), df_pos_all["x"].max()]

        # MSD top monomers
        xrange = np.linspace(frange[0], frange[1], 100)

        if  distFromLayer is not None:
            xprofile = []
            print("length all: " + str(len(df_pos_all)))
                
            for i in range(len(xrange)-1):
                try:
                    xprofile.append(monomers.select_atoms(f"prop x > {xrange[i]} and prop x < {xrange[i+1]}").positions[:,2].max())
                except:
                    xprofile.append(np.nan)
                
            yprofile = interp1d(xrange[:-1],xprofile,kind="linear",fill_value="extrapolate")
            mask = ((df_pos_all["z"] > yprofile(df_pos_all["x"])-distFromLayer) & (df_pos_all["x"]>=frange[0]) & (df_pos_all["x"]<=frange[1]))

            df_pos_close = df_pos_all.apply(lambda r: r[mask])
            print("length close: " + str(len(df_pos_close)))

            indices = monomers.indices[mask]
            selection = "index "+' '.join([str(i) for i in indices])
        else:
            df_pos_close = df_pos_all.copy()
            selection = "all"

        try:
            close_monomers = monomers.select_atoms(selection)
            MSD_close = msd.EinsteinMSD(close_monomers, select=selection, msd_type='xyz', fft=True)
            MSD_close.run(start=start, verbose=False)
        except:
            df_msd_time = pd.DataFrame({"time":[],"MSD":[]})
            return df_pos_all, df_pos_close, df_msd_time
        
        MSD_result_close = np.mean(MSD_close.results.msds_by_particle[::],axis=0)
        ts = u.trajectory[-1]

        lagtimes = np.arange(nframes)*timestep # make the lag-time axis
        df_msd_time = pd.DataFrame({"time":lagtimes,"MSD":MSD_close.results.timeseries})
        
        if outNames:
            dfs = [df_pos_all,df_pos_close,df_msd_time]
            for df, name in zip(dfs,outNames):
                df.to_csv(f"{name}",index=False)

        transform = wrap(u.atoms)
        u.trajectory.add_transformations(transform)
        u.trajectory[-1]

        zmin = df_pos_all["z"].min()
        df_pos_close = pd.DataFrame(close_monomers.positions, columns=["x","y","z"])
        df_pos_close["MSD"] = MSD_result_close
        df_pos_close["resname"] = close_monomers.residues.resnames

        df_pos_all = pd.DataFrame(monomers.positions, columns=["x","y","z"])
        df_pos_all["resname"] = monomers.residues.resnames

        df_pos_all["z"] =  df_pos_all["z"] - zmin
        df_pos_close["z"] =  df_pos_close["z"] - zmin
        
        return df_pos_all, df_pos_close, df_msd_time