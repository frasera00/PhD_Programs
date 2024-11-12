import numpy as np
from scipy.interpolate import griddata

class LayerHeight:
    def __init__(self, folder):
        self.folder = folder

    def calc_LayerHeight(self, selPoly, selLay, grid_size_x=None, grid_size_y=None, folder=None):
        try:
            u = self.universe
        except AttributeError:
            raise Exception("No universe found")
        
        if not folder:
            folder = self.folder

        LAY = u.select_atoms(f"{selLay}")
        DMS = u.select_atoms(f"{selPoly}")
        x = DMS.positions[:,0]*.1
        y = DMS.positions[:,1]*.1
        z = DMS.positions[:,2]*.1
        z -= LAY.positions[:,2].max()*.1

        box = u.dimensions[:2]
        if grid_size_x is None:
            grid_size_x = int(box[0])
        if grid_size_y is None:
            grid_size_y = int(box[1])

        x_bins = np.linspace(np.min(x), np.max(x), grid_size_x + 1)
        y_bins = np.linspace(np.min(y), np.max(y), grid_size_y + 1)
        grid_x, grid_y = np.meshgrid(np.linspace(np.min(x), np.max(x), grid_size_x),
                                    np.linspace(np.min(y), np.max(y), grid_size_y))
                            
        # Create an array to hold the maximum z-values
        max_z_grid = np.full((grid_size_y, grid_size_x), np.nan)

        # Bin the points and find the maximum z in each bin
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                mask = (x >= x_bins[i]) & (x < x_bins[i+1]) & (y >= y_bins[j]) & (y < y_bins[j+1])
                if np.any(mask):
                    max_z_grid[j, i] = np.max(z[mask])

        # Handle the edges
        grid_z = max_z_grid[~np.isnan(max_z_grid)]

        # Interpolate the max_z_grid over the grid_x and grid_y
        #grid_z = griddata((grid_x.flatten(), grid_y.flatten()), max_z_grid.flatten(), (grid_x, grid_y), method='cubic')
        #grid_z = np.nan_to_num(grid_z)
        mean_z = np.mean(grid_z)
        std_z = np.std(grid_z)

        return mean_z, std_z