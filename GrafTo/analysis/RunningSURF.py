import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay

class Surface:
    """
    A class to represent a 3D surface and perform various operations on it, such as voxelization,
    smoothing, surface extraction, clustering, and area calculation.

    Attributes:
    positions (np.ndarray): An array of positions representing the surface points.
    """
    
    def __init__(self, positions: np.ndarray):
        """
        Initialize the Surface with given positions.

        Parameters:
        positions (np.ndarray): An array of positions.
        """
        self.positions = positions

    def voxelize(self, points: np.ndarray, voxel_size: tuple = (1, 1, 1)) -> tuple:
        """
        Convert points to a voxel grid.

        Parameters:
        points (np.ndarray): An array of points to be voxelized.
        voxel_size (tuple): The size of each voxel.

        Returns:
        tuple: A tuple containing the voxel grid, minimum coordinates, and maximum coordinates.
        """
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)

        grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
        voxel_grid = np.zeros(grid_shape, dtype=bool)
        
        voxel_indices = ((points - min_coords) / voxel_size).astype(int)
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        
        return voxel_grid, min_coords, max_coords

    def pad_voxel_grids(self, voxel_grids: list) -> list:
        """
        Pad a list of voxel grids to the same shape.

        Parameters:
        voxel_grids (list): A list of voxel grids to be padded.

        Returns:
        list: A list of padded voxel grids.
        """
        max_shape = np.max([grid.shape for grid in voxel_grids], axis=0)
        padded_grids = [
            np.pad(grid, [(0, max_dim - grid_dim) for grid_dim, max_dim in zip(grid.shape, max_shape)], mode='constant', constant_values=0)
            for grid in voxel_grids
        ]
        return padded_grids

    def average_voxel_grids(self, voxel_grids: list) -> np.ndarray:
        """
        Compute the average of a list of voxel grids.

        Parameters:
        voxel_grids (list): A list of voxel grids to be averaged.

        Returns:
        np.ndarray: The averaged voxel grid.
        """
        sum_grid = np.sum(voxel_grids, axis=0)
        avg_grid = sum_grid / len(voxel_grids)
        return avg_grid

    def calculate_surface_area(self, voxel_grid: np.ndarray, voxel_size: tuple = (1, 1, 1), min_coords: tuple = (0, 0, 0), level: float = 0.5, out: str = None, maxlen: float = 1, plot: bool = True) -> tuple:
        """
        Calculate the surface area of the voxel grid and plot the surface.

        Parameters:
        voxel_grid (np.ndarray): The voxel grid.
        voxel_size (tuple): The size of each voxel.
        min_coords (tuple): The minimum coordinates of the grid.
        level (float): The level at which to extract the surface.
        out (str): The output file name for the plot.
        maxlen (float): The maximum edge length for triangles to be considered.
        plot (bool): Whether to plot the surface.

        Returns:
        tuple: The surface area, projected area, vertices, and faces.
        """
        verts, faces, _, _ = marching_cubes(voxel_grid, level=level)
        verts_nm = verts * np.array(voxel_size)  # Convert vertices to nanometers

        if plot:
            self.plot_surface(verts_nm, faces, voxel_size=voxel_size, min_coords=min_coords, out=out+"_surface.png")

        # Calculate the area of each triangular face in the mesh
        area = 0
        for face in faces:
            v0 = verts_nm[face[0]]
            v1 = verts_nm[face[1]]
            v2 = verts_nm[face[2]]
            # Cross product of vectors v0v1 and v0v2 gives twice the area of the triangle
            area += np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2

        surface_area = area
        projected_area = self.projected_area_and_plot(verts_nm, eps=1, min_samples=3, max_edge_length=maxlen, out=out+"_projection.png")

        return surface_area, projected_area, verts, faces

    def calculate_cluster_area(self, vertices: np.ndarray, max_edge_length: float = None) -> tuple:
        """
        Calculate the area of clusters in the surface mesh.

        Parameters:
        vertices (np.ndarray): Vertices of the surface.
        max_edge_length (float): The maximum edge length for triangles to be considered.

        Returns:
        tuple: Total area, Delaunay triangulation, and triangles to plot.
        """
        tri = Delaunay(vertices)

        def triangle_area(pts: np.ndarray) -> float:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        total_area = 0
        triangles_to_plot = []
        for simplex in tri.simplices:
            triangle_vertices = tri.points[simplex]
            
            if max_edge_length is not None:
                edge_lengths = np.linalg.norm(np.diff(np.vstack([triangle_vertices, triangle_vertices[0]]), axis=0), axis=1)
                if np.any(edge_lengths > max_edge_length):
                    continue
            
            total_area += triangle_area(triangle_vertices)
            triangles_to_plot.append(simplex)

        return total_area, tri, triangles_to_plot

    def plot_cluster_triangles(self, ax: plt.Axes, vertices: np.ndarray, tri: Delaunay, triangles_to_plot: list, label: str = None, color: str = 'blue'):
        """
        Plot the clustered triangles using Matplotlib.

        Parameters:
        ax (plt.Axes): The Matplotlib axes to plot on.
        vertices (np.ndarray): Vertices of the surface.
        tri (Delaunay): The Delaunay triangulation of the vertices.
        triangles_to_plot (list): List of triangles to plot.
        label (str): Label for the plot.
        color (str): Color for the plot.
        """
        ax.triplot(vertices[:, 0], vertices[:, 1], tri.simplices, 'go-', lw=1.0)
        ax.plot(vertices[:, 0], vertices[:, 1], 'o')
        for simplex in triangles_to_plot:
            triangle_vertices = tri.points[simplex]
            polygon = plt.Polygon(triangle_vertices, edgecolor=color, fill=None, label=label)
            ax.add_patch(polygon)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_aspect('equal')

    def projected_area_and_plot(self, vertices: np.ndarray, eps: float = 0.5, min_samples: int = 3, max_edge_length: float = 1, out: str = None, plot: bool = True) -> float:
        """
        Calculate the projected area and plot the projection.

        Parameters:
        vertices (np.ndarray): Vertices of the surface.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        max_edge_length (float): The maximum edge length for triangles to be considered.
        out (str): The output file name for the plot.
        plot (bool): Whether to plot the projection.

        Returns:
        float: The projected area.
        """
        xy_points = vertices[:, :2]

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_points)
        labels = db.labels_
        unique_labels = np.unique(labels)
        
        if plot:
            fig, ax = plt.subplots(figsize=(5, 5))
            colors = plt.get_cmap("tab10", len(unique_labels))

        total_area = 0
        
        for idx, k in enumerate(unique_labels):
            if k == -1:
                continue
            
            class_member_mask = (labels == k)
            cluster_points = xy_points[class_member_mask]
            
            if len(cluster_points) < 3:
                continue
            
            area, tri, triangles_to_plot = self.calculate_cluster_area(cluster_points, max_edge_length=max_edge_length)
            total_area += area
            
            if plot:
                self.plot_cluster_triangles(ax, cluster_points, tri, triangles_to_plot, label=f'Cluster {k}', color=colors(idx))

        if plot:
            ax.set_title('Projected Area and Clusters')
            ax.legend()
            plt.tight_layout()
            if out:
                fig.savefig(out, dpi=350, bbox_inches="tight")
            plt.show()

        return total_area

    def plot_surface(self, verts: np.ndarray, faces: np.ndarray, voxel_size: tuple = (1, 1, 1), min_coords: tuple = (0, 0, 0), out: str = None):
        """
        Plot the 3D surface using Matplotlib.

        Parameters:
        verts (np.ndarray): Vertices of the surface.
        faces (np.ndarray): Faces of the surface.
        voxel_size (tuple): The size of each voxel.
        min_coords (tuple): The minimum coordinates of the grid.
        out (str): The output file name for the plot.
        """
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            print("Error: No vertices or faces extracted.")
            return

        print(f"Number of vertices: {verts.shape[0]}")
        print(f"Number of faces: {faces.shape[0]}")

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        ax.add_collection3d(mesh)

        # Set axis labels
        ax.set_xlabel('x (nm)', labelpad=10)
        ax.set_ylabel('y (nm)', labelpad=10)
        ax.set_zlabel('z (nm)', labelpad=10)

        # Set axis limits
        ax.set_xlim(min_coords[0], min_coords[0] + verts[:, 0].max() * voxel_size[0])
        ax.set_ylim(min_coords[1], min_coords[1] + verts[:, 1].max() * voxel_size[1])
        ax.set_zlim(min_coords[2], min_coords[2] + verts[:, 2].max() * voxel_size[2])

        # Show only the first and last ticks on the z-axis
        ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
        ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])
        ax.set_zticks([ax.get_zticks()[0], ax.get_zticks()[-1]])

        if out:
            fig.savefig(out, dpi=350, bbox_inches="tight")

        plt.show()

    def calculate_marching_cubes(self, voxel_size: tuple = (1, 1, 1), sigma: float = None, layers_to_ignore: int = 0, level: float = 0.5, out: str = None, maxlen: float = 1) -> tuple:
        """
        Calculate the marching cubes algorithm on the voxel grid and plot the surface.

        Parameters:
        voxel_size (tuple): The size of each voxel.
        sigma (float): The standard deviation for Gaussian kernel for smoothing.
        layers_to_ignore (int): The number of layers to ignore from the top and bottom.
        level (float): The level at which to extract the surface.
        out (str): The output file name for the plot.
        maxlen (float): The maximum edge length for triangles to be considered.

        Returns:
        tuple: The base area, surface area, projected area, vertices, and faces.
        """
        points = self.positions
        voxel_grid, min_coords, max_coords = self.voxelize(points, voxel_size)
        
        if layers_to_ignore > 0:
            voxel_grid = voxel_grid[layers_to_ignore:-layers_to_ignore, layers_to_ignore:-layers_to_ignore, :]  # Ignore layers at minimum x
        
        padded_voxel_grids = self.pad_voxel_grids([voxel_grid])
        avg_voxel_grid = self.average_voxel_grids(padded_voxel_grids)
        
        # Apply Gaussian smoothing (optional)
        if sigma is not None:
            smoothed_grid = gaussian_filter(avg_voxel_grid.astype(float), sigma=sigma)
            binary_grid = smoothed_grid >= 0.5
        else:
            binary_grid = avg_voxel_grid
        
        base_area = (binary_grid.shape[0] - 1) * voxel_size[0] * (binary_grid.shape[1] - 1) * voxel_size[1]
        surface_area, projection_area_xy, verts, faces = self.calculate_surface_area(binary_grid, voxel_size, min_coords, level=level, out=out, maxlen=maxlen)
        
        return base_area, surface_area, projection_area_xy, verts, faces

    def calculate_rmsd_from_avg_height(self, points):
        z_values = points[:, 2]
        avg_height = np.mean(z_values)
        squared_displacements = (z_values - avg_height) ** 2
        mean_squared_displacement = np.mean(squared_displacements)
        rmsd = np.sqrt(mean_squared_displacement)
        
        return rmsd
