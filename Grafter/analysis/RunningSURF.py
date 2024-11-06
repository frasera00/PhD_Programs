import seaborn as sns
from natsort import natsorted
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from MDAnalysis.transformations import wrap

class Surface():
    def __init__(self, folder):
        self.folder = folder

    def voxelize(self, points, voxel_size=(1, 1, 1)):
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
        grid = np.zeros(grid_shape, dtype=bool)
        
        voxel_indices = ((points - min_coords) / voxel_size).astype(int)
        grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        
        return grid, min_coords, max_coords

    def pad_voxel_grids(self, voxel_grids):
        max_shape = np.max([grid.shape for grid in voxel_grids], axis=0)
        padded_grids = [np.pad(grid, [(0, max_dim - grid_dim) for grid_dim, max_dim in zip(grid.shape, max_shape)], mode='constant', constant_values=0) for grid in voxel_grids]
        return padded_grids

    def average_voxel_grids(self, voxel_grids):
        sum_grid = np.sum(voxel_grids, axis=0)
        avg_grid = sum_grid / len(voxel_grids)
        return avg_grid

    def calculate_surface_area(self, grid, voxel_size=(1, 1, 1), min_coords=(0,0,0), level=None, out=None, maxlen=1):
        verts, faces, _, _ = marching_cubes(grid, level=level)
        verts_nm = verts * np.array(voxel_size)  # Convert vertices to nanometers
        self.plot_surface(grid, verts_nm, faces, voxel_size=voxel_size, min_coords=min_coords, out=out+"_surface.svg")
        self.plot_surface(grid, verts_nm, faces, voxel_size=voxel_size, min_coords=min_coords, out=out+"_surface.png")


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

    def calculate_cluster_area(self, vertices, max_edge_length=None):
        tri = Delaunay(vertices)

        def triangle_area(pts):
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        total_area = 0
        triangles_to_plot = []
        for simplex in tri.simplices:
            vertices = tri.points[simplex]
            
            if max_edge_length is not None:
                edge_lengths = np.linalg.norm(np.diff(np.vstack([vertices, vertices[0]]), axis=0), axis=1)
                if np.any(edge_lengths > max_edge_length):
                    continue
            
            total_area += triangle_area(vertices)
            triangles_to_plot.append(simplex)

        return total_area, tri, triangles_to_plot

    def plot_cluster_triangles(self, ax, vertices, tri, triangles_to_plot, label=None, color='blue'):
        ax.triplot(vertices[:, 0], vertices[:, 1], triangles_to_plot, color=color, alpha=0.3)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_aspect('equal')

    def projected_area_and_plot(self, vertices, eps=0.5, min_samples=3, max_edge_length=1, out=None):
        xy_points = vertices[:, :2]

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_points)
        labels = db.labels_
        unique_labels = np.unique(labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
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
            
            self.plot_cluster_triangles(ax, cluster_points, tri, triangles_to_plot, label=f'Cluster {k}', color=colors(idx))
            
            centroid_x, centroid_y = np.mean(cluster_points[:, 0]), np.mean(cluster_points[:, 1])
            ax.text(centroid_x, centroid_y, f'Area: {area:.2f}', fontsize=8, ha='center', va='center', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7))

        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title('Delaunay Triangulation of Projected Points')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if out:
            fig.savefig(out, dpi=450, bbox_inches="tight")
        
        return total_area

    def plot_surface(self, grid, verts, faces, voxel_size=(1, 1, 1), min_coords=(0, 0, 0), out=None):
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            print("Error: No vertices or faces extracted.")
            return
        
        print(f"Number of vertices: {verts.shape[0]}")
        print(f"Number of faces: {faces.shape[0]}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        ax.add_collection3d(mesh)
        
        ax.set_xlabel('x (nm)', labelpad=10)
        ax.set_ylabel('y (nm)', labelpad=10)
        ax.set_zlabel('z (nm)', labelpad=10)
        
        ax.set_xlim(min_coords[0], min_coords[0] + grid.shape[0] * voxel_size[0])
        ax.set_ylim(min_coords[1], min_coords[1] + grid.shape[1] * voxel_size[1])
        ax.set_zlim(min_coords[2], min_coords[2] + grid.shape[2] * voxel_size[2])
        
        # Show only the first and last ticks on the z-axis
        ax.set_xticks(ax.get_xticks()[::len(ax.get_xticks())-1])
        ax.set_yticks(ax.get_yticks()[::len(ax.get_yticks())-1])
        ax.set_zticks(ax.get_zticks()[::len(ax.get_zticks())-1])

        ax.set_aspect('equal')
        ax.set_title("Marching cubes surface")

        from scipy.interpolate import griddata

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method='linear')

        fig2, ax2 = plt.subplots()
        contourf = ax2.contourf(xi, yi, zi, levels=15, cmap='viridis')  # Use a colormap of your choice
        background_color = plt.cm.viridis(0)   # the last color in cmap
        ax2.set_fc(background_color)
        ax2.set_title('Contour lines of the surface')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')
        ax2.set_aspect('equal')
        # Add a colorbar to show the mapping from colors to values
        cbar = fig2.colorbar(contourf, orientation='horizontal', pad=0.1, location="top")
        cbar.set_label('Height (nm)')

        # If you want to save this figure
        if out:
            fig2.savefig(f"{out}_contour.png", dpi=350, bbox_inches="tight")

        fig1, ax1 = plt.subplots()
        ax1.scatter(verts[:, 0], verts[:, 2], s=1)
        ax1.set_xlim(min_coords[0], min_coords[0] + grid.shape[0] * voxel_size[0])
        ax1.set_ylim(min_coords[2], min_coords[2] + grid.shape[2] * voxel_size[2])
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('z (nm)')
        ax1.set_aspect('equal')
        ax1.set_title('Vertices in x-z plane')

        fig.tight_layout()
        fig1.tight_layout()
        fig2.tight_layout()

        if out:
            fig.savefig(out, dpi=350, bbox_inches="tight")


    def average_surface_area_and_visualize(self, positions, voxel_size=(1, 1, 1), sigma=False, layers_to_ignore=False, level=0.5, out=None, maxlen=1):
        
        points = positions
        grid, min_coords, max_coords = self.voxelize(points, voxel_size)
        
        grid_shape = grid.shape
        if layers_to_ignore:
            grid = grid[layers_to_ignore:-layers_to_ignore, layers_to_ignore:-layers_to_ignore, :]  # Ignore layers at minimum x
        
        padded_voxel_grids = self.pad_voxel_grids([grid])
        avg_voxel_grid = self.average_voxel_grids(padded_voxel_grids)
        
        # Apply Gaussian smoothing
        if sigma is not False:
            smoothed_grid = gaussian_filter(avg_voxel_grid.astype(float), sigma=sigma)
            binary_grid = smoothed_grid >= 0.5
        else:
            binary_grid = avg_voxel_grid
        
        base_area = (binary_grid.shape[0]-1)*np.array(voxel_size)[0] * (binary_grid.shape[1]-1)*np.array(voxel_size)[1]
        surface_area, projection_area_xy, verts, faces = self.calculate_surface_area(binary_grid, voxel_size, min_coords, level=level, out=out, maxlen=maxlen)
        
        return base_area, surface_area, projection_area_xy, verts, faces

    def calculate_rmsd_from_avg_height(self, points):
        z_values = points[:, 2]
        avg_height = np.mean(z_values)
        squared_displacements = (z_values - avg_height) ** 2
        mean_squared_displacement = np.mean(squared_displacements)
        rmsd = np.sqrt(mean_squared_displacement)
        
        return rmsd
