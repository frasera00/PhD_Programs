# Standard library
import random
import subprocess
import sys
from collections import Counter
import re
from pathlib import Path
from typing import  (
    Optional, 
    List, 
    Dict, 
    Any, 
    Tuple, 
    Union,
    Literal 
)

# Third-party libraries
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.optimize import leastsq
from scipy.stats import gamma
import json

# Local application imports
from GrafTo.analysis import (
    MSD as msd,
    Density as density,
    Layer_height as layer_height,
    Plotting as plotter,
    Surface_reconstruction as surface,
)
from GrafTo.utils import (
    Building as builder,
    Writer_and_reader as writer_and_reader,
)
from GrafTo.run.Assembling import Assembler as Asm 

#defining a class called NewSystem
#inherits methods of all auxiliary classes

class NewSystem(Asm):
    """
    A class representing a new system for grafting.

    Attributes:
    - MAX_A: The maximum value for A.
    - MAX_DIST: The maximum distance.
    - folder: The root folder.
    - name: The name of the system.
    - dataframes: A dictionary of dataframes.
    - name_mapping: A dictionary of names to map bead types.
    - surf_dist: The surface distance.
    - grafting_density: The grafting density.
    - matrix: The matrix information.
    - dispersity: The chain dispersity information.
    - surf_geometry: The surface geometry.
    - tilt_angle: The tilt molecule information.
    - dimensions: The dimensions of the system.
    - uBulk: The universe object for the bulk.
    - universe: The universe object for the system.

    Methods:
    - read_inputs: Reads the inputs from a file or dictionary.
    - update_names: Updates the names of the dataframes.
    - update_dimensions: Updates the dimensions of the system.
    - create_cylindrical_pore: Makes a cylindrical pore in the matrix.
    - create_universe_from_df: Creates a universe object from a dataframe.
    - build_surface: Builds the surface of the system.
    - graft_matrix: Grafts the matrix onto the surface.
    - merge_dataframes: Creates a unified dataframe.
    - generate_itp_files: Generates the itp files.
    """
    
    MAX_A, MAX_DIST = 0.92, 25
    FCC_LATTICE_CONSTANT = 0.47 * 2**(1/6)  # Ångströms
    MIN_GRID_SIZE = 1
    MAX_GRID_SIZE = 1000  # Practical upper limit

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        gro_file: Optional[str] = None,
        traj_file: Optional[str] = None,
        system_name: str = "System",
        molecule_sizes: Optional[List[int]] = None,
    ) -> None:
    
        """Initializes a NewSystem object with components for simulation and analysis.

        Args:
            root_dir: Path to the root folder. Defaults to None.
            gro_file: Path to the .gro coordinate file. Defaults to None.
            traj_file: Path to the trajectory file (e.g., .xtc or .trr). Defaults to None.
            system_name: Name of the system. Defaults to "System".
            molecule_sizes: List of molecule sizes (e.g., atom counts per molecule). 
                    Defaults to an empty list.
        """

        if molecule_sizes is None:
            molecule_sizes = []

        # Core system attributes
        self._matrix: Optional[Union[List, str]] = None
        self.root_dir = Path(root_dir) if root_dir else None
        self.system_name: str = system_name
        self.surf_geometry: Optional[Literal["flat", "cylindrical", "slit"]] = None
        
        # Data containers
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.name_mapping: Dict[str, Union[str, List[str]]] = {}
        self.mol_sizes: List[int] = molecule_sizes or []
        self.dimensions: Optional[List[float]] = None
        self.universe: Optional[mda.Universe] = None
        self.uBulk: Optional[mda.Universe] = None
        
        # Initial setup
        self._setup(gro_file, traj_file)

    def _setup(self, 
               gro_file: Optional[str], 
               traj_file: Optional[str]
    ) -> None:
        
        """Initializes the MDAnalysis Universe with provided coordinate and trajectory files.

        Args:
            gro: Path to the .gro coordinate file. If None, no universe is created.
            traj: Path to the trajectory file (e.g., .xtc or .trr). If None, only the .gro file is loaded.
        """

        # Initialize components
        self.builder: builder.Builder = builder.Builder(self)
        self.writer_reader: writer_and_reader.WriterAndReader = writer_and_reader.WriterAndReader(self)
        self.plotter: plotter.Plotter = plotter.Plotter(self)
        self.msd_calculator: msd.MSD = msd.MSD(self)
        self.layer_analyzer: layer_height.LayerHeight = layer_height.LayerHeight(self)
        self.density_analyzer: density.DensityProfile = density.DensityProfile(self)
        self.surface_analyzer: surface.Surface = surface.Surface(self)

        if gro_file is None:
            return None

        if traj_file:
            self.universe = mda.Universe(gro_file, traj_file)
        else:
            self.universe = mda.Universe(gro_file)

    @property
    def universe(self) -> Optional[mda.Universe]:
        return self._universe

    @universe.setter
    def universe(self, value: mda.Universe) -> None:
        self._universe = value

    def plot_mol_distribution(
        self, 
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (6, 4)  # Explicit figure control
    ) -> plt.Axes:
        
        """Plots a histogram of molecule size distribution.

        Args:
            ax: Matplotlib Axes object. If None, a new figure/axes will be created.

        Returns:
            The Axes object with the plotted histogram.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)  # Configurable

        ax.hist(
            self.molecule_sizes,
            bins=100,
            alpha=0.5,
            color="xkcd:tomato",
            histtype="stepfilled",
            label="Molecule size distribution"
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Count")
        return ax
    
    def unwrap_coordinates(
        self, 
        coordinates: np.ndarray,  # "positions" → "coordinates"
        box_dimensions: np.ndarray  # "box" → "box_dimensions"
    ) -> np.ndarray:        
        
        """Unwraps periodic boundary conditions for a trajectory.

        Args:
            positions: Array of shape (n_frames, 3) containing particle coordinates.
            box: Array of shape (3,) containing box dimensions.

        Returns:
            Unwrapped coordinates as a numpy array of the same shape as input positions.
        """
        unwrapped_positions = np.zeros_like(coordinates)
        unwrapped_positions[0] = coordinates[0]  # First frame is reference

        for i in range(1, coordinates.shape[0]):
            delta = coordinates[i] - coordinates[i - 1]
            delta -= np.round(delta / box_dimensions) * box_dimensions
            unwrapped_positions[i] = unwrapped_positions[i - 1] + delta

        return unwrapped_positions
            
    def update_names(
        self,
        df_names: List[str],
        in_dict_names: Dict[str, Tuple[str, str]],
    ) -> None:
        
        """Updates the names and types of dataframes based on an input dictionary.

        Args:
            df_names: List of dataframe keys to update.
            in_dict_names: Dictionary mapping keys to (type, bead) tuples.
                        Example: {"lipid": ("L", "P"), "water": ("W", "OW")}

        Modifies:
            - self.dataframes: Updates 'type' and 'bead' columns for each key.
            - self.name_mapping: Adds new entries if keys are missing.
        """
        for key in df_names:
            self.dataframes[key]["type"] = in_dict_names[key][0]
            self.dataframes[key]["bead"] = in_dict_names[key][1]
            if key not in self.name_mapping:
                self.name_mapping[key] = in_dict_names[key]

    def get_molecule_sizes(
        self,
        selection: str,
        out_file: Optional[Union[str, Path]] = None,
    ) -> List[int]:
        
        """Extracts and returns unique molecule sizes from a selection, optionally saving to file.

        Args:
            selection: MDAnalysis atom selection string (e.g., "resname L*" or "type A").
            out_file: Optional path to save the sizes. Can be str or Path. If relative,
                    will be saved under self.root_dir.

        Returns:
            List of unique molecule sizes (integers), naturally sorted.

        Raises:
            ValueError: If no numerical component is found in resnames.
            AttributeError: If selection returns no atoms.

        Example:
            >>> system.get_molecule_sizes("resname W*", "water_sizes.txt")
            [3, 4, 5]
        """
        # Get residue names from selection
        try:
            resnames = self.universe.select_atoms(selection).residues.resnames
        except AttributeError as e:
            raise AttributeError(f"Selection '{selection}' returned no atoms") from e

        # Extract sizes with error handling
        molecule_sizes = set()
        for resname in resnames:
            match = re.search(r'\d+', resname)
            if not match:
                raise ValueError(f"Resname '{resname}' contains no numerical component")
            molecule_sizes.add(int(match.group()))

        # Sort naturally and handle output
        sorted_sizes = natsorted(molecule_sizes)
        
        if out_file:
            output_path = Path(self.root_dir) / str(out_file)
            with output_path.open('w') as f:
                f.writelines(f"{size}\n" for size in sorted_sizes)
        
        return sorted_sizes

    def update_dimensions(self, df: pd.DataFrame) -> None:

        """Calculates and updates system dimensions from spatial dataframe coordinates.

        Computes the bounding box dimensions (lx, ly, lz) by finding the range
        of coordinates in each axis.

        Args:
            df: DataFrame containing 'x', 'y', and 'z' coordinate columns.

        Raises:
            KeyError: If required coordinate columns are missing.
            ValueError: If DataFrame is empty.

        Example:
            >>> df = pd.DataFrame({'x': [0, 10], 'y': [0, 5], 'z': [0, 2]})
            >>> system.update_dimensions(df)
            >>> system.dimensions
            [10.0, 5.0, 2.0]
        """
        # Ensure dimensions is initialized
        # Validate input
        required_columns = {'x', 'y', 'z'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise KeyError(f"DataFrame missing required coordinate columns: {missing}")
        
        if df.empty:
            raise ValueError("Cannot compute dimensions from empty DataFrame")

        # Calculate dimensions
        try:
            lx = df["x"].max() - df["x"].min()
            ly = df["y"].max() - df["y"].min()
            lz = df["z"].max() - df["z"].min()
        except TypeError as e:
            raise TypeError("Coordinate columns must contain numeric values") from e

        # Update instance variable
        self.dimensions = [lx, ly, lz]
    
    def create_cylindrical_pore(
        self,
        df: pd.DataFrame,
        pore_radius: float,
        edge_margin: float = 10.0,
        radial_expansion: float = 7.0
    ) -> pd.DataFrame:
        
        """Creates a cylindrical pore in the structure by filtering coordinates.

        The method:
        1. Calculates the center point of the XY plane
        2. Removes points inside the core pore radius
        3. Maintains points either:
        - Within an expanded radius near the mid-section (z > margin and z < max_z - margin)
        - Outside the mid-section (z < margin or z > max_z - margin)

        Args:
            df: DataFrame containing 'x', 'y', 'z' coordinates
            pore_radius: Radius of the central pore to remove (Å)
            edge_margin: Z-axis margin to preserve at boundaries (Å) (default: 10.0)
            radial_expansion: Additional radius beyond pore_radius to keep in mid-section (Å) (default: 7.0)

        Returns:
            Filtered DataFrame with cylindrical pore

        Raises:
            KeyError: If required columns ('x', 'y', 'z') are missing
            ValueError: If pore_radius is negative or DataFrame is empty

        Example:
            >>> df = load_coordinates()
            >>> filtered_df = system.create_cylindrical_pore(df, pore_radius=20)
        """
        # Input validation
        required_columns = {'x', 'y', 'z'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise KeyError(f"Missing required columns: {missing}")
        
        if pore_radius < 0:
            raise ValueError("Pore radius must be non-negative")
        
        if df.empty:
            raise ValueError("Cannot create pore in empty DataFrame")

        # Calculate center and boundaries
        x_center, y_center = df[["x", "y"]].mean()
        max_z = df["z"].max()
        min_z = df["z"].min()
        
        # Calculate radial distances squared (for efficiency)
        radial_distance_sq = (df["x"] - x_center)**2 + (df["y"] - y_center)**2
        
        # Create filtering conditions
        outside_core_pore = radial_distance_sq >= pore_radius**2
        
        mid_section = (df["z"] > edge_margin) & (df["z"] < max_z - edge_margin)
        expanded_radius = radial_distance_sq < (pore_radius + radial_expansion)**2
        
        edge_section = (df["z"] <= edge_margin) | (df["z"] >= max_z - edge_margin)
        
        # Apply filters
        filtered_df = df[
            outside_core_pore & (
                (mid_section & expanded_radius) | edge_section
            )
        ].copy()
        
        # Clean up and return
        return filtered_df.reset_index(drop=True).round(decimals=3)

    @property
    def matrix(self) -> Union[List, str]:
        """Get the current matrix configuration.
        
        Returns:
            Either:
            - ["build", (nx, ny, nz)] for FCC construction
            - ["file", filename] for file loading
            - None if not set
        """
        return self._matrix

    @matrix.setter
    def matrix(self, value: Union[List, str]) -> None:
        """Set the matrix configuration with validation.
        
        Args:
            value: Matrix configuration to set
            
        Raises:
            ValueError: If configuration format is invalid
        """
        if value is None:
            self._matrix = None
            return
            
        if not isinstance(value, (list, str)):
            raise ValueError("Matrix must be list or str")
            
        # Validate build configuration
        if isinstance(value, list) and value[0] == "build":
            if len(value) != 2 or len(value[1]) != 3:
                raise ValueError(
                    "Build matrix format must be ['build', (nx, ny, nz)]"
                )
            nx, ny, nz = value[1]

            if not all(
                self.MIN_GRID_SIZE <= dim <= self.MAX_GRID_SIZE
                for dim in (nx, ny, nz)
            ):
                raise ValueError(
                    f"Grid dimensions must be between {self.MIN_GRID_SIZE} "
                    f"and {self.MAX_GRID_SIZE}"
                )
                
        # Validate file configuration
        elif isinstance(value, list) and value[0] == "file":
            if len(value) != 2:
                raise ValueError(
                    "File matrix format must be ['file', filename]"
                )
            if not isinstance(value[1], str):
                raise ValueError("Filename must be a string")
                
        else:
            raise ValueError(
                "Matrix must be ['build', (nx,ny,nz)] or ['file', filename]"
            )
            
        self._matrix = value

    def build_surface(
        self,
        names: Optional[Tuple[str, str]] = None,
        radius: Optional[float] = None,
        matrix: Optional[Union[List, str]] = None
    ) -> None:
        """Constructs the system surface from either file or built-in FCC lattice.

        Args:
            names: Tuple of (atom_name, residue_name) for surface particles.
                Defaults to self.name_mapping["bulk"] if None.
            radius: Radius for cylindrical pore creation (if surf_geometry is cylindrical).
            matrix: Either:
                - ["build", (nx, ny, nz)] for FCC lattice generation
                - ["file", filename] to load from file
                If None, uses self.matrix.

        Raises:
            ValueError: If matrix format is invalid or required data is missing.
            FileNotFoundError: If matrix source file doesn't exist.
            KeyError: If bulk names are not found in name_mapping.
        """
        # Set default values
        names = names or self.name_mapping.get("bulk")
        if not names:
            raise KeyError("No bulk names provided and none found in name_mapping")
        
        matrix = matrix or self.matrix
        if not matrix:
            raise ValueError("No matrix construction method provided")

        name_atom, name_residue = names

        # Handle matrix construction
        if matrix[0] == "build":
            self._build_fcc_surface(matrix[1], name_residue, name_atom)
        elif matrix[0] == "file":
            self._load_surface_from_file(name_residue)
        else:
            raise ValueError(f"Unknown matrix construction method: {matrix[0]}")

        # Apply pore geometry if specified
        if self.surf_geometry == "cylindrical" and radius is not None:
            self.dataframes["bulk"] = self.make_hole(self.dataframes["bulk"], radius)

        # Update system properties
        self.update_names(["bulk"], {"bulk": [name_residue, name_atom]})
        self.update_dimensions(self.dataframes["bulk"])
        
        # Create and merge universe
        self._create_and_merge_universe()

    def _build_fcc_surface(
        self,
        dimensions: Tuple[int, int, int],
        resname: str,
        atomname: str
    ) -> None:
        """Construct FCC lattice surface."""
        try:
            nx, ny, nz = dimensions
            a = 0.47 * 2**(1/6)  # FCC lattice constant
            df_bulk, lx, ly, lz = self._build_fcc(nx=nx, ny=ny, nz=nz, a=a)
            self.dataframes["bulk"] = df_bulk
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid build parameters in matrix") from e

    def _load_surface_from_file(self, resname: str) -> None:
        """Load surface coordinates from file."""
        try:
            u = mda.Universe("file")  # Note: Hardcoded filename should be parameterized
            atoms = u.select_atoms(f"resname {resname}")
            self.dataframes["bulk"] = (
                pd.DataFrame(atoms.positions * 0.1)  # Convert Å to nm
                .sort_values(by=["z", "y", "x"])
                .reset_index(drop=True)
            )
        except (OSError, AttributeError) as e:
            raise FileNotFoundError("Failed to load surface file") from e

    def _create_and_merge_universe(self) -> None:
        """Create universe from bulk dataframe and merge with existing."""
        self.uBulk = self.create_universe_from_df(
            self.dataframes["bulk"],
            box=self.dimensions,
            convert_unit=1
        )     

        try:
            self.universe = mda.Merge(self.universe, self.uBulk)
        except:
            self.universe = self.uBulk

    def _build_fcc(
            self,
            nx: int,
            ny: int,
            nz: int,
            a: Optional[float] = None
        ) -> Tuple[pd.DataFrame, float, float, float]:
            
            """Constructs an FCC lattice.
            
            Args:
                a: Lattice constant (Å). Defaults to class constant if None.
                nx: Number of unit cells in x-direction
                ny: Number of unit cells in y-direction
                nz: Number of unit cells in z-direction
                
            Returns:
                Tuple of:
                - DataFrame with particle positions
                - System dimensions (lx, ly, lz)
                
            Raises:
                ValueError: If grid sizes are invalid
            """
            a = a or self.FCC_LATTICE_CONSTANT
            
            # Validate grid sizes
            if not all(dim >= self.MIN_GRID_SIZE for dim in (nx, ny, nz)):
                raise ValueError(
                    f"Grid dimensions must be ≥ {self.MIN_GRID_SIZE}"
                )
                
            # FCC basis vectors
            basis = np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ])
            
            # Generate grid
            x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
            grid = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            
            # Calculate positions
            positions = []
            for point in grid:
                for base in basis:
                    positions.append(a * (point + base))
                    
            df = pd.DataFrame(positions, columns=["x", "y", "z"])
            
            # Calculate system dimensions
            lx, ly, lz = nx * a, ny * a, nz * a
            
            return df, lx, ly, lz
    
    def graft_matrix(
        self,
        tilt_angle: float = 0,
        surf_norm: str = "z"
    ) -> None:
        """Grafts polymer matrix onto the surface with optional tilt and dispersity control.

        Handles different surface geometries (flat, cylindrical, slit) and polymer
        dispersity options (monodisperse, polydisperse).

        Args:
            tilt_angle: Angle (degrees) to tilt grafted molecules (default: 0)
            surf_norm: Surface normal direction ('x', 'y', or 'z') (default: 'z')

        Raises:
            ValueError: For invalid surface normals or missing required data
            RuntimeError: If no chain size or distribution is provided
        """
        # Validate surface normal
        if surf_norm not in {'x', 'y', 'z'}:
            raise ValueError(f"Invalid surface normal: {surf_norm}. Must be 'x', 'y', or 'z'")

        # Process dispersity configuration
        self._process_dispersity()

        # Prepare surface layer based on geometry
        holes = self._prepare_surface_layer(surf_norm)

        # Perform grafting operation
        self._perform_grafting(
            tilt_angle=tilt_angle,
            surf_norm=surf_norm,
            holes=holes
        )

        # Update system state
        self._update_system_state()

    def _process_dispersity(self) -> None:
        """Process dispersity configuration and set appropriate attributes."""
        disp_type, disp_args = self.dispersity

        if disp_type == "poly" and disp_args:
            print(f"\nUsing Schultz-zimm distribution with args {disp_args}")
            self.chain_size = None
            self.fractions = None
            self.distribution_params = disp_args
        elif disp_type == "mono" and disp_args:
            self._process_monodisperse(disp_args)
            self.distribution_params = None
        else:
            raise RuntimeError("No chain size or distribution given, please provide one")

    def _process_monodisperse(self, chain_size: Union[int, List[Tuple[int, float]]]) -> None:
        """Handle monodisperse chain size configuration."""
        if len(chain_size) == 1:
            print(f"\nUsing chain size {chain_size[0]}")
            self.chain_size = chain_size[0]
            self.fractions = None
        else:
            print("\nUsing multiple chain sizes:")
            self.chain_size, self.fractions = [], []
            for cs, fraction in chain_size:
                print(f"Using chain size {cs} with fraction {fraction}")
                self.chain_size.append(cs)
                self.fractions.append(fraction)

    def _prepare_surface_layer(self, surf_norm: str) -> bool:
        """Prepare the surface layer based on geometry.
        
        Returns:
            bool: True if surface has holes (cylindrical/slit), False otherwise
        """
        if self.surf_geometry == "flat":
            norm_max = self.dataframes["bulk"][surf_norm].max()
            self.dataframes["layer"] = self.dataframes["bulk"][
                self.dataframes["bulk"][surf_norm] == norm_max
            ].copy()
            return False
        else:  # cylindrical or slit
            self.dataframes["layer"] = self.find_pore(
                self.MAX_DIST,
                self.dataframes["bulk"],
                self.surf_geometry
            )
            return True

    def _perform_grafting(
        self,
        tilt_angle: float,
        surf_norm: str,
        holes: bool
    ) -> None:
        """Perform the grafting algorithm."""
        grafting_results = self._graft_surface(
            grafting_density=self.grafting_density,
            surf_dist=self.surf_dist,
            surface_df=self.dataframes["layer"],
            geometry=self.surf_geometry,
            surf_norm=surf_norm,
            tilt_angle=tilt_angle,
            dist_params=self.distribution_params,
            chain_size=self.chain_size,
            fractions=self.fractions,
            holes=holes
        )
        
        (self.dataframes["polymer"],
        self.dataframes["under_polymer"],
        self.nSils,
        self.molecule_sizes) = grafting_results

    def _update_system_state(self) -> None:
        """Update names, merge dataframes, and recreate universe."""
        # Clean up dataframes
        self.dataframes["bulk"] = self.dataframes["bulk"].drop(
            labels=self.dataframes["layer"].index,
            axis=0,
            inplace=False
        )
        self.dataframes["layer"] = self.dataframes["layer"].drop(
            labels=self.dataframes["under_polymer"].index,
            axis=0,
            inplace=False
        )

        # Update naming
        name_mapping = self.name_mapping.copy()
        name_mapping["under_polymer"][0] = name_mapping["under_polymer"][0][0]
        self.update_names(list(self.dataframes.keys()), name_mapping)

        # Merge and create universe
        self.merge_dataframes()
        self.universe = self.create_universe_from_df(
            self.dataframes["unified"],
            box=self.dimensions,
            convert_unit=1
        )

    def merge_dataframes(self) -> None:
        """Merges all component dataframes into a unified representation.
        
        Combines polymer, under_polymer, layer, and bulk dataframes with proper:
        - Type and bead labeling
        - Molecular size annotations
        - Index management
        
        Updates:
            - self.dataframes["unified"] with merged dataframe
            - System dimensions via update_dimensions()
        """

        # Debug: Verify all required dataframes exist
        required_dfs = ["polymer", "under_polymer", "layer", "bulk"]
        for df_name in required_dfs:
            if df_name not in self.dataframes:
                raise ValueError(f"Missing required dataframe: {df_name}")
            if self.dataframes[df_name].empty:
                raise ValueError(f"Dataframe {df_name} is empty")
        
        # Debug: Check molecule sizes
        if not hasattr(self, 'molecule_sizes') or not self.molecule_sizes:
            raise ValueError("No molecule sizes defined")

        # Get under_polymer dataframe
        under_polymer_df = self.dataframes["under_polymer"]

        # Initialize empty dataframe with correct columns
        columns = ["x", "y", "z", "bead", "type"]
        df_mol = pd.DataFrame(columns=columns)
        
        if self.nSils > 0:
            # Pre-calculate name components for efficiency
            polymer_type_base = self.name_mapping["polymer"][0][0]
            polymer_bead = self.name_mapping["polymer"][1]
            
            # Process each silane molecule
            start_idx = 0
            under_polymer_idx = 0
            polymer_start = 0
            
            for mol_size in self.molecule_sizes:
                # Calculate slice indices
                polymer_end = polymer_start + mol_size
                mol_end = start_idx + mol_size
                
                # Add polymer segment
                polymer_slice = self.dataframes["polymer"].iloc[polymer_start:polymer_end]
                df_mol = pd.concat([df_mol, polymer_slice], axis=0, ignore_index=True)
                
                # Label types and beads
                type_label = f"{polymer_type_base}{mol_size}"
                
                # First atom gets "END" bead type
                df_mol.iloc[start_idx, df_mol.columns.get_loc('type')] = type_label
                df_mol.iloc[start_idx, df_mol.columns.get_loc('bead')] = "END"
                
                # Remaining atoms get standard bead type
                df_mol.iloc[start_idx+1:mol_end, df_mol.columns.get_loc('type')] = type_label
                df_mol.iloc[start_idx+1:mol_end, df_mol.columns.get_loc('bead')] = polymer_bead
                
                # Add corresponding under_polymer atom
                if under_polymer_idx >= len(under_polymer_df):
                    raise ValueError(
                        f"Invalid under_polymer index {under_polymer_idx} "
                        f"(max {len(under_polymer_df)-1}). "
                        f"polymer_start={polymer_start}, mol_size={mol_size}"
                    )
        
                under_polymer_atom = under_polymer_df.iloc[[under_polymer_idx]].copy()                
                df_mol = pd.concat([df_mol, under_polymer_atom], axis=0, ignore_index=True)
                df_mol.iloc[-1, df_mol.columns.get_loc('type')] += str(mol_size)
                
                # Update indices
                under_polymer_idx += 1
                polymer_start = polymer_end
                start_idx = mol_end + 1  # +1 for the under_polymer atom
        
        # Merge all components
        components = [
            df_mol,
            self.dataframes["layer"],
            self.dataframes["bulk"]
        ]
        df_unified = pd.concat(components, axis=0, ignore_index=True)
        
        # Update system state
        self.dataframes["unified"] = df_unified
        self.update_dimensions(df_unified)

    def create_itp_files(
        self,
        molecules: List[Dict[str, object]],
        out_name: str = "topol.top"
    ) -> str:
        """Generates GROMACS .itp molecular topology files from molecule specifications.
        
        Args:
            molecules: List of molecule definitions, each containing:
                - name: Molecule name (e.g., "SIL")
                - atoms: List of atom dictionaries with:
                    - atom: Atom name
                    - res: Residue name
                    - type: Atom type
                    - charge: Partial charge
                    - mass: Atomic mass
                - bonds: Optional list of bond definitions
                - angles: Optional list of angle definitions
            out_name: Output filename (default: "topol.top")
        
        Returns:
            Generated file content as string
        
        Raises:
            ValueError: If required fields are missing
            IOError: If file cannot be written
        """
        # Validate input molecules
        self._validate_molecules(molecules)
        
        # Generate file sections
        sections = []
        for mol in molecules:
            sections.append(self._create_molecule_section(mol))
            sections.append(self._create_atoms_section(mol))
            
            if "bonds" in mol:
                sections.append(self._create_bonds_section(mol["bonds"]))
                
            if "angles" in mol:
                sections.append(self._create_angles_section(mol["angles"]))
        
        # Combine sections and write file
        file_content = "\n".join(sections)
        self._write_itp_file(file_content, out_name)
        
        return file_content

    def _validate_molecules(self, molecules: List[Dict]) -> None:
        """Validate molecule definitions before processing."""
        required_atom_fields = {"atom", "res", "type", "charge", "mass"}
        
        for i, mol in enumerate(molecules):
            if "name" not in mol:
                raise ValueError(f"Molecule at index {i} missing 'name' field")
                
            if "atoms" not in mol:
                raise ValueError(f"Molecule {mol.get('name')} missing 'atoms' section")
                
            for j, atom in enumerate(mol["atoms"]):
                missing = required_atom_fields - set(atom.keys())
                if missing:
                    raise ValueError(
                        f"Atom {j} in molecule {mol['name']} missing fields: {missing}"
                    )

    def _create_molecule_section(self, mol: Dict) -> str:
        """Generate [ moleculetype ] section."""
        return (
            "[ moleculetype ]\n"
            "; molname\tnrexcl\n"
            f"  {mol['name']:<16}1\n"
        )

    def _create_atoms_section(self, mol: Dict) -> str:
        """Generate [ atoms ] section."""
        lines = ["[ atoms ]", "; id\ttype\tresnr\tresidu\tatom\tcgnr\tcharge\tmass"]
        
        for j, atom in enumerate(mol["atoms"], 1):
            lines.append(
                f"{j:6d} {atom['type']:>6} {j:6d} {atom['res']:>6} "
                f"{atom['atom']:>6} {j:6d} {atom['charge']:8.3f} {atom['mass']:8.3f}"
            )
        
        return "\n".join(lines)

    def _create_bonds_section(self, bonds: List[Dict]) -> str:
        """Generate [ bonds ] section."""
        lines = ["[ bonds ]", "; id1    id2   funct   b0    Kb"]
        
        for bond in bonds:
            lines.append(
                f"{bond['id1']:6d} {bond['id2']:6d} {bond['funct']:6d} "
                f"{bond.get('b0', 0.470):8.3f} {bond.get('Kb', 3800.000):8.3f}"
            )
        
        return "\n".join(lines)

    def _create_angles_section(self, angles: List[Dict]) -> str:
        """Generate [ angles ] section."""
        lines = ["[ angles ]", "; i    j    k    funct   angle   force.c."]
        
        for angle in angles:
            lines.append(
                f"{angle['i']:6d} {angle['j']:6d} {angle['k']:6d} "
                f"{angle['funct']:6d} {angle.get('angle', 180.000):8.3f} "
                f"{angle.get('force.c.', 35.000):8.3f}"
            )
        
        return "\n".join(lines)

    def _write_itp_file(self, content: str, filename: str) -> None:
        """Write content to output file."""
        output_path = Path(self.root_dir) / filename
        
        try:
            with output_path.open("w") as f:
                f.write(content)
            print(f"Successfully wrote topology file: {output_path}")
        except IOError as e:
            print(f"Error writing topology file: {e}")
            raise

    def generate_itp_files_PDMS(self, rep_range, dict_names=None):
        """
        Generates the itp files.

        Parameters:
        - rep_range: The range of repetitions.
        """

        if dict_names is None:
            if not self.name_mapping:
                raise Exception("No dictionary of names given, please provide one")
        else:
            self.name_mapping = dict_names
             

        def generate_itp(nrep):
            out_file = open(f"{self.root_dir}/itps/lay_pdms_{nrep}.itp","w")

            #molecule type
            out_file.write(f"[ moleculetype ]\n; molname  nrexcl\n  {self.name_mapping['polymer'][0]}{nrep}       1\n\n[ atoms ]\n")
            for i in range(1,nrep+1):
                if i == 1 :
                    out_file.write(f"{i} {self.name_mapping['polymer'][1]}   {i} {self.name_mapping['polymer'][0][0]}{nrep}  END   {i} 0.0 72.0\n")
                else:
                    out_file.write(f"{i} {self.name_mapping['polymer'][1]}   {i} {self.name_mapping['polymer'][0][0]}{nrep}  {self.name_mapping['polymer'][1]}   {i} 0.0 72.0\n")

            tip = nrep+1
            out_file.write(f"{tip} {self.name_mapping['under_polymer'][1]} {tip} {self.name_mapping['under_polymer'][0][0]}{nrep} {self.name_mapping['under_polymer'][1]} {tip} 0.0 72.0\n")

            #bonds
            out_file.write("\n[ bonds ]\n")
            for i in range(2,nrep):
                j = i+1 
                out_file.write(f"  {i}   {j} 1 0.448 11500\n")
            out_file.write("\n; END-PDMS\n")
            out_file.write(" 1   2 1 0.446 11000\n")
            out_file.write("\n; PDMS-SURF\n")
            out_file.write(f" {nrep}   {tip} 1 0.470 3800\n")

            #angles
            if nrep > 2:
                out_file.write("\n[ angles ]\n")
                for i in range(2,nrep-2):
                    j = i+1
                    k = i+2
                    out_file.write(f"  {i}   {j}   {k} 1 86 45.89\n")
                
            
                out_file.write("\n; END-PDMS\n")
                out_file.write("  1   2   3 1 87 78\n")
                out_file.write("\n; PDMS-SURF\n")
                pre_tip = nrep-1 
                out_file.write(f" {pre_tip}   {nrep}   {tip} 1 180 35\n")

            #dihedrals
            if nrep > 3:
                out_file.write("\n[ dihedrals ]\n")
                for i in range(2,nrep-3):
                    j = i+1
                    k = i+2
                    l = i+3
                    out_file.write(f"  {i}   {j}   {k}   {l}  1 1.18 1.4 1\n")
                out_file.write("\n; END-PDMS\n")
                out_file.write("  1   2   3   4 1 1.85 1.2 1\n")
                out_file.write("\n; PDMS-SURF\n")
                #out_file.write(f"  {i+1}   {j+1}   {k+1}   {l+1}  1 1.18 1.4 1")
            out_file.close()

        subprocess.run(f"mkdir -p {self.root_dir}/itps;", shell=True, executable="/bin/bash")

        print("Generating itp files")
        # for nrep in tqdm(rep_range):
        for nrep in rep_range:
            generate_itp(nrep)

    def out_gro(self, fname: str = "initial_config.gro", z_padding: float = 200.0) -> None:
        """Writes the current system configuration to a GROMACS .gro file.

        Args:
            fname: Output filename (default: "initial_config.gro")
        
        Raises:
            ValueError: If universe or atom positions are invalid
            IOError: If file cannot be written
        """
        # Validate system state
        if not hasattr(self, 'universe') or not self.universe:
            raise ValueError("No universe available to write")
        
        if not len(self.universe.atoms):
            raise ValueError("No atoms in universe to write")

        # Prepare output path
        output_path = Path(self.root_dir) / fname
        
        try:
            # Ensure positions are numpy array
            self.universe.atoms.positions = np.asarray(self.universe.atoms.positions)
            
            # Adjust box dimensions (add 200 to z-axis)
            new_dimensions = self.universe.dimensions.copy()
            new_dimensions[2] += z_padding
            self.universe.dimensions = new_dimensions
            
            # Write GRO file using context manager
            with mda.coordinates.GRO.GROWriter(str(output_path)) as gro_writer:
                gro_writer.write(self.universe)
                
            print(f"Successfully wrote GRO file: {output_path}")
            
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid universe state: {str(e)}") from e
        except IOError as e:
            raise IOError(f"Failed to write GRO file: {str(e)}") from e

    def out_topology(
        self,
        fname: str = "initial_config.gro",
        includes: List[str] = None,
        out_name: str = "topol.top",
        topol: str = "created_by_silanizer",
        out_sizes: bool = False
    ) -> None:
        """Generates complete topology output including .gro file and topology files.

        Args:
            fname: Output .gro filename (default: "initial_config.gro")
            includes: List of include statements for topology file
            out_name: Output topology filename (default: "topol.top")
            topol: System description for topology header
            out_sizes: Whether to output molecular sizes file

        Raises:
            ValueError: If required data is missing
            IOError: If files cannot be written
        """
        # Initialize includes if not provided
        if includes is None:
            includes = []

        # Write coordinate file
        self.out_gro(fname)

        # Write molecular sizes if requested
        if out_sizes:
            self._write_molecular_sizes()

        # Generate topology based on dispersity type
        if self.dispersity[0] == "poly":
            self._write_polydisperse_topology(includes, out_name, topol)
        else:
            self._write_monodisperse_topology(includes, out_name, topol)

    def _write_molecular_sizes(self) -> None:
        """Write molecular sizes to file."""
        output_path = Path(self.root_dir) / "molecule_sizes.dat"
        try:
            with output_path.open("w") as f:
                f.writelines(f"{size}\n" for size in self.molecule_sizes)
        except IOError as e:
            raise IOError(f"Failed to write molecular sizes: {e}") from e

    def _write_polydisperse_topology(
        self,
        includes: List[str],
        out_name: str,
        topol: str
    ) -> None:
        """Generate topology for polydisperse system."""
        # Count polymer types
        poly_types = [f"{self.name_mapping['polymer'][0]}{n}" for n in self.molecule_sizes]
        poly_counts = Counter(poly_types)
        
        # Prepare molecule counts
        poly_mols = [[p, 1] for p in poly_types]
        surf_mols = [
            [self.name_mapping["layer"][0], len(self.dataframes["layer"])],
            [self.name_mapping["bulk"][0], len(self.dataframes["bulk"])]
        ]
        
        # Write topology file
        output_path = Path(self.root_dir) / out_name
        self.write_topol(
            str(output_path),
            topol,
            poly_mols + surf_mols,
            includes,
            "w"
        )

    def _write_monodisperse_topology(
        self,
        includes: List[str],
        out_name: str,
        topol: str
    ) -> None:
        """Generate topology for monodisperse system."""
        natoms = len(self.molecule_sizes)
        size = self.molecule_sizes[0]
        
        surf_mols = [
            [f"{self.name_mapping['polymer'][0]}{size}", natoms],
            [self.name_mapping["layer"][0], len(self.dataframes["layer"])],
            [self.name_mapping["bulk"][0], len(self.dataframes["bulk"])]
        ]
        
        output_path = Path(self.root_dir) / out_name
        self.write_topol(
            str(output_path),
            topol,
            surf_mols,
            includes,
            "w"
        )

    def cylinder_fitting(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        p: np.ndarray,
        axis: Literal["vertical", "tilted"],
        plot: Literal["yes", "no"] = "no"
    ) -> Tuple[np.ndarray, float]:
        """Fits a cylindrical equation to 3D point cloud data.

        Args:
            x: X-coordinates of points
            y: Y-coordinates of points
            z: Z-coordinates of points
            p: Initial parameters:
                [0] Xc - x coordinate of cylinder center
                [1] Yc - y coordinate of cylinder center
                [2] alpha - rotation angle about x-axis (radians)
                [3] beta - rotation angle about y-axis (radians)
                [4] r - radius of cylinder
            axis: Orientation constraint:
                "vertical" - fixed along z-axis
                "tilted" - allowed to tilt
            plot: Whether to generate visualization plots ("yes" or "no")

        Returns:
            Tuple containing:
            - Estimated parameters (same format as input p)
            - Computed surface area of the fitted cylinder (in nm²)

        Raises:
            ValueError: If input arrays have inconsistent lengths
        """
        # Validate input shapes
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z arrays must have same length")

        # Fit cylinder based on axis constraint
        if axis == "vertical":
            est_p = self._fit_vertical_cylinder(x, y, z, p)
        else:
            est_p = self._fit_tilted_cylinder(x, y, z, p)

        # Generate plots if requested
        if plot == "yes":
            self._plot_cylinder_fit(x, y, z, est_p)

        # Compute and return results
        surface_area = self._compute_cylinder_area(est_p, z)
        return est_p, surface_area

    def _fit_vertical_cylinder(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        p: np.ndarray
    ) -> np.ndarray:
        """Fit a z-axis aligned cylinder."""
        # Simplified parameters for vertical case
        reduced_p = np.array([p[0], p[1], p[4]])

        def fitfunc(p, x, y, z):
            """Equation for vertical cylinder."""
            return (-np.cos(0.0)*(p[0] - x) - z*np.cos(0.0)*np.sin(0.0) - 
                    np.sin(0.0)*np.sin(0.0)*(p[1] - y))**2 + (
                    z*np.sin(0.0) - np.cos(0.0)*(p[1] - y))**2

        def errfunc(p, x, y, z):
            """Error function for vertical fit."""
            return fitfunc(p, x, y, z) - p[2]**2

        return leastsq(errfunc, reduced_p, args=(x, y, z), maxfev=1000)[0]

    def _fit_tilted_cylinder(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        p: np.ndarray
    ) -> np.ndarray:
        """Fit a cylinder with arbitrary orientation."""
        def fitfunc(p, x, y, z):
            """Equation for general cylinder."""
            return (-np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - 
                    np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (
                    z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2

        def errfunc(p, x, y, z):
            """Error function for general fit."""
            return fitfunc(p, x, y, z) - p[4]**2

        return leastsq(errfunc, p, args=(x, y, z), maxfev=1000)[0]

    def _plot_cylinder_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        est_p: np.ndarray
    ) -> None:
        """Generate 3D visualization of cylinder fit."""
        fig = plt.figure(figsize=(12, 6))
        
        # Perspective view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(x, y, z, alpha=0.3, c="black")
        
        # Top view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x, y, z, alpha=0.3, c="black")
        ax2.view_init(90, 90)
        
        # Generate cylinder surface
        z_range = np.linspace(np.min(z), np.max(z), 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z_range)
        
        x_grid = est_p[4]*np.cos(theta_grid) + est_p[0]
        y_grid = est_p[4]*np.sin(theta_grid) + est_p[1]
        
        for ax in (ax1, ax2):
            ax.plot_surface(x_grid, y_grid, z_grid, 
                        cmap=plt.cm.YlGnBu_r, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("cylinder_fit_combined.png", dpi=300)
        plt.close()

    def _compute_cylinder_area(
        self,
        est_p: np.ndarray,
        z: np.ndarray
    ) -> float:
        """Calculate surface area of fitted cylinder."""
        radius_nm = est_p[4] * 0.1  # Convert Å to nm
        height_nm = (np.max(z) - np.min(z)) * 0.1  # Convert Å to nm
        return 2 * np.pi * radius_nm * (radius_nm + height_nm)

    def _graft_surface(
        self,
        grafting_density: float,
        surf_dist: float,
        surface_df: pd.DataFrame,
        geometry: Literal["flat", "cylindrical", "slit"],
        surf_norm: str,
        tilt_angle: float = 0,
        dist_params: Optional[Tuple[float, float]] = None,
        chain_size: Optional[Union[int, List[int]]] = None,
        fractions: Optional[List[float]] = None,
        holes: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[int]]:
        """Grafts polymer molecules onto a surface with controlled density and orientation.

        Args:
            grafting_density: Grafting density (molecules per unit area)
            surf_dist: Distance between grafted molecules and surface
            surface_df: DataFrame containing surface coordinates and normals
            geometry: Surface geometry ("flat", "cylindrical", or "slit")
            surf_norm: Surface normal direction ("x", "y", or "z")
            tilt_angle: Maximum tilt angle for molecules (degrees)
            dist_params: Parameters for polydisperse grafting (a, scale)
            chain_size: Monodisperse chain length or list of lengths
            fractions: Fractions for each chain size (if multiple sizes)
            holes: Whether surface has holes/defects

        Returns:
            Tuple containing:
            - df_Sil: DataFrame of grafted molecule coordinates
            - df_lay1: DataFrame of surface attachment points
            - numOfMols: Number of grafted molecules
            - molecule_sizes: List of molecular sizes

        Raises:
            ValueError: For invalid grafting density or missing required parameters
        """
        # Validate input parameters
        self._validate_grafting_inputs(
            grafting_density, chain_size, fractions, dist_params
        )

        # Calculate grafting area and maximum density
        areaXY, maxgrafting_density = self._calculate_grafting_area(surface_df, geometry, surf_norm)
        numOfMols = round(grafting_density * areaXY)

        if grafting_density > maxgrafting_density:
            raise ValueError(f"Maximum grafting density is {maxgrafting_density:.3f}!")

        # Select random grafting sites
        grafting_sites = self._select_grafting_sites(surface_df, numOfMols)

        # Graft molecules based on dispersity type
        if dist_params:
            return self._graft_polydisperse(
                grafting_sites, surface_df, surf_dist, surf_norm, 
                tilt_angle, holes, dist_params, numOfMols
            )
        elif chain_size:
            return self._graft_monodisperse(
                grafting_sites, surface_df, surf_dist, surf_norm,
                tilt_angle, holes, chain_size, fractions, numOfMols
            )

    def _validate_grafting_inputs(
        self,
        grafting_density: float,
        chain_size: Optional[Union[int, List[int]]],
        fractions: Optional[List[float]],
        dist_params: Optional[Tuple[float, float]]
    ) -> None:
        """Validate grafting input parameters."""
        if grafting_density <= 0:
            raise ValueError("Grafting density must be positive")
        
        if dist_params is None and chain_size is None:
            raise ValueError("Must specify either dist_params or chain_size")
        
        if isinstance(chain_size, list) and fractions is None:
            raise ValueError("Must specify fractions for multiple chain sizes")

    def _calculate_grafting_area(
        self,
        df_cut: pd.DataFrame,
        geometry: str,
        surf_norm: str
    ) -> Tuple[float, float]:
        """Calculate available grafting area and maximum density."""
        if geometry == "flat":
            coords = ["x", "y", "z"]
            axis = [i for i in coords if i != surf_norm]
            xi, yi = df_cut[axis[0]], df_cut[axis[1]]
            
            sideX, sideY = np.max(xi) - np.min(xi), np.max(yi) - np.min(yi)
            areaXY = sideX * sideY
        elif geometry == "cylindrical":
            p = np.array([np.mean(df_cut["x"]), np.mean(df_cut["y"]), 0, 0, 10])
            _, areaXY = self.cylinder_fitting(
                df_cut["x"], df_cut["y"], df_cut["z"], p, "vertical", plot="yes"
            )
        elif geometry == "slit":
            areaXY = (df_cut["z"].max() - df_cut["z"].min()) * \
                    (df_cut["x"].max() - df_cut["x"].min()) * 2
        
        maxgrafting_density = float(len(df_cut) / areaXY)
        return areaXY, maxgrafting_density

    def _select_grafting_sites(
        self,
        df_cut: pd.DataFrame,
        numOfMols: int
    ) -> np.ndarray:
        """Randomly select surface sites for grafting."""
        return np.random.choice(df_cut.index, numOfMols, replace=False)

    def _graft_molecule(
        self,
        df_calc: pd.Series,
        last_pos: List[float],
        molSize: int,
        surf_dist: float,
        surf_norm: str,
        tilt_angle: float,
        holes: bool
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate coordinates for a single grafted molecule."""
        xS, yS, zS = [], [], []
        for j in range(molSize, 0, -1):
            if holes:
                xp = df_calc["x"] + surf_dist * df_calc["normx"] * j
                yp = df_calc["y"] + surf_dist * df_calc["normy"] * j
                zp = df_calc["z"]
            else:
                coords = ["x", "y", "z"]
                axis = [i for i in coords if i != surf_norm]
                xp = df_calc[axis[0]] + last_pos[0] * random.uniform(0, tilt_angle)
                yp = df_calc[axis[1]] + last_pos[1] * random.uniform(0, tilt_angle)
                zp = df_calc[surf_norm] + j * surf_dist + last_pos[2] * random.uniform(0, tilt_angle)
            
            last_pos = [xp, yp, zp]
            xS.append(xp)
            yS.append(yp)
            zS.append(zp)
        
        return last_pos, xS, yS, zS

    def _graft_polydisperse(
        self,
        grafting_sites: np.ndarray,
        df_cut: pd.DataFrame,
        surf_dist: float,
        surf_norm: str,
        tilt_angle: float,
        holes: bool,
        dist_params: Tuple[float, float],
        numOfMols: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[int]]:
        """Graft molecules with polydisperse chain lengths."""
        xS, yS, zS = [], [], []
        molecule_sizes = []
        last_pos = [0, 0, 0]
        
        a = 1 / (dist_params[0] - 1)
        b = a
        
        for i, k in enumerate(grafting_sites, 1):
            molSize = 0
            while molSize == 0:
                molSize = int(gamma.rvs(a, scale=dist_params[1]/b, size=1)[0])
            
            df_calc = df_cut.loc[k]
            last_pos, xS_new, yS_new, zS_new = self._graft_molecule(
                df_calc, last_pos, molSize, surf_dist, surf_norm, tilt_angle, holes
            )
            
            xS.extend(xS_new)
            yS.extend(yS_new)
            zS.extend(zS_new)
            molecule_sizes.append(molSize)
            
            self._update_progress(i, numOfMols)
        
        return self._assemble_results(
            xS, yS, zS, df_cut, grafting_sites, numOfMols, molecule_sizes, holes, surf_norm
        )

    def _graft_monodisperse(
        self,
        grafting_sites: np.ndarray,
        df_cut: pd.DataFrame,
        surf_dist: float,
        surf_norm: str,
        tilt_angle: float,
        holes: bool,
        chain_size: Union[int, List[int]],
        fractions: Optional[List[float]],
        numOfMols: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[int]]:
        """Graft molecules with monodisperse or predefined chain lengths."""
        xS, yS, zS = [], [], []
        molecule_sizes = []
        last_pos = [0, 0, 0]
        
        if isinstance(chain_size, int):
            # Single chain size
            molSize = chain_size
            for i, k in enumerate(grafting_sites, 1):
                df_calc = df_cut.loc[k]
                last_pos, xS_new, yS_new, zS_new = self._graft_molecule(
                    df_calc, last_pos, molSize, surf_dist, surf_norm, tilt_angle, holes
                )
                
                xS.extend(xS_new)
                yS.extend(yS_new)
                zS.extend(zS_new)
                molecule_sizes.append(molSize)
                
                self._update_progress(i, numOfMols)
        else:
            # Multiple chain sizes
            split_sizes = [int(f * numOfMols) for f in fractions]
            split_sizes[-1] = numOfMols - sum(split_sizes[:-1])
            idx_splits = np.split(grafting_sites, np.cumsum(split_sizes)[:-1])
            
            counter = 1
            for cs, idx_subset in zip(chain_size, idx_splits):
                molSize = cs
                for i, k in enumerate(idx_subset, 1):
                    df_calc = df_cut.loc[k]
                    last_pos, xS_new, yS_new, zS_new = self._graft_molecule(
                        df_calc, last_pos, molSize, surf_dist, surf_norm, tilt_angle, holes
                    )
                    
                    xS.extend(xS_new)
                    yS.extend(yS_new)
                    zS.extend(zS_new)
                    molecule_sizes.append(molSize)
                    
                    self._update_progress(counter, numOfMols)
                    counter += 1

        return self._assemble_results(
            xS, yS, zS, df_cut, grafting_sites, numOfMols, molecule_sizes, holes, surf_norm
        )

    def _update_progress(self, current: int, total: int) -> None:
        """Update progress display."""
        sys.stdout.write(f"Adding molecules: {current/total*100:.2f}%\r")
        sys.stdout.flush()

    def _assemble_results(
        self,
        xS: List[float],
        yS: List[float],
        zS: List[float],
        df_cut: pd.DataFrame,
        grafting_sites: np.ndarray,
        numOfMols: int,
        molecule_sizes: List[int],
        holes: bool,
        surf_norm: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[int]]:
        """Assemble final results from grafting operation."""
        print(f"\nNumber of molecules: {numOfMols} - "
            f"Number of spots: {len(df_cut)} - "
            f"Max. grafting dens.: {len(df_cut)/numOfMols:.3f}\n")
        
        if holes:
            df_Sil = pd.DataFrame({"x": xS, "y": yS, "z": zS})
        else:
            coords = ["x", "y", "z"]
            axis = {surf_norm: zS}
            axis.update({c: s for c, s in zip(
                [i for i in coords if i != surf_norm], [xS, yS]
            )})
            df_Sil = pd.DataFrame(axis)
        
        df_lay1 = df_cut.loc[grafting_sites]
        return df_Sil, df_lay1, numOfMols, molecule_sizes


    def find_pore(
        MAX_DIST: float,
        cyl: pd.DataFrame,
        geometry: Literal["slit", "flat", "cylindrical"]
    ) -> pd.DataFrame:
        """Identifies pore coordinates in a cylindrical or slit geometry.

        Args:
            MAX_DIST: Maximum distance for pore boundary detection
            cyl: DataFrame containing material coordinates
            geometry: Pore geometry type ("slit", "flat", or "cylindrical")

        Returns:
            DataFrame containing pore coordinates with normals
        """
        def centroid(df: pd.DataFrame) -> Tuple[float, float]:
            """Calculate 2D centroid of coordinates."""
            return np.mean(df["x"]), np.mean(df["y"])

        def find_normals(
            df: pd.DataFrame,
            geo: str,
            ctd: Tuple[float, float]
        ) -> pd.DataFrame:
            """Calculate normal vectors for surface points."""
            if geo in ("slit", "flat"):
                # Calculate plane normal from first 3 points
                Ps = df[["x", "y", "z"]].iloc[:3].values
                normal = np.cross(Ps[:,1] - Ps[:,0], Ps[:,2] - Ps[:,0])
                df['normx'], df['normy'], _ = normal / np.linalg.norm(normal)
            elif geo == "cylindrical":
                # Calculate radial normals for cylinder
                dx = df['x'] - ctd[0]
                dy = df['y'] - ctd[1]
                dist = np.sqrt(dx**2 + dy**2)
                df['normx'] = dx / dist
                df['normy'] = dy / dist
                
            return df

        def find_hole(
            df: pd.DataFrame,
            center: Tuple[float, float],
            delta: float
        ) -> pd.DataFrame:
            """Identify boundary atoms of the pore."""
            df['distance'] = np.sqrt((df['x'] - center[0])**2 + (df['y'] - center[1])**2)
            dz = df["z"].max() - df["z"].min()
            R_max = df['distance'].min()
            
            return df[
                (df['distance'] >= R_max - delta) & 
                (df['distance'] <= R_max + delta) &
                (df['z'] <= df["z"].max() - dz * 0.05) &
                (df['z'] >= df["z"].min() + dz * 0.05)
            ]

        center = centroid(cyl)
        pore_df = find_hole(cyl, center, MAX_DIST)
        return find_normals(
            pore_df.sort_values(by=["z", "x", "y"]),
            geometry,
            center
        )

    def plot_layers(
        n: int,
        df_Sil: pd.DataFrame,
        df_layer: pd.DataFrame,
        df_layer1: pd.DataFrame,
        df_Si_layer: pd.DataFrame
    ) -> None:
        """Visualize molecular layers with normals.

        Args:
            n: Number of layers to plot
            df_Sil: Grafted molecules coordinates
            df_layer: Surface layer coordinates
            df_layer1: Attachment points coordinates
            df_Si_layer: Silicon layer coordinates
        """
        # 3D plot of all components
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_Sil["x"], df_Sil["y"], df_Sil["z"], label='Grafted molecules')
        ax.scatter(df_Si_layer["x"], df_Si_layer["y"], df_Si_layer["z"], label='Si layer')
        ax.scatter(df_layer1["x"], df_layer1["y"], df_layer1["z"], label='Attachment points')
        ax.legend()
        plt.savefig("cylinder_silane.png", dpi=300)
        plt.close()

        # 2D plots per layer
        z_layers = sorted(set(df_Sil["z"]))[:n]
        for z in z_layers:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            
            # Filter data for current layer
            layer_attach = df_layer1[df_layer1["z"] == z]
            layer_grafted = df_Sil[df_Sil["z"] == z]
            layer_surface = df_layer[df_layer["z"] == z]

            # Plot points
            ax.scatter(layer_grafted["x"], layer_grafted["y"], c="red", label='Grafted')
            ax.scatter(layer_attach["x"], layer_attach["y"], c="blue", label='Attachment')
            ax.scatter(layer_surface["x"], layer_surface["y"], c="green", label='Surface')

            # Plot normals
            ax.quiver(
                layer_attach["x"], layer_attach["y"],
                layer_attach["normx"], layer_attach["normy"],
                angles='xy', scale_units='xy', scale=1
            )
            
            ax.legend()
            plt.savefig(f"normals_{z}.png", dpi=300)
            plt.close()

    def load_inputs(
        path: str
    ) -> Tuple[str, float, float, List, List, str, Dict, float, str]:
        """Load and validate simulation parameters from JSON input file.

        Args:
            path: Path to JSON input file

        Returns:
            Tuple containing:
            - folder: Output directory path
            - surf_dist: Surface distance (Å)
            - grafting_density: Grafting density (chains/nm²)
            - matrix: Matrix construction parameters
            - dispersity: Chain dispersity parameters
            - surf_geometry: Surface geometry type
            - name_mapping: Atom name mappings
            - tilt_angle: Molecular tilt angle (degrees)
            - sample: Sample name
        """
        with open(path) as f:
            inputs = json.load(f)

        # Extract and validate parameters
        sample = inputs["name"]
        root_dir = Path(inputs["root_dir"])
        surf_dist = inputs["surface_distance"] * 10  # Convert nm to Å
        
        # Matrix configuration
        matrix_file = inputs["matrix"]["file"]
        matrix = (
            ["file", matrix_file] if matrix_file
            else ["build", inputs["matrix"]["size"]]
        )

        # Dispersity configuration
        mono = inputs["chain dispersity"]["monodisperse"]
        dispersity = (
            ["mono", mono] if mono
            else ["poly", inputs["chain dispersity"]["polydisperse"]]
        )

        # Surface geometry
        surf_geometry = (
            "cylindrical" if inputs["surface geometry"]["cylindrical"]
            else "flat"
        )

        name_mapping = inputs["atom names"]
        tilt_angle = inputs["perturbation"]

        # Create output directory
        root_dir.mkdir(parents=True, exist_ok=True)

        # Print configuration summary
        config_summary = f"""
        Configuration Summary:
        - Sample: {sample}
        - Output folder: {root_dir}
        - Surface distance: {surf_dist * 0.1} nm
        - Grafting density: {inputs["grafting_density"]} chains/nm²
        - Matrix: {'file: ' + matrix_file if matrix_file else 'build: ' + str(inputs["matrix"]["size"])}
        - Dispersity: {dispersity[0] + (': ' + str(dispersity[1]) if dispersity[1] else '')}
        - Surface geometry: {surf_geometry}
        - Tilt angle: {tilt_angle}°
        """

        print(config_summary)

        return (
            str(root_dir), surf_dist, inputs["grafting_density"], matrix,
            dispersity, surf_geometry, name_mapping, tilt_angle, sample
        )