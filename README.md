# PhD_Programs
This repository contains programs developed during my PhD for setting up and analyzing molecular systems for grafting.

## Overview

The Grafting.py script defines a NewSystem class, which is designed to construct and analyze molecular systems with grafted structures. It provides an interface to build surfaces, apply grafting techniques, and calculate properties such as mean square displacement, layer height, and density profiles.

The notebook demonstrates example workflows using Grafting.py for constructing different system geometries, creating input files, and building and visualizing the system.
Required packages:
- numpy
- pandas
- matplotlib
- scipy
- MDAnalysis
- tqdm
- natsort
- Grafter (a custom library for molecular system setup and analysis)

## Contents

    1. Grafting.py: Core module for building and analyzing grafting systems.
    Notebook_create_system_pore.ipynb: Example notebook demonstrating system setup and grafting workflows.
    2. Notebook: Notebook_create_system_pore.ipynb

The notebook provides a guided example of using Grafting.py to create a molecular system. Key steps include:
- Setting up a working directory and visualization parameters.
- Constructing geometries:
  - Building a block/slab structure.
  - Creating a cylindrical pore in a block matrix.
- Configuring system parameters in an input file (input_grafter.json) for reproducibility.
- Executing grafting and visualization of the final configuration.

## Running the Notebook

- Open Notebook_create_system_pore.ipynb in Jupyter.
- Execute each cell sequentially to:
  - Build structures.
    - Visualize geometries.
    - Save and visualize the final configuration.

## Example Workflow

    Define System Parameters: Modify parameters such as surface distance, grafting density, and molecule types.
    Run Grafting Procedures: Use NewSystem.graft_matrix() to apply grafting based on the defined parameters.
    Visualize and Analyze: Plot results for validation and analysis.

## Output

- .gro files representing molecular structures.
- JSON files storing input parameters for reproducibility.
- Visualization of system components to validate configuration.
