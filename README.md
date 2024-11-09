# PhD_Programs
This repository contains programs developed during my PhD for setting up and analyzing molecular systems for grafting.

## Overview

The Grafter program is designed to construct and analyze molecular systems with grafted structures. It provides an interface to build surfaces, apply grafting techniques, and calculate properties such as mean square displacement, layer height, and density profiles.

The notebooks "Notebook_creat_system_PDMS" and "Notebook_creat_system_pore" demonstrate example workflows using the Grafter program for constructing different system geometries, creating input files, and building and visualizing the system.

The "Analysis" notebooks contain the analysis of a polydisperse PDMS-grafted surface, using the Grafter analysis modules. 

Required packages:
- numpy
- pandas
- matplotlib
- scipy
- MDAnalysis
- tqdm
- natsort

## Contents

    1. Grafting.py: Core module for building and analyzing grafting systems.
    2. Notebook: Notebook_create_system_pore.ipynb
    3. Notebook: Notebook_create_system_PDMS.ipynb
    4. Notebook: Analysis_CAH.ipynb
    5. Notebook: Analysis_marching_cubes.ipynb
    6. Notebook: Analysis_orientation.ipynb
    7. Notebook: Analysis_marching_cubes.ipynb

## Running the Grafter for creating systems

  - Define System Parameters: Modify parameters such as grafting density, molecule types and system geometry to load or create a matrix. The input_grafter.json files are used as input. The notebook shows an example of how to set the parameters.
  - Load parameters using read_inputs_grafter()
  - Run Grafting Procedures: Use graft_matrix() to apply grafting based on the defined parameters.
  - Assemble a system by connecting different .gro files. Using input_assembler.json as input to read_inputs_assembler(), the methods from Assembler class can be called to build a system. The function run_assembler() runs the assembly.
  - Visualize: Plot structure to check system with plot_system().

## Running the Grafter modules for analyzing

  - Load a simulated system to the NewSystem(gro=<gro_file>, traj=<traj_file>) class. The trajectory file is not mandatory, a single frame can be loaded
  - Call functions for analysis. The following modules are available:
    - Plotting : to visualize
    - RunningCA: to run Contact Angle analysis
    - RunningDEN: to run density analysis
    - RuningLH: to run layer height analysis
    - Running MSD: to run MSD analysis
    - RunningSURF: to run surface analysis (using marching cubes)

