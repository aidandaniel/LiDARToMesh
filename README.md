# LiDARToMesh

Turns airborne LiDAR .laz files into Poisson surface meshes and generates robotics no-go zones using gradient mapping.

## Overview

This project:
- Loads airborne LiDAR (.laz/.las) point clouds
- Applies voxel-based decimation to reduce point count
- Estimates normals using a KD-tree neighborhood
- Reconstructs a watertight surface mesh using Poisson surface reconstruction
- Produces gradient-based robot no-go zone maps (using dot products / slope/normal analysis)

Typical uses:
- Rapidly converting large LiDAR tiles into meshes for visualization or simulation
- Creating terrain slope / accessibility layers to identify areas unsafe for robots

## Features

- Voxel decimation to control output density and speed up further processing
- KD-tree normal estimation (fast and robust for dense point clouds)
- Poisson surface reconstruction for smooth, watertight meshes
- Gradient mapping for robot navigation safety zones

## Requirements

- Python 3.8+
- pip

Python libraries required (install with pip):
- laspy (with lazrs backend for .laz support): `laspy[lazrs]`
- numpy
- open3d

Install with:

pip install laspy[lazrs] numpy open3d

(If you use a virtual environment it's recommended. On some platforms open3d may require extra system packages — consult Open3D installation docs if you run into build issues.)

## Quick workflow (high level)

1. Download or collect your .laz/.las files (see dataset reference below).
2. Load point cloud with laspy and convert to a numpy array or Open3D point cloud.
3. Apply voxel decimation to reduce the number of points while preserving shape:
   - Choose an appropriate voxel size (trade-off: smaller = more detail, larger = faster).
4. Estimate normals using a KD-tree (nearest neighbors) on the decimated cloud:
   - Use a neighborhood size tuned to point density (e.g., 16–64 neighbors).
   - Orient normals consistently (e.g., towards the sensor or upwards).
5. Run Poisson surface reconstruction:
   - Tune depth / scale parameters to balance detail and memory usage.
   - Post-process the mesh (crop, remove small components, simplify).
6. Compute gradient mapping for robotics no-go zones:
   - Use dot products between normals and the "up" vector to estimate slope.
   - Threshold by slope or curvature to mark no-go areas for the robot.



Adjust voxel_size, KD-tree neighbors, Poisson depth, and slope thresholds to your data and desired output quality.

## Tips & Parameter Guidance

- Voxel size: start with 0.2–1.0 m for airborne LiDAR; smaller values preserve detail but increase compute.
- KD-tree neighbors: 16–64 is typical; more neighbors smooth normals but may blur sharp features.
- Poisson depth: higher depth yields more detail but uses more memory; try depth 8–12 depending on tile size.
- Postprocessing: remove small disconnected mesh components and low-density regions returned by Poisson (density pruning).
- If memory is an issue, process tiles in chunks and stitch or merge meshes afterward.

## Dataset reference

The datasets used for development/experimentation were obtained from OpenTopography:
https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.112016.3294.1

## Output

- Mesh files (e.g., .ply/.obj/.stl) suitable for visualization and simulation
- Raster or vector layers marking robotics no-go zones (based on slope/curvature thresholds)
