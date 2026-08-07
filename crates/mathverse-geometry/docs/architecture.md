# Architecture

## Purpose
2D/3D geometric primitives and algorithms for scientific computing.

## Core Components
- `shapes2d` / `shapes3d`: Primitive definitions
- `primitives2d`: Extended curves and segments
- `spatial`: Bounding structures and trees
- `intersection`: Collision detection
- `distance`: Proximity calculations
- `mesh3d`: Indexed triangle meshes
- `metrics`: Area, angle, inertia

## Data Flow
```
User Input -> Shape -> Measure / Transform -> Output
```

## Module Boundaries
Public modules expose traits and structs only. Internal algorithms are hidden.
