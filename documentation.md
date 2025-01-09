# Comprehensive Documentation for 2D Capacitance Solver Project

## Overview

The **2D Capacitance Solver** is a Python-based application designed to simulate and analyze the capacitance of 2D structures. The project is divided into two main modules: the **backend** and the **frontend**. The backend handles the core computational logic, including grid generation, solving Laplace's equation, and extracting capacitance values. The frontend provides a graphical user interface (GUI) for users to interact with the solver, visualize results, and manage the simulation setup.

This documentation provides a detailed explanation of the project's structure, functionality, and usage.

---

## Table of Contents

1. **Project Structure**
   - Backend Module
   - Frontend Module
2. **Backend Module**
   - Coordinate System
   - Laplace Solver
   - Capacitance Extraction
   - Nonuniform Grid Generation
3. **Frontend Module**
   - Main Controller
   - Grid Model
   - Property Panel
   - Grid Drawer
   - Main Window
   - Results Panel
4. **Usage Guide**
   - Running the Application
   - Setting Up the Simulation
   - Visualizing Results
5. **Dependencies**
6. **Future Enhancements**

---

## 1. Project Structure

The project is organized into two main modules:

- **Backend Module**: Handles the core computational logic, including grid generation, solving Laplace's equation, and extracting capacitance values.
- **Frontend Module**: Provides a graphical user interface (GUI) for users to interact with the solver, visualize results, and manage the simulation setup.

### Backend Module

The backend module is responsible for the following tasks:
- **Grid Generation**: Creates a coordinate system and grid for the simulation.
- **Laplace Solver**: Solves Laplace's equation to compute the potential distribution in the grid.
- **Capacitance Extraction**: Extracts the capacitance matrix and charges from the potential distribution.
- **Nonuniform Grid Generation**: Generates a nonuniform grid for better resolution around small features.

### Frontend Module

The frontend module provides a user-friendly interface for:
- **Grid Setup**: Allows users to define conductors and dielectric regions.
- **Simulation Control**: Enables users to run simulations and visualize results.
- **Results Display**: Displays the capacitance matrix, charges, and electric field distribution.

---

## 2. Backend Module

### 2.1 Coordinate System

The `CoordinateSystem` class is responsible for managing the grid's physical and normalized coordinates. It handles the conversion between physical coordinates (in meters) and grid indices.

#### Key Methods:
- `__init__`: Initializes the coordinate system with grid points, physical dimensions, and minimum feature size.
- `get_cell_size`: Returns the normalized cell size at given indices.
- `physical_to_normalized`: Converts physical coordinates to normalized coordinates.
- `normalized_to_physical`: Converts normalized coordinates to physical coordinates.
- `physical_to_grid`: Converts physical coordinates to grid indices.

### 2.2 Laplace Solver

The `LaplaceSolver` class solves Laplace's equation to compute the potential distribution in the grid. It supports mixed boundary conditions and handles conductor geometries.

#### Key Methods:
- `__init__`: Initializes the solver with a coordinate system and boundary conditions.
- `set_conductor_labels`: Sets conductor labels in the grid.
- `setup_linear_system`: Sets up the linear system for solving Laplace's equation.
- `solve`: Solves the linear system using LU decomposition.
- `calculate_electric_field`: Calculates the electric field components from the potential distribution.

### 2.3 Capacitance Extraction

The `CapacitanceExtraction` class extracts the capacitance matrix and charges from the potential distribution. It also handles the visualization of electric fields.

#### Key Methods:
- `__init__`: Initializes the capacitance extractor with physical dimensions, conductor geometries, and dielectric properties.
- `extract_capacitance_matrix`: Extracts the capacitance matrix and charges.
- `visualize_fields`: Visualizes the potential distribution and electric fields.

### 2.4 Nonuniform Grid Generation

The `NonuniformGrid` class generates a nonuniform grid with higher resolution around small features.

#### Key Methods:
- `generate_grid`: Generates a nonuniform grid based on conductor geometries.
- `visualize_grid`: Visualizes the nonuniform grid with conductors.

---

## 3. Frontend Module

### 3.1 Main Controller

The `MainController` class acts as the bridge between the frontend and backend. It initializes the backend components and manages the interaction between the GUI and the solver.

#### Key Methods:
- `__init__`: Initializes the main controller and backend components.
- `update_backend`: Updates the backend with the current grid properties, conductors, and dielectrics.
- `calculate_capacitance`: Triggers the capacitance calculation and updates the results panel.

### 3.2 Grid Model

The `GridModel` class manages the grid state, including conductors, dielectric regions, and simulation results.

#### Key Methods:
- `add_conductor`: Adds a conductor to the grid.
- `add_dielectric`: Adds a dielectric region to the grid.
- `update_epsilon_grid`: Updates the permittivity grid based on dielectric regions.
- `get_conductor_geometries`: Returns the conductor geometries in grid coordinates.
- `get_conductor_potentials`: Returns the list of conductor potentials.

### 3.3 Property Panel

The `PropertyPanel` class provides a GUI for users to set properties of conductors and dielectric regions.

#### Key Methods:
- `update_properties`: Updates the property panel with current elements.
- `_on_potential_changed`: Handles changes in conductor potential.
- `_on_epsilon_changed`: Handles changes in dielectric constant.

### 3.4 Grid Drawer

The `GridDrawer` class is an OpenGL-based widget for visualizing the grid, conductors, and dielectric regions.

#### Key Methods:
- `_draw_grid`: Draws the background grid.
- `_draw_conductors`: Draws conductors on the grid.
- `_draw_dielectrics`: Draws dielectric regions on the grid.
- `mousePressEvent`: Handles mouse press events for drawing and selecting elements.

### 3.5 Main Window

The `MainWindow` class is the main application window that contains the grid drawer, property panel, and results panel.

#### Key Methods:
- `_init_ui`: Initializes the main UI components.
- `_create_actions`: Creates application actions (e.g., new, open, save).
- `_create_toolbars`: Creates toolbars for drawing and editing.
- `_create_dock_widgets`: Creates dock widgets for the property and results panels.

### 3.6 Results Panel

The `ResultsPanel` class displays the simulation results, including the capacitance matrix and charges.

#### Key Methods:
- `update_results`: Updates the results panel with new data from the backend.
- `_update_matrix_display`: Updates the capacitance matrix display.
- `_update_charges_display`: Updates the charges display.
- `_show_visualization`: Shows the field visualization using backend methods.

---

## 4. Usage Guide

### 4.1 Running the Application

To run the application, execute the `main.py` script. This will launch the GUI, allowing you to set up the simulation, run calculations, and visualize results.

### 4.2 Setting Up the Simulation

1. **Define Conductors**: Use the drawing tools to define conductors on the grid. Set their potentials using the property panel.
2. **Define Dielectric Regions**: Add dielectric regions to the grid and set their permittivity values.
3. **Adjust Grid Settings**: Optionally, enable nonuniform grid generation for better resolution around small features.

### 4.3 Visualizing Results

1. **Run Simulation**: Click the "Calculate Capacitance" button to run the simulation.
2. **View Results**: The results panel will display the capacitance matrix and charges.
3. **Visualize Fields**: Use the "Show Field Visualization" button to visualize the electric field distribution.

---

## 5. Dependencies

The project relies on the following Python libraries:
- **NumPy**: For numerical computations.
- **SciPy**: For solving linear systems and numerical integration.
- **Matplotlib**: For visualization.
- **PyQt5**: For the graphical user interface.
- **OpenGL**: For rendering the grid and elements.

---


---

This documentation provides a comprehensive overview of the 2D Capacitance Solver project. For further details, refer to the source code and inline comments.
