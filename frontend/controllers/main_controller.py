from PyQt5.QtWidgets import QMainWindow
import numpy as np
from ..views.main_window import MainWindow
from ..models.grid_model import GridModel
from backend import CoordinateSystem, LaplaceSolver, NonuniformGrid, CapacitanceExtraction
from ..utils.constants import (
    DEFAULT_DIMENSIONS, 
    DEFAULT_EPSILON_R,
    GRID_DEFAULT_SIZE
)

import matplotlib.pyplot as plt

class MainController:
    def __init__(self):
        """Initialize the main controller"""
        # Initialize models
        self.grid_model = GridModel(controller=self)
        
        # Initialize backend components
        self._init_backend()
        
        # Initialize main window
        self.main_window = MainWindow(self)

        self.property_panel = self.main_window.property_panel

        
    def _init_backend(self):
        """Initialize backend components"""
        # Get grid properties from model
        nx, ny = self.grid_model.nx, self.grid_model.ny
        lx, ly = self.grid_model.lx, self.grid_model.ly
        min_feature_size = self.grid_model.get_minimum_feature_size()
        
        print(f"Debug: Initializing backend with grid size {nx}x{ny}")
        print(f"Debug: Physical dimensions: {lx}x{ly} meters")
        print(f"Debug: Minimum feature size: {min_feature_size} meters")
        
        try:
            # Create coordinate system
            self.coord_system = CoordinateSystem(
                nx=self.grid_model.nx,
                ny=self.grid_model.ny,
                lx=self.grid_model.lx,
                ly=self.grid_model.ly,
                min_feature_size=self.grid_model.get_minimum_feature_size()
            )
            
            # Initialize capacitance extractor
            self.capacitance_extractor = CapacitanceExtraction(
                physical_dimensions=(self.grid_model.lx, self.grid_model.ly),
                conductor_geometries=[],  # Start with empty geometry
                epsilon_r=DEFAULT_EPSILON_R,
                min_grid_points=self.grid_model.nx,
                use_nonuniform_grid=self.grid_model.use_nonuniform_grid
            )
            
            print("Debug: Backend initialization successful")
            
        except Exception as e:
            print(f"Error initializing backend: {e}")
            raise
    def show(self):
        """Show the main window"""
        self.main_window.show()

    def update_backend(self, grid_properties, conductors, dielectrics, force_grid_size=None):
        """Update backend state following backend logic"""
        try:
            # Convert conductor geometries (keeping physical coordinates)
            conductor_geometries = []
            conductor_potentials = []
            for conductor in conductors:
                # Ensure we are accessing the correct attributes
                conductor_geometries.append(conductor.geometry)  # Ensure this is a Conductor object
                conductor_potentials.append(conductor.potential)  # Ensure this is a float
        
            # Create epsilon_r grid with correct dimensions
            nx, ny = int(grid_properties['nx']), int (grid_properties['ny'])
            epsilon_grid = np.ones((nx, ny)) * DEFAULT_EPSILON_R
        
            # Apply dielectric regions to the grid
            for dielectric in dielectrics:
                (x1, x2), (y1, y2) = dielectric.geometry
                i1 = int(np.clip(x1 / grid_properties['lx'] * nx, 0, nx-1))
                i2 = int(np.clip(x2 / grid_properties['lx'] * nx, 0, nx-1))
                j1 = int(np.clip(y1 / grid_properties['ly'] * ny, 0, ny-1))
                j2 = int(np.clip(y2 / grid_properties['ly'] * ny, 0, ny-1))
                epsilon_grid[i1:i2+1, j1:j2+1] = dielectric.epsilon_r
        
            # Create new capacitance extractor with updated geometry and dielectrics
            self.capacitance_extractor = CapacitanceExtraction(
                physical_dimensions=(grid_properties['lx'], grid_properties['ly']),
                conductor_geometries=conductor_geometries,
                epsilon_r=epsilon_grid,
                min_grid_points=nx,
                use_nonuniform_grid=self.grid_model.use_nonuniform_grid,
                force_grid_size=force_grid_size
            )
        
            # Update coordinate system
            self.coord_system = self.capacitance_extractor.coord_system
            self.grid_model.coord_system = self.coord_system
        
            # Try to calculate solution if conductors exist
            if conductor_geometries:
                # Calculate capacitance matrix and charges
                C_matrix, charges, extraction_time = self.capacitance_extractor.extract_capacitance_matrix(
                    conductor_potentials
                )
            
                # Calculate electric field
                Ex, Ey, E_mag = self.capacitance_extractor.solver.calculate_electric_field()
            
                # Package results
                backend_data = {
                    'capacitance_matrix': C_matrix,
                    'charges': charges,
                    'potentials': self.capacitance_extractor.solver.potentials,
                    'Ex': Ex,
                    'Ey': Ey,
                    'E_mag': E_mag,
                    'extraction_time': extraction_time,
                    'epsilon_grid': epsilon_grid
                }
                return backend_data
        
            return None
            
        except Exception as e:
            print(f"Backend update failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_capacitance(self):
        """Calculate capacitance for current configuration"""
        try:
            if not hasattr(self.grid_model, 'saved_state'):
                print("Debug: No saved state available")
                return False
                
            print("Debug: Starting capacitance calculation...")
            
            # Use the saved state for calculation
            backend_data = self.update_backend(
                grid_properties=self.grid_model.saved_state['grid_properties'],
                conductors=self.grid_model.conductors,  # Pass conductor objects directly
                dielectrics=self.grid_model.dielectric_regions  # Pass dielectric objects directly
            )
            
            if backend_data is not None:
                print("Debug: Backend calculation successful")
                print(f"Debug: Capacitance matrix shape: {backend_data['capacitance_matrix'].shape}")
                
                # Store results in grid model
                self.grid_model.store_solution(backend_data)
                
                # Explicitly update results panel
                if hasattr(self.grid_model, 'results_panel') and self.grid_model.results_panel is not None:
                    print("Debug: Updating results panel...")
                    self.grid_model.results_panel.update_results(backend_data)
                else:
                    print("Debug: Results panel not available")
                
                return True
                
            print("Debug: Backend calculation returned no data")
            return False
                
        except Exception as e:
            print(f"Capacitance calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _convert_and_validate_geometries(self, conductors, nx, ny):
            """Convert and validate conductor geometries to grid coordinates"""
            conductor_geometries = []
            for conductor in conductors:
                (x1, x2), (y1, y2) = conductor.geometry
                i1 = int(x1 / self.grid_model.lx * nx)
                i2 = int(x2 / self.grid_model.lx * nx)
                j1 = int(y1 / self.grid_model.ly * ny)
                j2 = int(y2 / self.grid_model.ly * ny)
                
                # Validate indices
                if i1 < 0 or i2 >= nx or j1 < 0 or j2 >= ny:
                    raise ValueError(f"Conductor geometry {conductor.geometry} exceeds grid boundaries")
                
                conductor_geometries.append((slice(i1, i2+1), slice(j1, j2+1)))
            
            return conductor_geometries

    def run_test_case(self):
        """Run a test case with predefined geometry"""
        # Clear existing elements
        self.grid_model.clear()
        
        # Add test conductors
        conductor1_geom = ((20e-6, 30e-6), (40e-6, 60e-6))
        conductor2_geom = ((70e-6, 80e-6), (40e-6, 60e-6))
        self.grid_model.add_conductor(conductor1_geom, potential=1.0)
        self.grid_model.add_conductor(conductor2_geom, potential=0.0)
        
        # Add test dielectric
        dielectric_geom = ((40e-6, 60e-6), (35e-6, 65e-6))
        self.grid_model.add_dielectric(dielectric_geom, epsilon_r=3.9)
        
        # Convert geometries to grid coordinates for the solver
        conductor_geometries = self.grid_model.get_conductor_geometries()
        conductor_potentials = self.grid_model.get_conductor_potentials()
        
        # Set conductor labels in solver
        self.solver.set_conductor_labels(conductor_geometries)
        
        # Solve the system
        potential_dist = self.solver.solve(conductor_potentials)
        
        # Calculate fields
        Ex, Ey, E_mag = self.solver.calculate_electric_field()
        
        # Store solution in grid model
        self.grid_model.store_solution(potential_dist, (Ex, Ey, E_mag))
        
        # Use backend visualization
        self.CapacitanceExtraction.visualize_fields(conductor_potentials)

    def show_small_feature_dialog(self, recommended_size):
        """Show dialog for small feature options"""
        if not hasattr(self, 'property_panel'):
            print("Error: Property panel is not initialized.")
            return
        # Call the property panel to show the dialog
        self.property_panel.show_small_feature_dialog(recommended_size)