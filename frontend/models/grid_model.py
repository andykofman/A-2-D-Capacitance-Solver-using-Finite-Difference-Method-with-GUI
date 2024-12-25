"""
Grid model
 serve as our data interface between the frontend and backend.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from ..utils.constants import (
    GRID_DEFAULT_SIZE, 
    DEFAULT_DIMENSIONS, 
    DEFAULT_EPSILON_R,
    GRID_MIN_SIZE,
    MIN_FEATURE_SIZE    
)

@dataclass
class Conductor:
    """Data class for conductor properties"""
    id: int
    geometry: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x1,x2), (y1,y2)) in meters
    potential: float = 0.0
    name: str = ""

@dataclass
class DielectricRegion:
    """Data class for dielectric region properties"""
    geometry: Tuple[Tuple[float, float], Tuple[float, float]]
    epsilon_r: float
    name: str = ""

class GridModel:
    """Model managing the grid state and computations"""
    
    def __init__(self, controller = None):
        """Initialize grid model"""
        # Grid properties
        self.nx, self.ny = GRID_DEFAULT_SIZE
        self.lx, self.ly = DEFAULT_DIMENSIONS
        self.controller = controller
        self.coord_system = None
        
        # Material properties
        self.epsilon_grid = np.ones((self.nx, self.ny)) * DEFAULT_EPSILON_R
        
        # Elements
        self.conductors: List[Conductor] = []
        self.dielectric_regions: List[DielectricRegion] = []
        
        # Results storage
        self.potential_distribution: Optional[np.ndarray] = None
        self.capacitance_matrix: Optional[np.ndarray] = None
        self.charges: Optional[np.ndarray] = None
        
        # Reference to property panel (will be set by MainWindow)
        self.property_panel = None
        
        # Reference to results panel (will be set by MainWindow)
        self.results_panel = None
        
        self.use_nonuniform_grid = False
        self.base_nx, self.base_ny = GRID_DEFAULT_SIZE
    
    # def add_conductor(self, geometry: Tuple[Tuple[float, float], Tuple[float, float]], 
    #                  potential: float = 0.0) -> int:
    #     """Add a conductor to the grid"""
    #     conductor_id = len(self.conductors) + 1
    #     conductor = Conductor(
    #         id=conductor_id,
    #         geometry=geometry,
    #         potential=potential,
    #         name=f"Conductor {conductor_id}"
    #     )
    #     self.conductors.append(conductor)
    #     self.notify_property_panel()
    #     return conductor_id

    def add_conductor(self, geometry: Tuple[Tuple[float, float], Tuple[float, float]], 
                       potential: float = 1.0) -> int:
        """Add a conductor to the grid"""
        print(f"Adding conductor with geometry: {geometry} and potential: {potential}")
        
        # Store the conductor geometry in a consistent format
        self.conductors.append(Conductor(
            id=len(self.conductors) + 1,
            geometry=geometry,
            potential=potential,
            name=f"Conductor {len(self.conductors) + 1}"
        ))
        self.notify_property_panel()
        self.check_small_feature()  # Check for small features after adding
        print(f"Conductor added with geometry (in meters): {geometry}")
        return len(self.conductors)
    
    def add_dielectric(self, geometry: Tuple[Tuple[float, float], Tuple[float, float]], 
                       epsilon_r: float = DEFAULT_EPSILON_R) -> int:
        """Add a dielectric region to the grid"""
        print(f"Debug: Adding dielectric with geometry: {geometry} and epsilon_r: {epsilon_r}")
        
        if self.check_grid_resolution(geometry):
            print("Debug: Resolution needs to be increased for dielectric")
            return -1
        
        region = DielectricRegion(
            geometry=geometry,
            epsilon_r=epsilon_r,
            name=f"Dielectric {len(self.dielectric_regions) + 1}"
        )
        self.dielectric_regions.append(region)
        self.update_epsilon_grid()
        self.notify_property_panel()
        self.check_small_feature()  # Check for small features after adding
        print(f"Debug: Added dielectric region {len(self.dielectric_regions)}")
        return len(self.dielectric_regions)
    
    def update_epsilon_grid(self):
        """Update epsilon grid based on dielectric regions"""
        # Start with air everywhere
        self.epsilon_grid.fill(DEFAULT_EPSILON_R)
        
        # Apply dielectric regions in order
        for region in self.dielectric_regions:
            (x1, x2), (y1, y2) = region.geometry
            # Convert physical coordinates to grid indices
            i1 = int(x1 / self.lx * self.nx)
            i2 = int(x2 / self.lx * self.nx)
            j1 = int(y1 / self.ly * self.ny)
            j2 = int(y2 / self.ly * self.ny)
            self.epsilon_grid[i1:i2+1, j1:j2+1] = region.epsilon_r
            
        # Trigger backend update if controller exists
        # if self.controller:
        #     self.controller.update_backend({
        #         'nx': self.nx,
        #         'ny': self.ny,
        #         'lx': self.lx,
        #         'ly': self.ly
        #     }, self.conductors, self.dielectric_regions)
    
    def get_conductor_geometries(self) -> List[Tuple[slice, slice]]:
        """Get conductor geometries in grid coordinates for backend"""
        geometries = []
        for conductor in self.conductors:
            (x1, x2), (y1, y2) = conductor.geometry
            # Convert physical coordinates to grid indices
            i1 = int(x1 / self.lx * self.nx)
            i2 = int(x2 / self.lx * self.nx)
            j1 = int(y1 / self.ly * self.ny)
            j2 = int(y2 / self.ly * self.ny)
            geometries.append((slice(i1, i2+1), slice(j1, j2+1)))
        return geometries
    
    def get_conductor_potentials(self) -> List[float]:
        """Get list of conductor potentials"""
        return [c.potential for c in self.conductors]
    
    def clear(self):
        """Clear all elements from the grid"""
        self.conductors.clear()
        self.dielectric_regions.clear()
        self.epsilon_grid.fill(DEFAULT_EPSILON_R)
        self.potential_distribution = None
        self.capacitance_matrix = None
    
    def resize_grid(self, nx: int, ny: int):
        """Resize the grid with validation"""
        self.nx = max(nx, GRID_MIN_SIZE)
        self.ny = max(ny, GRID_MIN_SIZE)
        self.epsilon_grid = np.ones((self.nx, self.ny)) * DEFAULT_EPSILON_R
        
        # Ensure the epsilon grid is updated correctly
        self.update_epsilon_grid()
        
        # Update backend through controller if available
        if self.controller:
            self.controller.update_backend({
                'nx': self.nx,
                'ny': self.ny,
                'lx': self.lx,
                'ly': self.ly
            }, [], [])
    
    def get_minimum_feature_size(self):
        """Calculate minimum feature size from existing geometries"""
        if not self.conductors and not self.dielectric_regions:
            return 10e-6  # Default minimum feature size in meters
        
        min_size = float('inf')
        # Check conductor dimensions
        for conductor in self.conductors:
            (x1, x2), (y1, y2) = conductor.geometry
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            min_size = min(min_size, width, height)
            
        # Check dielectric dimensions
        for dielectric in self.dielectric_regions:
            (x1, x2), (y1, y2) = dielectric.geometry
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            min_size = min(min_size, width, height)
            
        return min_size if min_size != float('inf') else 1e-6
    
    def notify_property_panel(self):
        """Notify property panel to update"""
        if hasattr(self, 'property_panel'):
            self.property_panel.update_properties()

    def remove_conductor(self, conductor: Conductor):
        """Remove a conductor from the grid"""
        if conductor in self.conductors:
            self.conductors.remove(conductor)
            self.notify_property_panel()

    def remove_dielectric(self, region: DielectricRegion):
        """Remove a dielectric region from the grid"""
        if region in self.dielectric_regions:
            self.dielectric_regions.remove(region)
            self.update_epsilon_grid()
            self.notify_property_panel()

####################################################################

    def check_grid_resolution(self, new_geometry) -> bool:
        """
        Check if current grid resolution is sufficient for new geometry
        Returns True if resolution needs to be increased
        """
        # Calculate minimum feature size
        (x1, x2), (y1, y2) = new_geometry
        feature_size = min(abs(x2-x1), abs(y2-y1))
        
        # Calculate required points based on industry standard
        points_per_feature = 20
        dx_required = feature_size / points_per_feature
        
        # Calculate required grid points
        nx_required = int(np.ceil(self.lx / dx_required))
        ny_required = int(np.ceil(self.ly / dx_required))
        
        # Check if current resolution is sufficient
        current_dx = self.lx / self.nx
        current_dy = self.ly / self.ny
        
        print(f"Debug: New feature size: {feature_size*1e6:.2f}µm")
        print(f"Debug: Required resolution: {dx_required*1e6:.2f}µm")
        print(f"Debug: Current resolution: {current_dx*1e6:.2f}µm")
        
        if current_dx > dx_required or current_dy > dx_required:
            # Update grid size if needed
            self.nx = max(self.nx, nx_required)
            self.ny = max(self.ny, ny_required)
            return True
        
        return False
    
    def store_solution(self, backend_data):
        """Store solution data from backend calculations
    
        Args:
            backend_data (dict): Dictionary containing calculation results
                - potentials: Potential distribution
                - Ex, Ey: Electric field components
                - E_mag: Electric field magnitude
                - capacitance_matrix: Capacitance matrix
                - charges: Conductor charges
        """
        try:
            print("Debug: Storing solution in grid model")
            
            # Store all results in model
            self.potential_distribution = backend_data.get('potentials')
            self.Ex = backend_data.get('Ex')
            self.Ey = backend_data.get('Ey') 
            self.E_mag = backend_data.get('E_mag')
            self.capacitance_matrix = backend_data.get('capacitance_matrix')
            self.charges = backend_data.get('charges')
            
            # Update results panel if it exists
            if hasattr(self, 'results_panel') and self.results_panel is not None:
                print("Debug: Forwarding results to panel")
                self.results_panel.update_results(backend_data)
            else:
                print("Debug: No results panel available")
                
        except Exception as e:
            print(f"Error storing solution: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_solution(self):
        """Get current solution"""
        return self.potential_distribution, (
            getattr(self, 'Ex', None),
            getattr(self, 'Ey', None),
            getattr(self, 'E_mag', None)
        )
    
    def update_grid_discretization(self):
        """Update grid discretization based on geometry"""
        if not self.use_nonuniform_grid:
            self.coord_system = None
            return
            
        # Convert conductors and dielectrics to dict format
        conductors = [
            {'geometry': c.geometry, 'potential': c.potential}
            for c in self.conductors
        ]
        
        dielectrics = [
            {'geometry': d.geometry, 'epsilon_r': d.epsilon_r}
            for d in self.dielectric_regions
        ]
        
        if not conductors and not dielectrics:
            self.coord_system = None
            return
            
        # Update coordinate system through controller
        if self.controller:
            self.controller.update_backend({
                'nx': self.nx,
                'ny': self.ny,
                'lx': self.lx,
                'ly': self.ly
            }, conductors, dielectrics)

            
    def _calculate_refinement_regions(self, geometries):
        """Calculate regions that need refinement"""
        refined_regions = []
        for (x1, x2), (y1, y2) in geometries:
            # Add region around geometry with 20% padding
            dx = abs(x2 - x1) * 0.2
            dy = abs(y2 - y1) * 0.2
            refined_regions.append((
                (max(0, x1 - dx), min(self.lx, x2 + dx)),
                (max(0, y1 - dy), min(self.ly, y2 + dy))
            ))
        return refined_regions

    def reset(self):
        """Reset the grid model to its initial state"""
        self.conductors.clear()
        self.dielectric_regions.clear()
        self.epsilon_grid.fill(DEFAULT_EPSILON_R)
        self.potential_distribution = None
        self.capacitance_matrix = None
        self.charges = None
        self.nx = self.base_nx
        self.ny = self.base_ny
        print("Debug: Grid model reset to initial state")

    def check_small_feature(self):
        """Check for small features and prompt user for action"""
        min_feature_size = self.get_minimum_feature_size()
        recommended_size = self.calculate_recommended_size(min_feature_size)
        
        if min_feature_size < MIN_FEATURE_SIZE:
            # Trigger a dialog to ask user for action
            self.dialog_show = True
            self.controller.show_small_feature_dialog(recommended_size)
        
    def calculate_recommended_size(self, min_feature_size):
        """Calculate recommended grid size based on minimum feature size"""
        return (self.base_nx + (min_feature_size // 20), self.base_ny + (min_feature_size // 20))