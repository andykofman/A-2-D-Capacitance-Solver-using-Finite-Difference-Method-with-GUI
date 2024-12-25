# -*- coding: utf-8 -*-
"""
2D Capacitance Solver

"""
import sys
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import time
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve
import warnings
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
import numpy.ma as ma
import logging

# Configure backend logger
backend_logger = logging.getLogger('backend')
backend_logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('backend.log')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
backend_logger.addHandler(console_handler)
backend_logger.addHandler(file_handler)

class CoordinateSystem:
    def __init__(self, nx, ny, lx, ly, min_feature_size):
        """
        Initialize coordinate system with proper physical scaling
        
        Args:
            nx, ny: Grid points
            lx, ly: Physical dimensions in meters
            min_feature_size: Smallest feature size in meters
        """
        self.nx = nx
        self.ny = ny
        self.lx_physical = lx
        self.ly_physical = ly
        
        # Compute required resolution based on minimum feature
        points_per_feature = 20  # Industry standard
        dx_required = min_feature_size / points_per_feature
        
        # Verify grid resolution
        if lx/nx > dx_required or ly/ny > dx_required:
            required_nx = int(np.ceil(lx / dx_required))
            required_ny = int(np.ceil(ly / dx_required))
            warnings.warn(f"Grid resolution insufficient. Recommend at least {required_nx}x{required_ny} points")
        
        # Normalization factor (use minimum feature size)
        self.length_scale = min_feature_size
        
        # Create normalized coordinates
        self.lx = lx / self.length_scale
        self.ly = ly / self.length_scale
        
        # Validate grid points and physical dimensions
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid points must be positive integers")
        if lx <= 0 or ly <= 0:
            raise ValueError("Physical dimensions must be positive")
        
        # Create normalized coordinates
        self.x = np.linspace(0, self.lx, nx)
        self.y = np.linspace(0, self.ly, ny)
        
        # Ensure no invalid values
        if np.any(np.isnan(self.x)) or np.any(np.isnan(self.y)):
            raise ValueError("Invalid values in coordinate arrays")
        
        # Physical grid for output
        self.x_physical = self.x * self.length_scale
        self.y_physical = self.y * self.length_scale
        
        # Store grid spacings
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)

    def get_cell_size(self, i, j):
        """Get normalized cell size at given indices"""
        dx = self.dx[min(i, len(self.dx)-1)]
        dy = self.dy[min(j, len(self.dy)-1)]
        return dx, dy

    def get_physical_cell_size(self, i, j):
        """Get physical cell size at given indices"""
        dx, dy = self.get_cell_size(i, j)
        return dx * self.length_scale, dy * self.length_scale

    def physical_to_normalized(self, x_phys, y_phys):
        """Convert physical coordinates to normalized"""
        return x_phys/self.length_scale, y_phys/self.length_scale

    def normalized_to_physical(self, x_norm, y_norm):
        """Convert normalized coordinates to physical"""
        return x_norm*self.length_scale, y_norm*self.length_scale

    def physical_to_grid(self, x_phys, y_phys):
        """Convert physical coordinates to grid indices"""
        x_norm, y_norm = self.physical_to_normalized(x_phys, y_phys)
        i = int(np.clip(x_norm * (self.nx-1) / self.lx, 0, self.nx-1))
        j = int(np.clip(y_norm * (self.ny-1) / self.ly, 0, self.ny-1))
        return i, j

    def update_grid(self, x_new, y_new):
        """Update grid with new coordinates (for nonuniform grid)"""
        self.x = x_new / self.length_scale
        self.y = y_new / self.length_scale
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.x_physical = self.x * self.length_scale
        self.y_physical = self.y * self.length_scale

class LaplaceSolver:
    def __init__(self, coord_system, epsilon_r, boundary_condition='mixed'):
        """Initialize solver with mixed boundary conditions"""
        self.coord_system = coord_system
        self.boundary_condition = boundary_condition
        
        # Initialize epsilon grid
        if np.isscalar(epsilon_r):
            self.epsilon_grid = np.ones((coord_system.nx, coord_system.ny)) * epsilon_r * epsilon_0
        else:
            self.epsilon_grid = epsilon_r * epsilon_0
            
        # Initialize masks
        self.is_boundary = np.zeros((coord_system.nx, coord_system.ny), dtype=bool)
        self.is_boundary[0, :] = self.is_boundary[-1, :] = True
        self.is_boundary[:, 0] = self.is_boundary[:, -1] = True
        
        # Initialize conductor properties
        self.conductor_labels = None
        self.is_conductor = None
        self.potentials = None
        
        print(f"LaplaceSolver initialized with {boundary_condition} boundary conditions")

    def set_conductor_labels(self, conductor_geometries):
        """Set conductor labels in the grid"""
        nx, ny = self.coord_system.nx, self.coord_system.ny
        self.conductor_labels = np.zeros((nx, ny), dtype=int)
        self.is_conductor = np.zeros((nx, ny), dtype=bool)
        
        backend_logger.debug("Setting up conductors")
        
        for idx, geom in enumerate(conductor_geometries, start=1):
            i_slice, j_slice = geom
            print(f"Debug: Conductor {idx} spans [{i_slice.start}:{i_slice.stop}, {j_slice.start}:{j_slice.stop}]")
            
            # Validate indices
            if (i_slice.start < 0 or i_slice.stop > nx or 
                j_slice.start < 0 or j_slice.stop > ny):
                raise ValueError(f"Conductor {idx} geometry exceeds grid boundaries")
                
            self.conductor_labels[i_slice, j_slice] = idx
            self.is_conductor[i_slice, j_slice] = True
            
        print(f"Debug: Total conductor area: {np.sum(self.is_conductor)} cells")
        backend_logger.debug(f"Conductor geometries set: {self.conductor_labels}")

    def setup_linear_system(self, conductor_potentials):
        """Setup linear system with mixed boundary conditions"""
        nx, ny = self.coord_system.nx, self.coord_system.ny
        n = nx * ny
        A = lil_matrix((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                if self.is_conductor[i, j]:
                    # Fixed potential at conductors
                    A[idx, idx] = 1.0
                    b[idx] = conductor_potentials[self.conductor_labels[i, j] - 1]
                    continue

                # Get normalized cell sizes
                dx, dy = self.coord_system.get_cell_size(i, j)
                
                # Calculate interface permittivities
                eps = self.epsilon_grid[i,j]
                eps_left = eps_right = eps_bottom = eps_top = eps
                
                if i > 0:
                    eps_left = 0.5 * (eps + self.epsilon_grid[i-1,j])
                if i < nx-1:
                    eps_right = 0.5 * (eps + self.epsilon_grid[i+1,j])
                if j > 0:
                    eps_bottom = 0.5 * (eps + self.epsilon_grid[i,j-1])
                if j < ny-1:
                    eps_top = 0.5 * (eps + self.epsilon_grid[i,j+1])

                # Interior points
                if not self.is_boundary[i, j]:
                    if i > 0:
                        A[idx, idx-ny] = eps_left / (dx*dx)
                    if i < nx-1:
                        A[idx, idx+ny] = eps_right / (dx*dx)
                    if j > 0:
                        A[idx, idx-1] = eps_bottom / (dy*dy)
                    if j < ny-1:
                        A[idx, idx+1] = eps_top / (dy*dy)
                    A[idx, idx] = -(eps_left + eps_right)/(dx*dx) - \
                                 (eps_bottom + eps_top)/(dy*dy)
                
                # Boundary points with mixed conditions
                else:
                    if i == 0:  # Left boundary (Neumann)
                        A[idx, idx] = eps_right/dx
                        A[idx, idx+ny] = -eps_right/dx
                    elif i == nx-1:  # Right boundary (Neumann)
                        A[idx, idx] = eps_left/dx
                        A[idx, idx-ny] = -eps_left/dx
                    elif j == 0:  # Bottom boundary (Dirichlet)
                        A[idx, idx] = 1.0
                        b[idx] = 0.0
                    elif j == ny-1:  # Top boundary (Dirichlet)
                        A[idx, idx] = 1.0
                        b[idx] = 0.0

        return csr_matrix(A), b

    def solve(self, conductor_potentials):
        """Solve using LU decomposition"""
        A, b = self.setup_linear_system(conductor_potentials)
        
        # Add small regularization term
        diagonal = A.diagonal()
        min_diag = np.min(np.abs(diagonal[diagonal != 0]))
        regularization = min_diag * 1e-8
        A.setdiag(A.diagonal() + regularization)
        
        # Use spsolve directly
        print("Using LU decomposition")
        solution = spsolve(A, b)
        
        # Compute residual
        residual = np.linalg.norm(A.dot(solution) - b)
        print(f"Solver residual: {residual:.2e}")
        
        self.potentials = solution.reshape((self.coord_system.nx, self.coord_system.ny))
        return self.potentials
    
    def _compute_charge(self, region):
        """
        Compute charge using normalized coordinates
        Returns charge in Coulombs
        """
        nx, ny = self.coord_system.nx, self.coord_system.ny
        i_slice, j_slice = region
        i_start, i_end = i_slice.start, i_slice.stop
        j_start, j_end = j_slice.start, j_slice.stop
        
        charge = 0.0
        
        # Compute charge using normalized coordinates
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                dx, dy = self.coord_system.get_cell_size(i, j)  # Get normalized cell sizes
                
                # Calculate normalized E-field components
                if i == i_start:  # Left boundary
                    Ex = -(self.potentials[i, j] - self.potentials[i-1, j]) / dx
                    eps_x = 0.5 * (self.epsilon_grid[i,j] + self.epsilon_grid[i-1,j])
                    charge -= eps_x * Ex * dy  # Corrected charge calculation
                if i == i_end-1:  # Right boundary
                    Ex = -(self.potentials[i+1, j] - self.potentials[i, j]) / dx
                    eps_x = 0.5 * (self.epsilon_grid[i,j] + self.epsilon_grid[i+1,j])
                    charge += eps_x * Ex * dy  # Corrected charge calculation
                if j == j_start:  # Bottom boundary
                    Ey = -(self.potentials[i, j] - self.potentials[i, j-1]) / dy
                    eps_y = 0.5 * (self.epsilon_grid[i,j] + self.epsilon_grid[i,j-1])
                    charge -= eps_y * Ey * dx  # Corrected charge calculation
                if j == j_end-1:  # Top boundary
                    Ey = -(self.potentials[i, j+1] - self.potentials[i, j]) / dy
                    eps_y = 0.5 * (self.epsilon_grid[i,j] + self.epsilon_grid[i,j+1])
                    charge += eps_y * Ey * dx  # Corrected charge calculation
        
        # Scale charge back to physical units
        charge *= self.coord_system.length_scale
        
        return charge

    def calculate_electric_field(self):
        """
        Calculate electric field components using sophisticated methods
        Returns:
            Ex, Ey: Electric field components in V/m
            E_mag: Electric field magnitude in V/m
        """
        nx, ny = self.coord_system.nx, self.coord_system.ny
        dx = self.coord_system.dx[0] * self.coord_system.length_scale
        dy = self.coord_system.dy[0] * self.coord_system.length_scale

        # Create masked array to handle conductor boundaries
        potential_masked = ma.masked_array(self.potentials, mask=self.is_conductor)

        # Calculate E-field components using central differences with gaussian smoothing
        # Smooth potentials before differentiation to reduce numerical noise
        potential_smooth = gaussian_filter1d(potential_masked, sigma=1.0, axis=0)
        potential_smooth = gaussian_filter1d(potential_smooth, sigma=1.0, axis=1)

        # Use sophisticated gradient calculation with proper handling of boundaries
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))

        # Interior points - central difference
        Ex[1:-1, :] = -(potential_smooth[2:, :] - potential_smooth[:-2, :]) / (2 * dx)
        Ey[:, 1:-1] = -(potential_smooth[:, 2:] - potential_smooth[:, :-2]) / (2 * dy)

        # Boundary points - forward/backward difference
        # Left/Right boundaries
        Ex[0, :] = -(potential_smooth[1, :] - potential_smooth[0, :]) / dx
        Ex[-1, :] = -(potential_smooth[-1, :] - potential_smooth[-2, :]) / dx
        
        # Top/Bottom boundaries
        Ey[:, 0] = -(potential_smooth[:, 1] - potential_smooth[:, 0]) / dy
        Ey[:, -1] = -(potential_smooth[:, -1] - potential_smooth[:, -2]) / dy

        # Account for dielectric regions (D = εE)
        epsilon_rel = self.epsilon_grid / epsilon_0
        Ex = Ex / epsilon_rel
        Ey = Ey / epsilon_rel

        # Calculate magnitude
        E_mag = np.sqrt(Ex**2 + Ey**2)

        # Mask fields inside conductors
        Ex = ma.masked_array(Ex, mask=self.is_conductor)
        Ey = ma.masked_array(Ey, mask=self.is_conductor)
        E_mag = ma.masked_array(E_mag, mask=self.is_conductor)

        return Ex, Ey, E_mag

    def _apply_interface_conditions(self, Ex, Ey):
        """
        Apply proper interface conditions at dielectric boundaries
        """
        nx, ny = self.coord_system.nx, self.coord_system.ny
        
        # Find dielectric interfaces
        eps_diff_x = np.diff(self.epsilon_grid, axis=0)
        eps_diff_y = np.diff(self.epsilon_grid, axis=1)
        
        # Apply normal field discontinuity (D⊥ continuous)
        for i in range(nx-1):
            for j in range(ny):
                if abs(eps_diff_x[i,j]) > 1e-10:  # Interface detected
                    eps1 = self.epsilon_grid[i,j]
                    eps2 = self.epsilon_grid[i+1,j]
                    # Adjust normal field component
                    Ex[i,j] *= eps2/eps1
                    Ex[i+1,j] *= eps1/eps2

        for i in range(nx):
            for j in range(ny-1):
                if abs(eps_diff_y[i,j]) > 1e-10:  # Interface detected
                    eps1 = self.epsilon_grid[i,j]
                    eps2 = self.epsilon_grid[i,j+1]
                    # Adjust normal field component
                    Ey[i,j] *= eps2/eps1
                    Ey[i,j+1] *= eps1/eps2

        return Ex, Ey

    def get_interpolated_field(self, x_points, y_points):
        """
        Get interpolated electric field values at arbitrary points
        
        Args:
            x_points, y_points: Arrays of points where field is desired
            
        Returns:
            Ex, Ey: Interpolated field components
        """
        Ex, Ey, _ = self.calculate_electric_field()
        
        # Create regular grid interpolators
        x = self.coord_system.x_physical
        y = self.coord_system.y_physical
        
        Ex_interp = RegularGridInterpolator((x, y), Ex.T)
        Ey_interp = RegularGridInterpolator((x, y), Ey.T)
        
        # Stack coordinates for interpolation
        points = np.column_stack((x_points.flatten(), y_points.flatten()))
        
        # Get interpolated values
        Ex_points = Ex_interp(points).reshape(x_points.shape)
        Ey_points = Ey_interp(points).reshape(y_points.shape)
        
        return Ex_points, Ey_points

class CapacitanceExtraction:
    def __init__(self, physical_dimensions, conductor_geometries, epsilon_r, 
                 min_grid_points=400, use_nonuniform_grid=False, force_grid_size=None):
        """
        Initialize capacitance extraction with proper physical scaling
        
        Args:
            physical_dimensions: (lx, ly) in meters
            conductor_geometries: List of ((x1,x2), (y1,y2)) in meters
            epsilon_r: Relative permittivity (scalar or grid)
            min_grid_points: Minimum grid points per dimension
            force_grid_size: If provided, forces specific grid size regardless of feature size
        """
        lx, ly = physical_dimensions
        
        # Find minimum feature size from geometries
        min_feature = self._find_min_feature(conductor_geometries)
        
        if force_grid_size is not None:
            nx = ny = force_grid_size
        else:
            # Calculate required grid points based on minimum feature
            points_per_feature = 20  # industry standard
            nx = max(min_grid_points, int(np.ceil(lx / min_feature * points_per_feature)))
            ny = max(min_grid_points, int(np.ceil(ly / min_feature * points_per_feature)))
            
        # Initialize coordinate system with proper scaling
        self.coord_system = CoordinateSystem(nx, ny, lx, ly, min_feature)
        
        # Initialize epsilon grid
        if np.isscalar(epsilon_r):
            self.epsilon_grid = np.ones((nx, ny)) * epsilon_r
        else:
            # Use provided epsilon_r grid directly
            self.epsilon_grid = epsilon_r.copy()
        
        # Convert conductor geometries to normalized grid indices
        self.conductor_geometries = self._convert_geometries_to_grid(conductor_geometries)
        
        if use_nonuniform_grid:
            self._setup_nonuniform_grid()
        
        # Initialize solver
        self.solver = LaplaceSolver(self.coord_system, self.epsilon_grid)
        self.solver.set_conductor_labels(self.conductor_geometries)
        self._validate_units(physical_dimensions, conductor_geometries, epsilon_r)
        
        print(f"Initialized with grid size {nx}x{ny}")
        print(f"Minimum feature size: {min_feature*1e6:.2f} µm")
        print(f"Grid resolution: {(lx/nx)*1e6:.2f} µm x {(ly/ny)*1e6:.2f} µm")

    def _find_min_feature(self, geometries):
        """Find minimum feature size from geometries"""
        min_size = float('inf')
        for geom in geometries:
            (x1, x2), (y1, y2) = geom
            min_size = min(min_size, abs(x2-x1), abs(y2-y1))
        return min_size

    def _convert_geometries_to_grid(self, physical_geometries):
        """Convert physical geometries to grid indices"""
        grid_geometries = []
        for geom in physical_geometries:
            (x1, x2), (y1, y2) = geom
            i1, j1 = self.coord_system.physical_to_grid(x1, y1)
            i2, j2 = self.coord_system.physical_to_grid(x2, y2)
            grid_geometries.append((slice(i1, i2+1), slice(j1, j2+1)))
        return grid_geometries
    
    def _validate_units(self, physical_dimensions, conductor_geometries, epsilon_r):
      """Validate physical units and dimensions"""
      lx, ly = physical_dimensions
      
      # Check for reasonable physical dimensions (0.1nm to 1cm)
      if not (1e-10 <= lx <= 1e-2 and 1e-10 <= ly <= 1e-2):
          raise ValueError("Physical dimensions must be between 0.1nm and 1cm")
          
      # Validate conductor geometries
      for (x1, x2), (y1, y2) in conductor_geometries:
          # Check conductor dimensions
          if not (0 <= x1 < x2 <= lx and 0 <= y1 < y2 <= ly):
              raise ValueError("Conductor geometry outside domain bounds")
          # Check minimum feature size (>= 0.1nm)
          if min(x2-x1, y2-y1) < 1e-10:
              raise ValueError("Conductor features must be >= 0.1nm")
              
      # Validate epsilon_r
      if np.isscalar(epsilon_r):
          if epsilon_r <= 0:
              raise ValueError("Relative permittivity must be positive")
      else:
          if np.any(epsilon_r <= 0):
              raise ValueError("Relative permittivity must be positive everywhere")

    def _setup_nonuniform_grid(self):
        """Setup nonuniform grid with proper physical scaling"""
        grid_gen = NonuniformGrid(self.coord_system)
        x_new, y_new = grid_gen.generate_grid(self.conductor_geometries)
        self.coord_system.update_grid(x_new, y_new)

    def extract_capacitance_matrix(self, conductor_potentials):
        """
        Extract capacitance matrix with proper physical scaling
        Returns capacitance in Farads
        """
        start_time = time.time()
        n_conductors = len(self.conductor_geometries)
        capacitance_matrix = np.zeros((n_conductors, n_conductors))
        
        # Compute capacitance matrix
        for j in range(n_conductors):
            test_potentials = np.zeros(n_conductors)
            test_potentials[j] = 1.0
            
            self.solver.solve(test_potentials)
            
            for i in range(n_conductors):
                charge = self.solver._compute_charge(self.conductor_geometries[i])
                capacitance_matrix[i, j] = charge

        # Enforce physical properties
        # capacitance_matrix = self._enforce_physical_properties(capacitance_matrix)
        
        # Compute charges for given potentials
        charges = np.dot(capacitance_matrix, conductor_potentials)
        
        extraction_time = time.time() - start_time
        
        # Print results
        self._print_results(capacitance_matrix, charges, extraction_time)
        self._verify_charge_conservation(capacitance_matrix, charges)

        
        return capacitance_matrix, charges, extraction_time
    def _verify_charge_conservation(self, capacitance_matrix, charges, tolerance=1e-12):
        """Verify charge conservation with high precision"""
        # Total charge should be zero
        total_charge = np.sum(charges)
        if abs(total_charge) > tolerance:
            warnings.warn(f"Charge conservation violated: total charge = {total_charge:.2e} C")
            
        # Verify reciprocity (Cij = Cji)
        max_reciprocity_error = np.max(np.abs(capacitance_matrix - capacitance_matrix.T))
        if max_reciprocity_error > tolerance:
            warnings.warn(f"Reciprocity violated: max error = {max_reciprocity_error:.2e} F")
            
        # Verify positive definiteness
        eigenvals = np.linalg.eigvals(capacitance_matrix)
        if np.any(eigenvals < -tolerance):
            warnings.warn(f"Capacitance matrix not positive definite: min eigenvalue = {np.min(eigenvals):.2e}")
    def _enforce_physical_properties(self, capacitance_matrix):
        """Enforce physical properties of capacitance matrix"""
        n = len(capacitance_matrix)
        
        # Enforce symmetry
        capacitance_matrix = 0.5 * (capacitance_matrix + capacitance_matrix.T)
        
        # Enforce diagonal dominance
        for i in range(n):
            off_diag_sum = np.sum(np.abs(capacitance_matrix[i,:])) - np.abs(capacitance_matrix[i,i])
            if np.abs(capacitance_matrix[i,i]) <= off_diag_sum:
                capacitance_matrix[i,i] = -1 * off_diag_sum
        
        return capacitance_matrix

    def _print_results(self, capacitance_matrix, charges, extraction_time):
        """Print results in physical units"""
        n = len(capacitance_matrix)
        
        print("\nCapacitance Matrix (fF):")
        print("-" * 40)
        for i in range(n):
            row = [f"{c*1e15:8.3f}" for c in capacitance_matrix[i,:]]
            print("  ".join(row))
        
        print("\nCharges (fC):")
        print("-" * 20)
        for i, q in enumerate(charges):
            print(f"Q{i+1}: {q*1e15:8.3f}")
        
        print(f"\nExtraction time: {extraction_time:.3f} seconds")
        


    def visualize_fields(self, conductor_potentials):
            """
            Visualize fields with proper physical scaling and units
            """
            self.solver.solve(conductor_potentials)
            potentials = self.solver.potentials
            
            # Get physical coordinates in meters
            x_m = self.coord_system.x_physical
            y_m = self.coord_system.y_physical
            
            # Convert to micrometers for plotting
            x_um = x_m * 1e6
            y_um = y_m * 1e6
            
            # Calculate electric fields
            Ex, Ey, E_mag = self.solver.calculate_electric_field()
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Common axis settings for all subplots
            axes = [ax1, ax2, ax3, ax4]
            for ax in axes:
                ax.invert_yaxis()  # Invert y-axis to match GUI coordinates
            
            # 1. Potential Distribution
            potential_plot = ax1.pcolormesh(x_um, y_um, potentials.T, 
                                        shading='auto', cmap='viridis')
            fig.colorbar(potential_plot, ax=ax1, label='Potential (V)')
            self._add_conductor_outlines(ax1)
            ax1.set_title('Potential Distribution')
            ax1.set_xlabel('x (µm)')
            ax1.set_ylabel('y (µm)')
            
            # 2. Electric Field Magnitude
            E_mag_plot = ax2.pcolormesh(x_um, y_um, E_mag.T,
                                        shading='auto', cmap='plasma',  # Changed to 'plasma' for better color representation
                                        norm=LogNorm(vmin=E_mag[E_mag > 0].min(), 
                                                    vmax=E_mag.max()))  # Log normalization for better contrast
            fig.colorbar(E_mag_plot, ax=ax2, label='|E| (V/m)')
            self._add_conductor_outlines(ax2)
            ax2.set_title('Electric Field Magnitude')
            ax2.set_xlabel('x (µm)')
            ax2.set_ylabel('y (µm)')
            
            # 3. Electric Field Lines
            # Create uniform grid for streamplot
            x_uniform = np.linspace(x_um[0], x_um[-1], 50)
            y_uniform = np.linspace(y_um[0], y_um[-1], 50)
            X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform)
            
            # Convert to meters for interpolation
            X_meters = X_uniform * 1e-6
            Y_meters = Y_uniform * 1e-6
            
            # Get interpolated field components
            Ex_uniform, Ey_uniform = self.solver.get_interpolated_field(X_meters, Y_meters)
            
            # Create streamplot with correct orientation
            # Note: Keep Ex direction, but flip Ey direction due to inverted y-axis
            ax3.streamplot(x_uniform, y_uniform,
                        Ex_uniform.T, Ey_uniform.T,  # Remove negation, Ey already flipped by axis inversion
                        color='b', density=1.5, linewidth=0.5,
                        arrowsize=1.5)
            
            self._add_conductor_outlines(ax3)
            ax3.set_title('Electric Field Lines')
            ax3.set_xlabel('x (µm)')
            ax3.set_ylabel('y (µm)')
            
            # 4. Dielectric Distribution
            epsilon_plot = ax4.pcolormesh(x_um, y_um, 
                                        self.epsilon_grid.T/epsilon_0,
                                        shading='auto', cmap='viridis')
            fig.colorbar(epsilon_plot, ax=ax4, label='Relative Permittivity')
            self._add_conductor_outlines(ax4)
            ax4.set_title('Dielectric Distribution')
            ax4.set_xlabel('x (µm)')
            ax4.set_ylabel('y (µm)')
            
            plt.tight_layout()
            plt.show()
        
       

    def _add_conductor_outlines(self, ax):
        """Add conductor outlines to plot with correct coordinates"""
        for conductor in self.conductor_geometries:
            # Handle both slice and tuple formats
            if isinstance(conductor[0], slice):
                # Convert slice indices to physical coordinates
                x1 = self.coord_system.x_physical[conductor[0].start]
                x2 = self.coord_system.x_physical[conductor[0].stop-1]
                y1 = self.coord_system.y_physical[conductor[1].start]
                y2 = self.coord_system.y_physical[conductor[1].stop-1]
            else:
                # Direct tuple format from GUI
                (x1, x2), (y1, y2) = conductor
            
            # Convert to micrometers for plotting
            x1_um, x2_um = x1*1e6, x2*1e6
            y1_um, y2_um = y1*1e6, y2*1e6
            width = x2_um - x1_um
            height = y2_um - y1_um
            rect = patches.Rectangle((x1_um, y1_um), width, height,
                                   linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

class NonuniformGrid:
    def __init__(self, coord_system, refinement_factor=4.0):
        """
        Initialize nonuniform grid generator
        
        Args:
            coord_system: CoordinateSystem instance
            refinement_factor: Factor for grid refinement near conductors
        """
        self.coord_system = coord_system
        self.refinement_factor = refinement_factor
        
    def generate_grid(self, conductor_geometries):
        """
        Generate nonuniform grid with proper physical scaling
        
        Returns:
            x_new, y_new: New physical coordinates (in meters)
        """
        # Start with uniform physical coordinates
        x_uniform = np.linspace(0, self.coord_system.lx_physical, self.coord_system.nx)
        y_uniform = np.linspace(0, self.coord_system.ly_physical, self.coord_system.ny)
        
        # Generate refinement weights
        wx = self._generate_weights(x_uniform, conductor_geometries, 'x')
        wy = self._generate_weights(y_uniform, conductor_geometries, 'y')
        
        # Apply nonuniform transformation
        x_new = self._apply_nonuniform_transform(x_uniform, wx)
        y_new = self._apply_nonuniform_transform(y_uniform, wy)
        
        return x_new, y_new
    
    def _generate_weights(self, coords, geometries, direction):
        """Generate refinement weights based on conductor locations"""
        weights = np.ones_like(coords)
        
        for geom in geometries:
            if direction == 'x':
                start = coords[geom[0].start]
                end = coords[geom[0].stop-1]
            else:
                start = coords[geom[1].start]
                end = coords[geom[1].stop-1]
            
            # Add refinement near conductor edges
            dist_to_start = np.abs(coords - start)
            dist_to_end = np.abs(coords - end)
            min_dist = np.minimum(dist_to_start, dist_to_end)
            
            # Scale distances by minimum feature size
            scaled_dist = min_dist / self.coord_system.length_scale
            refinement = 1 + (self.refinement_factor - 1) * np.exp(-scaled_dist**2)
            weights = np.maximum(weights, refinement)
        
        return weights
    
    def _apply_nonuniform_transform(self, coords, weights):
        """Apply nonuniform transformation based on weights"""
        # Normalize weights
        weights = weights / np.mean(weights)
        
        # Calculate cumulative spacing
        dx = np.diff(coords)
        dx_new = dx / weights[:-1]
        x_new = np.zeros_like(coords)
        x_new[1:] = np.cumsum(dx_new)
        
        # Scale to match original domain size
        x_new *= coords[-1] / x_new[-1]
        
        return x_new

    

    def visualize_grid(self, conductor_geometries):
        """
        Visualize the non-uniform grid with conductors
        
        Args:
            conductor_geometries: List of conductor geometries in grid coordinates
        """
        # Generate the non-uniform grid
        x_new, y_new = self.generate_grid(conductor_geometries)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot vertical grid lines
        for x in x_new:
            ax.axvline(x*1e6, color='lightgray', linewidth=0.5)
        
        # Plot horizontal grid lines
        for y in y_new:
            ax.axhline(y*1e6, color='lightgray', linewidth=0.5)
        
        # Plot conductors
        for geom in conductor_geometries:
            i_slice, j_slice = geom
            x1 = x_new[i_slice.start]
            x2 = x_new[min(i_slice.stop-1, len(x_new)-1)]
            y1 = y_new[j_slice.start]
            y2 = y_new[min(j_slice.stop-1, len(y_new)-1)]
            
            rect = patches.Rectangle(
                (x1*1e6, y1*1e6), 
                (x2-x1)*1e6, 
                (y2-y1)*1e6,
                linewidth=2,
                edgecolor='red',
                facecolor='lightcoral',
                alpha=0.3,
                label='Conductor'
            )
            ax.add_patch(rect)
        
        # Set limits and labels
        ax.set_xlim(0, self.coord_system.lx_physical*1e6)
        ax.set_ylim(0, self.coord_system.ly_physical*1e6)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_title('Non-uniform Grid with Conductors')
        ax.grid(False)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('nonuniform_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig










