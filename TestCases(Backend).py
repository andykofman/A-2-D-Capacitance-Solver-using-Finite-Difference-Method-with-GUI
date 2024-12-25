# -*- coding: utf-8 -*-
"""
Test cases for the 2D Capacitance Solver

"""
import sys
from backend  import CapacitanceExtraction, LaplaceSolver, NonuniformGrid, CoordinateSystem
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sksparse.cholmod import cholesky
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy import integrate


# # # Domain size
# lx = ly = 100e-6  # 20 micrometers
# nx = ny = 400    # Grid points

# # Material properties
# epsilon_r = 3.9  # relative permittivity of SiO2
# epsilon_grid = np.ones((nx, ny)) * epsilon_0 * epsilon_r

# # Conductor dimensions
# conductor_width = 2e-6   # 2 µm width
# conductor_height = 2e-6  # 2 µm height
# spacing = 4e-6          # 4 µm spacing between conductors

# # Calculate center positions for symmetry
# center_x = lx/2
# center_y = ly/2

# # Define conductor geometries
# conductor_geometries = [
#     # Left conductor
#     ((center_x - spacing - conductor_width, center_x - spacing), 
#         (center_y - conductor_height/2, center_y + conductor_height/2)),
    
#     # Center conductor
#     ((center_x - conductor_width/2, center_x + conductor_width/2), 
#         (center_y - conductor_height/2, center_y + conductor_height/2)),
    
#     # Right conductor
#     ((center_x + spacing, center_x + spacing + conductor_width), 
#         (center_y - conductor_height/2, center_y + conductor_height/2))
# ]

# # Initialize extractor
# extractor = CapacitanceExtraction(nx, ny, conductor_geometries, epsilon_grid, 
#                                 lx=lx, ly=ly)

# # Test voltages
# conductor_potentials = np.array([0.0, 1.0, 0.0])  # Center at 1V, others at 0V

# # Perform extraction
# capacitance_matrix, charges, extraction_time = extractor.extract_capacitance_matrix(conductor_potentials)

# # Print results
# print("\nTest Results:")
# print(f"Grid size: {nx}x{ny}")
# print(f"Physical size: {lx*1e6:.1f}x{ly*1e6:.1f} µm")
# print(f"Extraction time: {extraction_time:.4f} seconds")
# print("\nCapacitance Matrix (F):")
# print(capacitance_matrix)
# print("\nCharges (C):")
# print(charges)

# # Original test case
# print("\nTest Case 1: Center conductor at 1V")
# conductor_potentials1 = np.array([0.0, 1.0, 0.0])
# fig = extractor.visualize_fields(conductor_potentials)
# capacitance_matrix, charges1, time1 = extractor.extract_capacitance_matrix(conductor_potentials1)

# print(f"\nCapacitance Matrix (F):")
# print(capacitance_matrix)
# print("\nCharges (C):")
# print(charges1)

# # # New test case: Alternating voltages
# print("\nTest Case 2: Alternating voltages [1V, 0V, 1V]")
# conductor_potentials2 = np.array([1.0, 0.0, 1.0])
# fig2 = extractor.visualize_fields(conductor_potentials2)
# _, charges2, time2 = extractor.extract_capacitance_matrix(conductor_potentials2)

# print("\nCharges for [1V, 0V, 1V] (C):")
# print(charges2)

# Validation checks
def validate_results(capacitance_matrix, charges1, charges2):
    """Validate physical properties of the solution"""
    print("\nValidation Checks:")
    
    # 1. Charge conservation
    print(f"1. Charge conservation:")
    print(f"   Sum of charges (Case 1): {np.sum(charges1):.2e} C")
    print(f"   Sum of charges (Case 2): {np.sum(charges2):.2e} C")
    
    # 2. Capacitance matrix symmetry
    max_asymmetry = np.max(np.abs(capacitance_matrix - capacitance_matrix.T))
    print(f"2. Matrix symmetry:")
    print(f"   Maximum asymmetry: {max_asymmetry:.2e} F")
    
    # 3. Diagonal dominance
    is_diag_dominant = all(abs(capacitance_matrix[i,i]) > 
                          sum(abs(capacitance_matrix[i,j]) for j in range(3) if j != i)
                          for i in range(3))
    print(f"3. Diagonal dominance: {is_diag_dominant}")
    
    # 4. Reciprocity check (C12/C21 ratios)
    c12_ratio = abs(capacitance_matrix[0,1] / capacitance_matrix[1,0])
    c23_ratio = abs(capacitance_matrix[1,2] / capacitance_matrix[2,1])
    print(f"4. Reciprocity ratios:")
    print(f"   C12/C21: {c12_ratio:.6f}")
    print(f"   C23/C32: {c23_ratio:.6f}")
    
    # 5. Physical scaling check
    print("5. Physical scaling check:")
    print(f"   Self-capacitance range: [{np.min(np.diag(capacitance_matrix)):.2e}, "
          f"{np.max(np.diag(capacitance_matrix)):.2e}] F")
    print(f"   Mutual-capacitance range: [{np.min(capacitance_matrix[~np.eye(3,dtype=bool)]):.2e}, "
          f"{np.max(capacitance_matrix[~np.eye(3,dtype=bool)]):.2e}] F")

# Run validation
# validate_results(capacitance_matrix, charges1, charges2)


##################################################################################


def convergence_test(base_size=50, max_multiplier=8, num_points=5):
    """
    Perform convergence analysis with increasing grid resolution.
    
    Args:
        base_size: Base grid size (nx = ny = base_size)
        max_multiplier: Maximum multiplication factor for grid size
        num_points: Number of test points
    """
    # Generate grid sizes
    multipliers = np.logspace(0, np.log10(max_multiplier), num_points)
    grid_sizes = [int(base_size * m) for m in multipliers]
    
    # Storage for results
    results = {
        'grid_sizes': grid_sizes,
        'C11': [],
        'C12': [],
        'extraction_times': [],
        'relative_changes': []
    }
    
    # Reference geometry (scaled for each grid size)
    lx = ly = 100e-6
    conductor_width = 2e-6
    conductor_height = 2e-6
    spacing = 4e-6
    
    print("\nConvergence Test:")
    print("-----------------")
    
    # Test each grid size
    for nx in grid_sizes:
        ny = nx  # Keep square grid
        print(f"\nTesting {nx}x{ny} grid...")
        
        # Create epsilon grid
        epsilon_grid = np.ones((nx, ny)) * epsilon_0 * 3.9
        
        # Calculate center positions
        center_x = lx/2
        center_y = ly/2
        
        # Define conductor geometries
        conductor_geometries = [
            ((center_x - spacing - conductor_width, center_x - spacing), 
             (center_y - conductor_height/2, center_y + conductor_height/2)),
            ((center_x - conductor_width/2, center_x + conductor_width/2), 
             (center_y - conductor_height/2, center_y + conductor_height/2)),
            ((center_x + spacing, center_x + spacing + conductor_width), 
             (center_y - conductor_height/2, center_y + conductor_height/2))
        ]
        
        # Run extraction
        extractor = CapacitanceExtraction(nx, ny, conductor_geometries, epsilon_grid, 
                                        lx=lx, ly=ly, use_nonuniform_grid=True)
        
        conductor_potentials = np.array([0.0, 1.0, 0.0])
        C, _, time = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Store results (keeping proper signs)
        results['C11'].append(C[1,1])  # Self-capacitance (positive)
        results['C12'].append(C[0,1])  # Mutual capacitance (negative)
        results['extraction_times'].append(time)
        
        if len(results['C11']) > 1:
            rel_change = abs(results['C11'][-1] - results['C11'][-2]) / abs(results['C11'][-2])
            results['relative_changes'].append(rel_change)
            print(f"Relative change in C11: {rel_change:.2e}")
        
        print(f"C11: {results['C11'][-1]:.3e} F")
        print(f"C12: {results['C12'][-1]:.3e} F")
        print(f"Time: {time:.2f} s")
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot C11 convergence (should be positive)
    ax1.semilogx(grid_sizes, results['C11'], 'bo-')
    ax1.set_xlabel('Grid Size (N)')
    ax1.set_ylabel('C11 (F)')
    ax1.set_title('Self-Capacitance Convergence')
    ax1.grid(True)
    
    # Plot C12 convergence (should be negative)
    ax2.semilogx(grid_sizes, results['C12'], 'ro-')
    ax2.set_xlabel('Grid Size (N)')
    ax2.set_ylabel('C12 (F)')
    ax2.set_title('Mutual-Capacitance Convergence')
    ax2.grid(True)
    
    # Plot relative change
    if len(results['relative_changes']) > 0:
        ax3.loglog(grid_sizes[1:], results['relative_changes'], 'go-')
        ax3.set_xlabel('Grid Size (N)')
        ax3.set_ylabel('Relative Change')
        ax3.set_title('Convergence Rate')
        ax3.grid(True)
    
    # Plot computation time
    ax4.loglog(grid_sizes, results['extraction_times'], 'mo-')
    ax4.set_xlabel('Grid Size (N)')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Computational Cost')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results, fig

# # Run convergence test
# results, fig = convergence_test(base_size=50, max_multiplier=8, num_points=20)

def validate_parallel_plate():
    """
    Validate solver against reference data for parallel plate capacitors.
    Tests configurations from the literature with known analytical solutions.
    """
    # Test cases from the table (using smaller dimensions for testing)
    test_cases = [
        {'L': 61e-6, 'd': 15e-6, 'C_ref': 20e-15},  # Scaled down by 10x
        {'L': 91.4e-6, 'd': 15e-6, 'C_ref': 22.6e-15},
        {'L': 30e-6, 'd': 15e-6, 'C_ref': 16.2e-15},
        {'L': 50e-6, 'd': 15e-6, 'C_ref': 26.58e-15}
    ]
    
    results = []
    
    print("\nParallel Plate Capacitor Validation:")
    print("-" * 60)
    print(f"{'L (µm)':>10} {'d (µm)':>10} {'r':>10} {'C_ref (fF)':>12} {'C_num (fF)':>12} {'Error (%)':>10}")
    print("-" * 60)
    
    for case in test_cases:
        L = case['L']  # side length in meters
        d = case['d']  # air gap in meters
        C_ref = case['C_ref']  # reference capacitance in Farads
        
        # Domain size should be larger than plate size
        domain_size = L * 1.5
        
        # Calculate plate positions (centered in domain)
        offset = (domain_size - L) / 2
        plate1_y = (domain_size - d) / 2
        plate2_y = plate1_y + d
        
        # Define conductor geometries
        conductor_geometries = [
            ((offset, offset + L), (plate1_y, plate1_y + 0.1e-6)),  # Bottom plate (very thin)
            ((offset, offset + L), (plate2_y, plate2_y + 0.1e-6))   # Top plate (very thin)
        ]
        
        # Initialize extractor with reasonable grid size
        nx = ny = 400  # Fixed grid size
        extractor = CapacitanceExtraction(
            lx = domain_size,
            ly = domain_size, 
            conductor_geometries=conductor_geometries,
            epsilon_r=1.0,  # Air
            nx =400, 
            ny= 400,
            use_nonuniform_grid= True
        )
        
        # Extract capacitance with 1V potential difference
        conductor_potentials = [1.0, 0.0]
        C_matrix, _, _ = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Get numerical capacitance (magnitude of mutual capacitance)
        C_num = abs(C_matrix[0,1] *1e3)
        
        # Calculate error
        error_percent = abs(C_num - C_ref) / C_ref * 100
        
        # Store results
        results.append({
            'L': L,
            'd': d,
            'r': L/d,
            'C_ref': C_ref,
            'C_num': C_num,
            'error': error_percent
        })
        
        # Print results (converting to µm and fF for display)
        print(f"{L*1e6:10.1f} {d*1e6:10.1f} {L/d:10.2f} {C_ref*1e15:12.2f} {C_num*1e15:12.2f} {error_percent:10.2f}")
        
        # Visualize field for one case
        if abs(L - 50e-6) < 1e-9:  # Save visualization for 50µm case
            fig = extractor.visualize_fields(conductor_potentials)
            plt.savefig('parallel_plate_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("-" * 60)
    
    # Calculate average error
    avg_error = np.mean([r['error'] for r in results])
    print(f"\nAverage error: {avg_error:.2f}%")
    
    
    return results

validation_results = validate_parallel_plate()




    
# Example usage with proper physical units
# def main():
#     # Define physical dimensions
#     lx = ly = 1000e-6  # 100 µm domain size
    
#     # Define conductor geometries in physical units (meters)
#     conductor_geometries = [
#         ((200e-6, 300e-6), (400e-6, 600e-6)),  # Conductor 1
#         ((700e-6, 800e-6), (400e-6, 600e-6))   # Conductor 2
#     ]
    
#     # Define material properties
#     epsilon_r = 3.9  # Silicon dioxide
    
#     # Initialize capacitance extraction
#     extractor = CapacitanceExtraction(
#         physical_dimensions=(lx, ly),
#         conductor_geometries=conductor_geometries,
#         epsilon_r=epsilon_r,
#         use_nonuniform_grid=True
#     )
    
#     # Define conductor potentials
#     conductor_potentials = [1.0, 0.0]  # 1V and 0V
    
#     # Extract capacitance
#     C_matrix, charges, time = extractor.extract_capacitance_matrix(conductor_potentials)
    
#     # Visualize results
#     fig = extractor.visualize_fields(conductor_potentials)
#     plt.show()
    
#     # Print physical results
#     print("\nResults Summary:")
#     print("-" * 50)
#     print(f"Domain size: {lx*1e6:.1f} µm × {ly*1e6:.1f} µm")
#     print(f"Minimum feature size: {extractor.coord_system.length_scale*1e6:.2f} µm")
#     print(f"Grid resolution: {extractor.coord_system.nx} × {extractor.coord_system.ny}")
#     print("\nCapacitance Matrix (fF):")
#     print(C_matrix * 1e15)  # Convert to femtofarads
#     print("\nCharges (fC):")
#     print(charges * 1e15)   # Convert to femtocoulombs
#     fig = extractor.visualize_fields(conductor_potentials)

#     print("\nVisualization saved as 'field_visualization.png'")


def test_three_conductors():
    """Test case with three symmetric conductors"""
    # Define physical dimensions
    lx = ly = 100e-6  # 100 µm domain size
    
    # Define three symmetric conductor geometries (meters)
    conductor_geometries = [
        ((20e-6, 30e-6), (40e-6, 60e-6)),  # Left conductor
        ((45e-6, 55e-6), (40e-6, 60e-6)),  # Center conductor
        ((70e-6, 80e-6), (40e-6, 60e-6))   # Right conductor
    ]
    
    # Define material properties
    epsilon_r = 3.9  # Silicon dioxide
    
    # Initialize capacitance extraction
    extractor = CapacitanceExtraction(
        physical_dimensions=(lx, ly),
        conductor_geometries=conductor_geometries,
        epsilon_r=epsilon_r,
        use_nonuniform_grid=True
    )
    
    # Define conductor potentials (V1=1V, V2=0V, V3=0V)
    conductor_potentials = [0.0, 10.0, 0.0]
    
    # Extract capacitance
    C_matrix, charges, time = extractor.extract_capacitance_matrix(conductor_potentials)
    
    # Visualize results
    fig = extractor.visualize_fields(conductor_potentials)
    
    # Print results
    print("\nThree Conductor Test Results:")
    print("-" * 50)
    print(f"Domain size: {lx*1e6:.1f} µm × {ly*1e6:.1f} µm")
    print(f"Grid resolution: {extractor.coord_system.nx} × {extractor.coord_system.ny}")
    print("\nCapacitance Matrix (fF):")
    print(C_matrix * 1e15)
    print("\nCharges (fC):")
    print(charges * 1e15)

def test_two_dielectrics():
    """Test case with two different dielectric regions"""
    # Define physical dimensions
    lx = ly = 100e-6  # 100 µm domain size
    
    # Define conductor geometries (meters)
    conductor_geometries = [
        ((20e-6, 30e-6), (40e-6, 60e-6)),  # Left conductor
        ((70e-6, 80e-6), (40e-6, 60e-6))   # Right conductor
    ]
    
    # Create two-dielectric permittivity grid
    nx = ny = 400
    epsilon_r = np.ones((nx, ny)) * 3.9  # Start with SiO2
    
    # Define second dielectric region (e.g., Si3N4, εr = 7.5) in right half
    epsilon_r[:, ny//2:] = 7.5
    
    # Initialize capacitance extraction
    extractor = CapacitanceExtraction(
        physical_dimensions=(lx, ly),
        conductor_geometries=conductor_geometries,
        epsilon_r=epsilon_r,
        use_nonuniform_grid=True
    )
    
    # Define conductor potentials
    conductor_potentials = [1.0, 0.0]
    
    # Extract capacitance
    C_matrix, charges, time = extractor.extract_capacitance_matrix(conductor_potentials)
    
    # Visualize results
    fig = extractor.visualize_fields(conductor_potentials)
    
    # Print results
    print("\nTwo Dielectric Test Results:")
    print("-" * 50)
    print(f"Domain size: {lx*1e6:.1f} µm × {ly*1e6:.1f} µm")
    print(f"Grid resolution: {extractor.coord_system.nx} × {extractor.coord_system.ny}")
    print("\nDielectric properties:")
    print("Region 1: εr = 3.9 (SiO2)")
    print("Region 2: εr = 7.5 (Si3N4)")
    print("\nCapacitance Matrix (fF):")
    print(C_matrix * 1e15)
    print("\nCharges (fC):")
    print(charges * 1e15)

def main():
    """Run all test cases"""
    # Original test
    # print("\nOriginal Two Conductor Test:")
    # print("-" * 50)
    # test_original()


    # test_physical_scales()

    # Three conductor test
    # test_three_conductors()
    # validate_solver_co()
    
    # Two dielectric test
    # test_two_dielectrics()
    
def test_physical_scales():
    """Test capacitance extraction at different physical scales"""
    
    # Test cases with different length scales
    test_cases = [
        {
            'name': 'Nanometer Scale',
            'scale': 1e-9,  # nanometers
            'domain': (1000, 1000),  # 1000nm x 1000nm
            'conductor_dims': [(200, 300, 400, 600),  # conductor 1: (x1,x2,y1,y2) in nm
                             (700, 800, 400, 600)]    # conductor 2
        },
        {
            'name': 'Micrometer Scale',
            'scale': 1e-6,  # micrometers
            'domain': (100, 100),  # 100µm x 100µm
            'conductor_dims': [(20, 30, 40, 60),      # in µm
                             (70, 80, 40, 60)]
        },
        {
            'name': 'Mixed Scale',
            'scale': 1e-6,  # reference in micrometers
            'domain': (50, 50),  # 50µm x 50µm
            'conductor_dims': [(2, 3, 4, 6),          # small conductors (2-6µm)
                             (40, 41, 40, 41)]        # large conductor (40-41µm)
        },
        {
            'name': 'Large Scale',
            'scale': 1e-3,  # millimeters
            'domain': (1, 1),  # 1mm x 1mm
            'conductor_dims': [(0.2, 0.3, 0.4, 0.6),  # in mm
                             (0.7, 0.8, 0.4, 0.6)]
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']} Test:")
        print("-" * 50)
        
        # Convert dimensions to meters
        lx = ly = case['domain'][0] * case['scale']
        
        # Convert conductor geometries to meters
        conductor_geometries = [
            ((x1 * case['scale'], x2 * case['scale']),
             (y1 * case['scale'], y2 * case['scale']))
            for x1, x2, y1, y2 in case['conductor_dims']
        ]
        
        # Calculate minimum feature size
        min_feature = min(
            min(abs(x2-x1), abs(y2-y1))
            for (x1, x2), (y1, y2) in conductor_geometries
        )
        
        print(f"Domain size: {case['domain'][0]} × {case['domain'][1]} "
              f"{case['name'].split()[0]}")
        print(f"Minimum feature size: {min_feature/case['scale']:.2f} "
              f"{case['name'].split()[0]}")
        
        # Initialize extractor with appropriate grid resolution
        extractor = CapacitanceExtraction(
            physical_dimensions=(lx, ly),
            conductor_geometries=conductor_geometries,
            epsilon_r=3.9,
            use_nonuniform_grid=True
        )
        
        # Extract capacitance
        conductor_potentials = [1.0, 0.0]
        C_matrix, charges, time = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Print results in appropriate units
        if case['scale'] == 1e-9:  # nanometer scale
            cap_scale = 1e15  # show in fF
            cap_unit = 'fF'
        elif case['scale'] == 1e-6:  # micrometer scale
            cap_scale = 1e12  # show in pF
            cap_unit = 'pF'
        else:  # millimeter scale
            cap_scale = 1e12  # show in pF
            cap_unit = 'pF'
        
        print(f"\nCapacitance Matrix ({cap_unit}):")
        print(C_matrix * cap_scale)
        print(f"\nCharges ({cap_unit}):")
        print(charges * cap_scale)
        print(f"\nComputation time: {time:.2f} seconds")
        
        # Save visualization with appropriate scale in filename
        fig = extractor.visualize_fields(conductor_potentials)
        plt.savefig(f'fields_{case["name"].lower().replace(" ", "_")}.png')
        plt.close()


def test_original():
    """Original test case with two conductors"""
    # Define physical dimensions
    lx = ly = 100e-6  # 100 µm domain size
    
    # Define conductor geometries
    conductor_geometries = [
        ((20e-6, 30e-6), (40e-6, 60e-6)),  # Conductor 1
        ((70e-6, 80e-6), (40e-6, 60e-6))   # Conductor 2
    ]
    
    # Define material properties
    epsilon_r = 3.9  # Silicon dioxide
    
    # Initialize capacitance extraction
    extractor = CapacitanceExtraction(
        physical_dimensions=(lx, ly),
        conductor_geometries=conductor_geometries,
        epsilon_r=epsilon_r,
        use_nonuniform_grid=True
    )
    
    # Define conductor potentials
    conductor_potentials = [1.0, 0.0]
    
    # Extract capacitance
    C_matrix, charges, time = extractor.extract_capacitance_matrix(conductor_potentials)
    
    # Visualize results
    fig = extractor.visualize_fields(conductor_potentials)
    
    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Domain size: {lx*1e6:.1f} µm × {ly*1e6:.1f} µm")
    print(f"Grid resolution: {extractor.coord_system.nx} × {extractor.coord_system.ny}")
    print("\nCapacitance Matrix (fF):")
    print(C_matrix * 1e15)
    print("\nCharges (fC):")
    print(charges * 1e15)

def convergence_test(base_size=300, max_multiplier=10, num_points=20):
    """
    Perform convergence analysis with increasing grid resolution.
    
    Args:
        base_size: Base grid size (nx = ny = base_size)
        max_multiplier: Maximum multiplication factor for grid size
        num_points: Number of test points
    """
    # Generate grid sizes
    multipliers = np.logspace(0, np.log10(max_multiplier), num_points)
    grid_sizes = [int(base_size * m) for m in multipliers]
    
    # Storage for results
    results = {
        'grid_sizes': grid_sizes,
        'capacitance_matrix': [],
        'extraction_times': [],
        'relative_changes': []
    }
    
    # Define test case geometry
    lx = ly = 100e-6  # 100 µm domain size
    
    # Define three symmetric conductor geometries (meters)
    conductor_geometries = [
        ((20e-6, 30e-6), (40e-6, 60e-6)),  # Left conductor
        ((45e-6, 55e-6), (40e-6, 60e-6)),  # Center conductor
        ((70e-6, 80e-6), (40e-6, 60e-6))   # Right conductor
    ]
    
    print("\nConvergence Test:")
    print("-----------------")
    
    # Test each grid size
    for nx in grid_sizes:
        print(f"\nTesting {nx}x{nx} grid...")
        
        # Initialize extractor
        extractor = CapacitanceExtraction(
            physical_dimensions=(lx, ly),
            conductor_geometries=conductor_geometries,
            epsilon_r=3.9,
            min_grid_points=nx,
            use_nonuniform_grid=True
        )
        
        # Test potentials: Center conductor at 1V, others at 0V
        conductor_potentials = [0.0, 1.0, 0.0]
        
        # Extract capacitance
        C_matrix, _, time = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Store results
        results['capacitance_matrix'].append(C_matrix)
        results['extraction_times'].append(time)
        
        # Calculate relative change if not first iteration
        if len(results['capacitance_matrix']) > 1:
            prev_C = results['capacitance_matrix'][-2]
            curr_C = results['capacitance_matrix'][-1]
            rel_change = np.max(np.abs((curr_C - prev_C) / prev_C))
            results['relative_changes'].append(rel_change)
            print(f"Maximum relative change: {rel_change:.2e}")
        
        print(f"Capacitance Matrix (fF):\n{C_matrix * 1e15}")
        print(f"Extraction time: {time:.2f} s")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot C11 convergence
    C11_values = [C[1,1] for C in results['capacitance_matrix']]
    ax1.semilogx(grid_sizes, C11_values, 'bo-')
    ax1.set_xlabel('Grid Size (N)')
    ax1.set_ylabel('C11 (F)')
    ax1.set_title('Self-Capacitance Convergence')
    ax1.grid(True)
    
    # Plot C12 convergence
    C12_values = [C[0,1] for C in results['capacitance_matrix']]
    ax2.semilogx(grid_sizes, C12_values, 'ro-')
    ax2.set_xlabel('Grid Size (N)')
    ax2.set_ylabel('C12 (F)')
    ax2.set_title('Mutual-Capacitance Convergence')
    ax2.grid(True)
    
    # Plot relative change
    if results['relative_changes']:
        ax3.loglog(grid_sizes[1:], results['relative_changes'], 'go-')
        ax3.set_xlabel('Grid Size (N)')
        ax3.set_ylabel('Relative Change')
        ax3.set_title('Convergence Rate')
        ax3.grid(True)
    
    # Plot computation time
    ax4.loglog(grid_sizes, results['extraction_times'], 'mo-')
    ax4.set_xlabel('Grid Size (N)')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Computational Cost')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results, fig

def convergence_test2(base_size=50, max_multiplier=15, num_points=20):
    """
    Perform convergence analysis with increasing grid resolution.
    
    Args:
        base_size: Base grid size (nx = ny = base_size)
        max_multiplier: Maximum multiplication factor for grid size
        num_points: Number of test points
    """
    # Generate grid sizes
    multipliers = np.logspace(0, np.log10(max_multiplier), num_points)
    grid_sizes = [int(base_size * m) for m in multipliers]
    
    # Storage for results
    results = {
        'grid_sizes': grid_sizes,
        'C11': [],
        'C12': [],
        'extraction_times': [],
        'relative_changes': []
    }
    
    # Reference geometry (scaled for each grid size)
    lx = ly = 1000e-6
    conductor_width = 20e-6
    conductor_height = 20e-6
    spacing = 40e-6
    
    print("\nConvergence Test:")
    print("-----------------")
    
    # Test each grid size
    for nx in grid_sizes:
        ny = nx  # Keep square grid
        print(f"\nTesting {nx}x{ny} grid...")
        
        # Calculate center positions
        center_x = lx/2
        center_y = ly/2
        
        # Define conductor geometries with smaller dimensions
        conductor_geometries = [
            ((center_x - spacing - conductor_width, center_x - spacing), 
             (center_y - conductor_height/2, center_y + conductor_height/2)),
            ((center_x - conductor_width/2, center_x + conductor_width/2), 
             (center_y - conductor_height/2, center_y + conductor_height/2)),
            ((center_x + spacing, center_x + spacing + conductor_width), 
             (center_y - conductor_height/2, center_y + conductor_height/2))
        ]
        
        # Initialize extractor
        extractor = CapacitanceExtraction(
            physical_dimensions=(lx, ly),
            conductor_geometries=conductor_geometries,
            epsilon_r=3.9,
            min_grid_points=nx,
            use_nonuniform_grid=True,
            force_grid_size=nx
        )
        
        # Test potentials: Center conductor at 1V, others at 0V
        conductor_potentials = [0.0, 1.0, 0.0]
        
        # Extract capacitance
        C_matrix, _, time = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Store results (keeping proper signs)
        results['C11'].append(C_matrix[1,1])  # Self-capacitance (positive)
        results['C12'].append(C_matrix[0,1])  # Mutual capacitance (negative)
        results['extraction_times'].append(time)
        
        if len(results['C11']) > 1:
            rel_change = abs(results['C11'][-1] - results['C11'][-2]) / abs(results['C11'][-2])
            results['relative_changes'].append(rel_change)
            print(f"Relative change in C11: {rel_change:.2e}")
        
        print(f"C11: {results['C11'][-1]:.3e} F")
        print(f"C12: {results['C12'][-1]:.3e} F")
        print(f"Time: {time:.2f} s")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot C11 convergence
    ax1.semilogx(grid_sizes, results['C11'], 'bo-')
    ax1.set_xlabel('Grid Size (N)')
    ax1.set_ylabel('C11 (F)')
    ax1.set_title('Self-Capacitance Convergence')
    ax1.grid(True)
    
    # Plot C12 convergence
    ax2.semilogx(grid_sizes, results['C12'], 'ro-')
    ax2.set_xlabel('Grid Size (N)')
    ax2.set_ylabel('C12 (F)')
    ax2.set_title('Mutual-Capacitance Convergence')
    ax2.grid(True)
    
    # Plot relative change
    if results['relative_changes']:
        ax3.loglog(grid_sizes[1:], results['relative_changes'], 'go-')
        ax3.set_xlabel('Grid Size (N)')
        ax3.set_ylabel('Relative Change')
        ax3.set_title('Convergence Rate')
        ax3.grid(True)
    
    # Plot computation time
    ax4.loglog(grid_sizes, results['extraction_times'], 'mo-')
    ax4.set_xlabel('Grid Size (N)')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Computational Cost')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results, fig

# Run convergence test
# results, fig = convergence_test2(base_size=300, max_multiplier=5, num_points=20)

def validate_parallel_plate():
    """
    Validate solver against reference data for parallel plate capacitors.
    Tests configurations from the literature with known analytical solutions.
    """
    # Test cases from the table (using smaller dimensions for testing)
    test_cases = [
        {'L': 61e-6, 'd': 15e-6, 'C_ref': 20e-15},  # Scaled down by 10x
        {'L': 91.4e-6, 'd': 15e-6, 'C_ref': 22.6e-15},
        {'L': 30e-6, 'd': 15e-6, 'C_ref': 16.2e-15},
        {'L': 50e-6, 'd': 15e-6, 'C_ref': 26.58e-15}
    ]
    
    results = []
    
    print("\nParallel Plate Capacitor Validation:")
    print("-" * 60)
    print(f"{'L (µm)':>10} {'d (µm)':>10} {'r':>10} {'C_ref (fF)':>12} {'C_num (fF)':>12} {'Error (%)':>10}")
    print("-" * 60)
    
    for case in test_cases:
        L = case['L']  # side length in meters
        d = case['d']  # air gap in meters
        C_ref = case['C_ref']  # reference capacitance in Farads
        
        # Domain size should be larger than plate size
        domain_size = L * 1.5
        
        # Calculate plate positions (centered in domain)
        offset = (domain_size - L) / 2
        plate1_y = (domain_size - d) / 2
        plate2_y = plate1_y + d
        
        # Define conductor geometries
        conductor_geometries = [
            ((offset, offset + L), (plate1_y, plate1_y + 0.1e-6)),  # Bottom plate (very thin)
            ((offset, offset + L), (plate2_y, plate2_y + 0.1e-6))   # Top plate (very thin)
        ]
        
        # Initialize extractor with reasonable grid size
        nx = ny = 400  # Fixed grid size
        extractor = CapacitanceExtraction(
            physical_dimensions=(domain_size, domain_size),
            conductor_geometries=conductor_geometries,
            epsilon_r=1.0,  # Air
            force_grid_size=400,
            use_nonuniform_grid= True
        )
        
        # Extract capacitance with 1V potential difference
        conductor_potentials = [1.0, 0.0]
        C_matrix, _, _ = extractor.extract_capacitance_matrix(conductor_potentials)
        
        # Get numerical capacitance (magnitude of mutual capacitance)
        C_num = abs(C_matrix[0,1] *1e3)
        
        # Calculate error
        error_percent = abs(C_num - C_ref) / C_ref * 100
        
        # Store results
        results.append({
            'L': L,
            'd': d,
            'r': L/d,
            'C_ref': C_ref,
            'C_num': C_num,
            'error': error_percent
        })
        
        # Print results (converting to µm and fF for display)
        print(f"{L*1e6:10.1f} {d*1e6:10.1f} {L/d:10.2f} {C_ref*1e15:12.2f} {C_num*1e15:12.2f} {error_percent:10.2f}")
        
        # Visualize field for one case
        if abs(L - 50e-6) < 1e-9:  # Save visualization for 50µm case
            fig = extractor.visualize_fields(conductor_potentials)
            plt.savefig('parallel_plate_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("-" * 60)
    
    # Calculate average error
    avg_error = np.mean([r['error'] for r in results])
    print(f"\nAverage error: {avg_error:.2f}%")
    
    
    return results

# Run validation
# validation_results = validate_parallel_plate()
# test_three_conductors()


def test_coplanar_capacitor():
    """
    Test case: Coplanar capacitor with dielectric interface
    This geometry is well-studied and has analytical solutions for comparison
    """
    # Physical dimensions (in meters)
    width = 10e-6  # 10 µm
    gap = 5e-6    # 2 µm
    length = 20e-6  # 20 µm
    
    # Define physical domain - make it larger to accommodate conductors
    lx = 50e-6  # 50 µm
    ly = 40e-6  # 40 µm
    
    # Center the conductors in the domain
    x_start = (lx - (2*width + gap))/2  # Center horizontally
    y_start = (ly - length)/2           # Center vertically
    
    # Define two conductors with gap in between
    conductor1 = ((x_start, x_start + width), 
                 (y_start, y_start + length))  # Left conductor
    conductor2 = ((x_start + width + gap, x_start + 2*width + gap), 
                 (y_start, y_start + length))  # Right conductor
    
    # Create dielectric layers (adjust grid size to match domain)
    nx = ny = 800  # Increased grid resolution
    epsilon_r = np.ones((nx, ny))  # Initialize with air
    epsilon_r[:, :ny//2] = 3.9  # Bottom half is SiO2 (εr = 3.9)
    
    # Initialize solver
    cap_solver = CapacitanceExtraction(
        physical_dimensions=(lx, ly),
        conductor_geometries=[conductor1, conductor2],
        epsilon_r=epsilon_r,
        min_grid_points=800,
        use_nonuniform_grid=True
    )
    
    # Test voltages
    voltages = np.array([1.0, -1.0])  # Differential voltage
    
    # Extract capacitance matrix
    C_matrix, charges, solve_time = cap_solver.extract_capacitance_matrix(voltages)
    
    # Visualize results
    fig = cap_solver.visualize_fields(voltages)
    
    # Print detailed results
    print("\nCoplanar Capacitor Test Results:")
    print(f"Domain size: {lx*1e6:.1f} x {ly*1e6:.1f} µm")
    print(f"Conductor width: {width*1e6:.1f} µm")
    print(f"Gap width: {gap*1e6:.1f} µm")
    print(f"Conductor length: {length*1e6:.1f} µm")
    print("\nCapacitance Matrix (fF):")
    print(C_matrix * 1e15)  # Convert to femtofarads
    
    # Calculate key parameters
    mutual_capacitance = C_matrix[0,1] * 1e15  # Convert to fF
    self_capacitance = C_matrix[0,0] * 1e15
    
    print(f"\nMutual Capacitance: {mutual_capacitance:.2f} fF")
    print(f"Self Capacitance: {self_capacitance:.2f} fF")
    print(f"Solve time: {solve_time:.3f} seconds")
    
    return C_matrix, charges, fig

# Run the test
if __name__ == "__main__":
    C_matrix, charges, fig = test_coplanar_capacitor()


def visualize_comparison(coord_system, conductor_geometries):
    """
    Visualize uniform vs non-uniform grid comparison
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Uniform Grid (left plot)
    x_uniform = np.linspace(0, coord_system.lx_physical, coord_system.nx)
    y_uniform = np.linspace(0, coord_system.ly_physical, coord_system.ny)
    
    # Plot uniform grid lines
    for x in x_uniform:
        ax1.axvline(x*1e6, color='lightgray', linewidth=0.5)
    for y in y_uniform:
        ax1.axhline(y*1e6, color='lightgray', linewidth=0.5)
        
    # Plot conductors on uniform grid
    for geom in conductor_geometries:
        i_slice, j_slice = geom
        x1 = x_uniform[i_slice.start]
        x2 = x_uniform[min(i_slice.stop-1, len(x_uniform)-1)]
        y1 = y_uniform[j_slice.start]
        y2 = y_uniform[min(j_slice.stop-1, len(y_uniform)-1)]
        
        rect = patches.Rectangle(
            (x1*1e6, y1*1e6), 
            (x2-x1)*1e6, 
            (y2-y1)*1e6,
            linewidth=2,
            edgecolor='red',
            facecolor='lightcoral',
            alpha=0.3
        )
        ax1.add_patch(rect)
    
    # Non-uniform Grid (right plot)
    grid_gen = NonuniformGrid(coord_system, refinement_factor=4.0)
    x_new, y_new = grid_gen.generate_grid(conductor_geometries)
    
    # Plot non-uniform grid lines
    for x in x_new:
        ax2.axvline(x*1e6, color='lightgray', linewidth=0.5)
    for y in y_new:
        ax2.axhline(y*1e6, color='lightgray', linewidth=0.5)
        
    # Plot conductors on non-uniform grid
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
            alpha=0.3
        )
        ax2.add_patch(rect)
    
    # Set labels and titles
    for ax in [ax1, ax2]:
        ax.set_xlim(0, coord_system.lx_physical*1e6)
        ax.set_ylim(0, coord_system.ly_physical*1e6)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.grid(False)
    
    ax1.set_title('Uniform Grid')
    ax2.set_title('Non-uniform Grid')
    
    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# Example usage
lx = ly = 10e-6  # 10 µm domain
min_feature = 0.5e-6  # 0.5 µm minimum feature

# Create two conductors (in physical coordinates)
# Modified conductor positions for better visualization
conductor_1 = ((3e-6, 4e-6), (2e-6, 8e-6))  # Vertical conductor
conductor_2 = ((4e-6, 7e-6), (4e-6, 5e-6))  # Horizontal conductor
conductors = [conductor_1, conductor_2]

# Initialize system
coord_system = CoordinateSystem(50, 50, lx, ly, min_feature)  # Reduced grid points for clarity

# Convert physical coordinates to grid indices
grid_conductors = []
for (x1, x2), (y1, y2) in conductors:
    i1, j1 = coord_system.physical_to_grid(x1, y1)
    i2, j2 = coord_system.physical_to_grid(x2, y2)
    grid_conductors.append((slice(i1, i2+1), slice(j1, j2+1)))

# Visualize comparison
visualize_comparison(coord_system, grid_conductors)


