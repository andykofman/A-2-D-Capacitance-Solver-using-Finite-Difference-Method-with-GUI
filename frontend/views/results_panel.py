from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, 
    QTableWidget, QTableWidgetItem, QPushButton,
    QFileDialog, QMessageBox
)
import numpy as np
from frontend.utils.constants import COLORS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class ResultsPanel(QWidget):
    def __init__(self, grid_model):
        super().__init__()
        self.grid_model = grid_model
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_matrix_tab()
        self._create_charges_tab()
        
        # Add visualization button
        self.visualize_button = QPushButton("Show Field Visualization")
        self.visualize_button.clicked.connect(self._show_visualization)
        layout.addWidget(self.visualize_button)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._export_results)
        layout.addWidget(self.export_button)
        
        # Store the latest backend data
        self.latest_data = None
        
    def _create_matrix_tab(self):
        """Create capacitance matrix tab"""
        matrix_widget = QWidget()
        matrix_layout = QVBoxLayout(matrix_widget)
        
        # Add table for capacitance matrix
        self.matrix_table = QTableWidget()
        self.matrix_table.setStyleSheet(f"background-color: {COLORS['surface']};")
        matrix_layout.addWidget(self.matrix_table)
        
        # Add explanation label
        explanation = QLabel("Capacitance values in femtofarads (fF)")
        explanation.setStyleSheet(f"color: {COLORS['text']};")
        matrix_layout.addWidget(explanation)
        
        self.tabs.addTab(matrix_widget, "Capacitance Matrix")
        
    def _create_charges_tab(self):
        """Create charges display tab"""
        charges_widget = QWidget()
        charges_layout = QVBoxLayout(charges_widget)
        
        # Add table for charges
        self.charges_table = QTableWidget()
        self.charges_table.setStyleSheet(f"background-color: {COLORS['surface']};")
        charges_layout.addWidget(self.charges_table)
        
        # Add explanation label
        explanation = QLabel("Charges in femtocoulombs (fC)")
        explanation.setStyleSheet(f"color: {COLORS['text']};")
        charges_layout.addWidget(explanation)
        
        self.tabs.addTab(charges_widget, "Charges")

    def update_results(self, backend_data):
        """Update results using backend data"""
        if backend_data is None:
            print("Debug: No backend data to display")
            return
        
        try:
            print("Debug: Updating results panel with new data")
            
            # Store the latest data
            self.latest_data = backend_data
            
            # Update capacitance matrix
            C_matrix = backend_data.get('capacitance_matrix')
            if C_matrix is not None:
                print(f"Debug: Updating capacitance matrix of shape {C_matrix.shape}")
                self._update_matrix_display(C_matrix)
            else:
                print("Debug: No capacitance matrix available")
            
            # Update charges
            charges = backend_data.get('charges')
            if charges is not None:
                print(f"Debug: Updating charges of length {len(charges)}")
                self._update_charges_display(charges)
            else:
                print("Debug: No charges data available")
            
            # Enable visualization button if field data is available
            has_fields = all(backend_data.get(field) is not None 
                            for field in ['Ex', 'Ey', 'E_mag'])
            self.visualize_button.setEnabled(has_fields)
            
        except Exception as e:
            print(f"Error updating results panel: {e}")
            import traceback
            traceback.print_exc()

    def _update_matrix_display(self, C_matrix):
        """Update capacitance matrix display"""
        try:
            if C_matrix is not None:
                C_matrix_fF = C_matrix * 1e15  # Convert to fF
                n = C_matrix_fF.shape[0]
                
                # Configure table
                self.matrix_table.setRowCount(n)
                self.matrix_table.setColumnCount(n)
                
                # Fill table
                for i in range(n):
                    for j in range(n):
                        item = QTableWidgetItem(f"{C_matrix_fF[i,j]:.2f}")
                        self.matrix_table.setItem(i, j, item)
                
                # Set headers
                self.matrix_table.setHorizontalHeaderLabels([f"C{i+1}" for i in range(n)])
                self.matrix_table.setVerticalHeaderLabels([f"C{i+1}" for i in range(n)])
                
                print(f"Debug: Matrix display updated with {n}x{n} matrix")
                
        except Exception as e:
            print(f"Error updating matrix display: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_charges_display(self, charges):
        """Update charges display"""
        if charges is not None:
            charges_fc = charges * 1e15  # Convert to fC
            
            # Configure table
            self.charges_table.setRowCount(len(charges_fc))
            self.charges_table.setColumnCount(1)
            
            # Fill table
            for i, charge in enumerate(charges_fc):
                item = QTableWidgetItem(f"{charge:.2f}")
                self.charges_table.setItem(i, 0, item)
            
            # Set headers
            self.charges_table.setHorizontalHeaderLabels(["Charge (fC)"])
            self.charges_table.setVerticalHeaderLabels([f"Q{i+1}" for i in range(len(charges_fc))])
    
    def _show_visualization(self):
        """Show field visualization using backend method"""
        try:
            if hasattr(self.grid_model, 'controller') and self.grid_model.controller:
                # Get conductor potentials
                conductor_potentials = self.grid_model.get_conductor_potentials()
                
                # Use backend visualization
                if hasattr(self.grid_model.controller, 'capacitance_extractor'):
                    # Call backend visualization method
                    self.grid_model.controller.capacitance_extractor.visualize_fields(conductor_potentials)
                else:
                    raise ValueError("Capacitance extractor not initialized")
            else:
                raise ValueError("Controller not available")
            
        except Exception as e:
            print(f"Error showing visualization: {e}")
            QMessageBox.warning(self, "Visualization Error", 
                              f"Error displaying visualization: {str(e)}")
        
    def _export_results(self):
        """Export results to file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", 
                "CSV files (*.csv);;All Files (*)"
            )
            
            if filename:
                if self.grid_model.capacitance_matrix is not None and self.grid_model.charges is not None:
                    # Export Capacitance Matrix
                    C_matrix_fF = self.grid_model.capacitance_matrix * 1e15
                    np.savetxt(f"{filename}_capacitance.csv", C_matrix_fF, delimiter=',', 
                               header='Capacitance Matrix (fF)')
                    
                    # Export Charges
                    charges_fC = self.grid_model.charges * 1e15
                    np.savetxt(f"{filename}_charges.csv", charges_fC, delimiter=',', 
                               header='Charges (fC)')
                    
                    QMessageBox.information(self, "Success", 
                                         "Results exported successfully!")
                else:
                    QMessageBox.warning(self, "Export Error", 
                                      "No results available to export.")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", 
                              f"Error exporting results: {str(e)}")