from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, 
    QDoubleSpinBox, QLabel, QGroupBox,
    QScrollArea, QCheckBox, QHBoxLayout, QComboBox, QMessageBox, QPushButton
)
from PyQt5.QtCore import Qt
from ..utils.constants import DEFAULT_EPSILON_R, COLORS, DEFAULT_DIELECTRIC_PRESETS
import sys 
from PyQt5.QtWidgets import QPushButton
import numpy as np

class PropertyPanel(QScrollArea):
    def __init__(self, grid_model):
        super().__init__()
        self.grid_model = grid_model
        
        # Create main widget and layout
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setAlignment(Qt.AlignTop)

        #save current location
        self.save_button = QPushButton("Save Current Location")
        self.save_button.clicked.connect(self._on_save_current)
        self.layout.addWidget(self.save_button)
        # Create conductor properties group
        self.conductor_group = QGroupBox("Conductor Properties")
        self.conductor_layout = QFormLayout()
        self.conductor_group.setLayout(self.conductor_layout)
        
        # Create dielectric properties group
        self.dielectric_group = QGroupBox("Dielectric Properties")
        self.dielectric_layout = QFormLayout()
        self.dielectric_group.setLayout(self.dielectric_layout)
        
        # Add nonuniform grid toggle
        self.grid_group = QGroupBox("Grid Settings")
        self.grid_layout = QFormLayout()
        self.nonuniform_checkbox = QCheckBox("Use Nonuniform Grid")
        self.nonuniform_checkbox.stateChanged.connect(self._on_nonuniform_changed)
        self.grid_layout.addRow(self.nonuniform_checkbox)
        self.grid_group.setLayout(self.grid_layout)
        self.layout.addWidget(self.grid_group)
        
        # Add groups to main layout
        self.layout.addWidget(self.conductor_group)
        self.layout.addWidget(self.dielectric_group)
        
        # Set widget as scroll area content
        self.setWidget(self.widget)
        self.setWidgetResizable(True)

        # Add Calculate button
        self.calculate_button = QPushButton("Calculate Capacitance")
        self.calculate_button.clicked.connect(self._on_calculate)
        self.layout.addWidget(self.calculate_button)
        
        # Style
        self.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['grid']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                color: {COLORS['text']};
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }}
        """)
    
    def update_properties(self):
        """Update property panel with current elements"""
        # Clear existing layouts
        self._clear_layout(self.conductor_layout)
        self._clear_layout(self.dielectric_layout)
        
        # Add conductor properties
        for conductor in self.grid_model.conductors:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(conductor.potential)
            spinbox.valueChanged.connect(
                lambda v, c=conductor: self._on_potential_changed(c, v)
            )
            self.conductor_layout.addRow(f"{conductor.name} (V):", spinbox)
        
        # Add dielectric properties with preset combo box
        for region in self.grid_model.dielectric_regions:
            # Create horizontal layout for dielectric controls
            control_layout = QHBoxLayout()
            
            # Add spinbox for manual value
            spinbox = QDoubleSpinBox()
            spinbox.setRange(1, 1000)
            spinbox.setDecimals(3)
            spinbox.setValue(region.epsilon_r)
            spinbox.valueChanged.connect(
                lambda v, r=region: self._on_epsilon_changed(r, v)
            )
            
            # Add preset combo box
            preset_combo = QComboBox()
            preset_combo.addItems(['Custom'] + list(DEFAULT_DIELECTRIC_PRESETS.keys()))
            preset_combo.currentTextChanged.connect(
                lambda t, r=region, s=spinbox: self._on_preset_changed(r, t, s)
            )
            
            control_layout.addWidget(preset_combo)
            control_layout.addWidget(spinbox)
            
            self.dielectric_layout.addRow(f"{region.name}:", control_layout)
    
    def _on_potential_changed(self, conductor, value):
        """Handle conductor potential change"""
        conductor.potential = value
        self.grid_model.update_epsilon_grid()
    
    def _on_epsilon_changed(self, region, value):
        """Handle dielectric constant change"""
        region.epsilon_r = value
        self.grid_model.update_epsilon_grid()
    
    def _on_preset_changed(self, region, preset_name, spinbox):
        """Handle dielectric preset selection"""
        if preset_name != 'Custom':
            epsilon_r = DEFAULT_DIELECTRIC_PRESETS[preset_name]
            spinbox.setValue(epsilon_r)
            region.epsilon_r = epsilon_r
            self.grid_model.update_epsilon_grid()
    
    def _clear_layout(self, layout):
        """Clear all widgets from layout"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_save_current(self):
        """Save current grid state to backend"""
        try:
            print("Debug: Saving current grid state to backend")
            
            # Get grid properties
            grid_properties = {
                'nx': self.grid_model.nx,
                'ny': self.grid_model.ny,
                'lx': self.grid_model.lx,
                'ly': self.grid_model.ly
            }
            print(f"Debug: Grid properties: {grid_properties}")

            # Get conductors
            conductors = []
            for conductor in self.grid_model.conductors:
                conductors.append({
                    'geometry': conductor.geometry,
                    'potential': conductor.potential
                })
            print(f"Debug: Found {len(conductors)} conductors")

            # Get dielectrics
            dielectrics = []
            for dielectric in self.grid_model.dielectric_regions:
                dielectrics.append({
                    'geometry': dielectric.geometry,
                    'epsilon_r': dielectric.epsilon_r
                })
            print(f"Debug: Found {len(dielectrics)} dielectric regions")

            # Save the current state without calculation
            self.grid_model.saved_state = {
                'grid_properties': grid_properties,
                'conductors': conductors,
                'dielectrics': dielectrics
            }
            print("Debug: Current state saved successfully")
                
        except Exception as e:
            print(f"Error saving current state: {e}")

    def _on_nonuniform_changed(self, state):
        """Handle nonuniform grid toggle"""
        self.grid_model.use_nonuniform_grid = bool(state)
        
        # Only update grid visualization if there are elements
        if self.grid_model.conductors or self.grid_model.dielectric_regions:
            self.grid_model.update_grid_discretization()
            # Force redraw
            if hasattr(self.grid_model, 'grid_drawer'):
                self.grid_model.grid_drawer.update()


    def _on_calculate(self):
        """Calculate capacitance for the saved configuration"""
        try:
            if hasattr(self.grid_model, 'saved_state'):
                # Use the saved state for calculation
                state = self.grid_model.saved_state
                result = self.grid_model.controller.update_backend(
                    grid_properties=state['grid_properties'],
                    conductors=state['conductors'],
                    dielectrics=state['dielectrics']
                )
                
                if result is not None:
                    potential_dist, fields = result
                    if potential_dist is not None:
                        self.grid_model.store_solution(potential_dist, fields)
                        print("Debug: Capacitance calculation completed")
                        self.grid_model.property_panel.update_properties()
                        self.grid_model.results_panel.update_results()
                    else:
                        print("Debug: No solution available yet")
                else:
                    print("Debug: Backend update failed")
            else:
                print("Debug: No saved state to calculate from")
            
            if self.grid_model.controller.calculate_capacitance():
                print("Debug: Calculation completed successfully")
            else:
                print("Debug: Calculation failed")
        except Exception as e:
            print(f"Capacitance calculation failed: {e}")

    def show_small_feature_dialog(self, recommended_size):
        """Show dialog for small feature options"""
        from PyQt5.QtWidgets import QMessageBox, QPushButton
        
        dialog = QMessageBox()
        dialog.setWindowTitle("Small Feature Detected")
        dialog.setText("A small feature has been detected. Choose an option:")
        
        option1 = QPushButton("Keep Grid the Same")
        option2 = QPushButton("Increment 100 Points")
        option3 = QPushButton(f"Set to Recommended Size: {recommended_size[0]} x {recommended_size[1]}")
        
        dialog.addButton(option1, QMessageBox.ActionRole)
        dialog.addButton(option2, QMessageBox.ActionRole)
        dialog.addButton(option3, QMessageBox.ActionRole)
        
        dialog.exec_()
        
        if dialog.clickedButton() == option1:
            self.add_red_annotation("Current grid is not sufficient.")
            self.add_increase_button()
            self.grid_model.controller.update_backend({
                'nx': self.grid_model.nx,
                'ny': self.grid_model.ny,
                'lx': self.grid_model.lx,
                'ly': self.grid_model.ly
            }, self.grid_model.conductors, self.grid_model.dielectric_regions, force_grid_size=False)
        elif dialog.clickedButton() == option2:
            new_nx = self.grid_model.nx + 100
            new_ny = self.grid_model.ny + 100
            self.grid_model.resize_grid(new_nx, new_ny)
            self.grid_model.controller.update_backend({
                'nx': new_nx,
                'ny': new_ny,
                'lx': self.grid_model.lx,
                'ly': self.grid_model.ly
            }, self.grid_model.conductors, self.grid_model.dielectric_regions, force_grid_size=True)
        elif dialog.clickedButton() == option3:
            self.grid_model.resize_grid(recommended_size[0], recommended_size[1])
            self.grid_model.controller.update_backend({
                'nx': recommended_size[0],
                'ny': recommended_size[1],
                'lx': self.grid_model.lx,
                'ly': self.grid_model.ly
            }, self.grid_model.conductors, self.grid_model.dielectric_regions, force_grid_size=True)
        
    def add_red_annotation(self, message):
        """Add a red annotation to the property panel"""
        # Implementation for adding a red annotation
        pass
    
    def add_increase_button(self):
        """Add button to increase grid size by 100 points"""
        # Implementation for adding the button
        pass