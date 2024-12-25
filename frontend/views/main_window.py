from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QAction, QDockWidget, QLabel, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from ..utils.constants import WINDOW_DEFAULT_SIZE, DrawingMode
from .grid_drawer import GridDrawer
from ..models.property_panel import PropertyPanel
from .results_panel import ResultsPanel
import sys 

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        # Set window properties
        self.setWindowTitle("Capacitance Solver")
        self.resize(*WINDOW_DEFAULT_SIZE)
        
        # Initialize UI components
        self._init_ui()
        self._create_actions()
        self._create_toolbars()
        self._create_dock_widgets()
        self._create_status_bar()
        
    def _init_ui(self):
        """Initialize main UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        layout = QVBoxLayout(central_widget)
        
        # Create grid drawer
        self.grid_drawer = GridDrawer(self.controller.grid_model)
        layout.addWidget(self.grid_drawer)
    
    def _create_actions(self):
        """Create application actions"""
        # File actions
        self.new_action = QAction("New", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self._on_new)
        
        self.open_action = QAction("Open", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._on_open)
        
        self.save_action = QAction("Save", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._on_save)
        
        # Edit actions
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self._on_undo)
        
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self._on_redo)

        #Delete action
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut("Delete")
        self.delete_action.triggered.connect(self._on_delete)
    
    def _create_toolbars(self):
        """Create toolbars"""
        # File toolbar
        file_toolbar = QToolBar("File")
        self.addToolBar(file_toolbar)
        file_toolbar.addAction(self.new_action)
        file_toolbar.addAction(self.open_action)
        file_toolbar.addAction(self.save_action)
        
        # Edit toolbar
        edit_toolbar = QToolBar("Edit")
        self.addToolBar(edit_toolbar)
        edit_toolbar.addAction(self.undo_action)
        edit_toolbar.addAction(self.redo_action)
        edit_toolbar.addAction(self.delete_action)
        
        # Drawing toolbar
        drawing_toolbar = QToolBar("Drawing")
        self.addToolBar(drawing_toolbar)
        
        # Select mode
        self.select_action = QAction("Select", self)
        self.select_action.setShortcut('S')
        self.select_action.setStatusTip('Select and modify elements')
        self.select_action.setCheckable(True)
        self.select_action.setChecked(True)
        self.select_action.triggered.connect(self._on_select_mode)
        drawing_toolbar.addAction(self.select_action)
        
        # Conductor mode
        self.conductor_action = QAction("Draw Conductor", self)
        self.conductor_action.setShortcut('C')
        self.conductor_action.setStatusTip('Draw conductors')
        self.conductor_action.setCheckable(True)
        self.conductor_action.triggered.connect(self._on_conductor_mode)
        drawing_toolbar.addAction(self.conductor_action)
        
        # Dielectric mode
        self.dielectric_action = QAction("Draw Dielectric", self)
        self.dielectric_action.setShortcut('D')
        self.dielectric_action.setStatusTip('Draw dielectric regions')
        self.dielectric_action.setCheckable(True)
        self.dielectric_action.triggered.connect(self._on_dielectric_mode)
        drawing_toolbar.addAction(self.dielectric_action)


        # Add test case button
        test_toolbar = QToolBar("Test")
        self.addToolBar(test_toolbar)
        
        self.test_action = QAction("Run Test Case", self)
        self.test_action.setStatusTip('Run predefined test case')
        self.test_action.triggered.connect(self._on_test_case)
        test_toolbar.addAction(self.test_action)
    
    def _create_dock_widgets(self):
        """Create dock widgets"""
        # Properties dock
        properties_dock = QDockWidget("Properties", self)
        properties_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create and set property panel
        self.property_panel = PropertyPanel(self.controller.grid_model)
        self.controller.grid_model.property_panel = self.property_panel
        properties_dock.setWidget(self.property_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, properties_dock)
       
        # Results dock
        results_dock = QDockWidget("Results", self)
        results_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.results_panel = ResultsPanel(self.controller.grid_model)
        # Set the results panel reference in the grid model
        self.controller.grid_model.results_panel = self.results_panel
        results_dock.setWidget(self.results_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, results_dock)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    # Action handlers
    def _on_new(self):
        """Reset everything for a new project"""
        try:
            # Reset the grid model
            self.controller.grid_model.reset()
            
            # Clear the property panel
            self.property_panel.update_properties()
            
            # Clear the results panel
            self.results_panel.update_results(None)
            
            self.statusBar.showMessage("New project created")
        except Exception as e:
            print(f"Error creating new project: {e}")
            QMessageBox.warning(self, "Error", "Failed to create a new project.")
    
    def _on_open(self):
        self.statusBar.showMessage("Open project")
    
    def _on_save(self):
        self.statusBar.showMessage("Save project")
    
    def _on_undo(self):
        self.statusBar.showMessage("Undo")
    
    def _on_redo(self):
        self.statusBar.showMessage("Redo")


    def _on_select_mode(self):
        """Switch to select mode"""
        self._update_drawing_mode(DrawingMode.SELECT)
        self.statusBar.showMessage("Select Mode")

    def _on_delete(self):
        """Delete selected element"""
        if self.grid_drawer.selected_conductor:
            self.controller.grid_model.remove_conductor(self.grid_drawer.selected_conductor)
            self.grid_drawer.selected_conductor = None
            self.statusBar.showMessage("Conductor deleted")
        elif self.grid_drawer.selected_dielectric:
            self.controller.grid_model.remove_dielectric(self.grid_drawer.selected_dielectric)
            self.grid_drawer.selected_dielectric = None
            self.statusBar.showMessage("Dielectric region deleted")
        self.grid_drawer.update()

    def _on_conductor_mode(self):
        """Switch to conductor drawing mode"""
        self._update_drawing_mode(DrawingMode.CONDUCTOR)
        self.statusBar.showMessage("Conductor Drawing Mode")

    def _on_dielectric_mode(self):
        """Switch to dielectric drawing mode"""
        self._update_drawing_mode(DrawingMode.DIELECTRIC)
        self.statusBar.showMessage("Dielectric Drawing Mode")

    def _update_drawing_mode(self, mode: DrawingMode):
        """Update drawing mode and toolbar state"""
        self.select_action.setChecked(mode == DrawingMode.SELECT)
        self.conductor_action.setChecked(mode == DrawingMode.CONDUCTOR)
        self.dielectric_action.setChecked(mode == DrawingMode.DIELECTRIC)
        self.grid_drawer.set_drawing_mode(mode)

    def _on_test_case(self):
        """Run test case"""
        self.controller.run_test_case()
        self.statusBar.showMessage("Test case completed")