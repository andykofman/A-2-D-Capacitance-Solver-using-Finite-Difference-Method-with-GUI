from enum import Enum
from PyQt5.QtCore import Qt

class DrawingMode(Enum):
    """Enum for different drawing modes"""
    SELECT = 0
    CONDUCTOR = 1
    DIELECTRIC = 2

# Grid Constants
GRID_DEFAULT_SIZE = (300, 300)  # Initial grid size, will be adjusted based on geometry
GRID_MIN_SIZE = 50
GRID_MAX_SIZE = 1000

# Physical Constants
DEFAULT_DIMENSIONS = (100e-6, 100e-6)  # 100µm x 100µm (renamed from DEFAULT_DOMAIN_SIZE)
MIN_FEATURE_SIZE = 10e-6  # Default 1µm minimum feature

# UI Constants
WINDOW_DEFAULT_SIZE = (1200, 800)
GRID_VIEW_MIN_SIZE = (400, 400)

# Material Properties
DEFAULT_EPSILON_R = 1.0  # Air default
DEFAULT_DIELECTRIC_PRESETS = {
    'Air': 1.0,
    'SiO2': 3.9,
    'Si3N4': 7.5,
    'Al2O3': 9.0
}

# Colors
CONDUCTOR_COLOR = '#1976D2'
DIELECTRIC_COLOR = '#4CAF50'
GRID_COLOR = '#757575'
BACKGROUND_COLOR = '#FFFFFF'

# Keyboard Shortcuts
SHORTCUTS = {
    'new': Qt.CTRL + Qt.Key_N,
    'open': Qt.CTRL + Qt.Key_O,
    'save': Qt.CTRL + Qt.Key_S,
    'undo': Qt.CTRL + Qt.Key_Z,
    'redo': Qt.CTRL + Qt.Key_Y,
    'delete': Qt.Key_Delete,
    'escape': Qt.Key_Escape
}


# Modern Material Design Color Palette
COLORS = {
    'primary': '#2196F3',      # Blue
    'secondary': '#FF4081',    # Pink
    'success': '#4CAF50',      # Green
    'warning': '#FFC107',      # Amber
    'error': '#F44336',        # Red
    'surface': '#FFFFFF',      # White
    'background': '#FAFAFA',   # Light Grey
    'grid': '#E0E0E0',        # Medium Grey
    'text': '#212121',        # Dark Grey
    
    # Element Colors
    'conductor': {
        'fill': '#1976D2',     # Darker Blue
        'stroke': '#2196F3',   # Blue
        'preview': '#64B5F6'   # Light Blue
    },
    'dielectric': {
        'fill': '#43A047',     # Darker Green
        'stroke': '#4CAF50',   # Green
        'preview': '#81C784'   # Light Green
    }
}

# UI Theme
THEME = {
    'shadow': '0 2px 4px rgba(0,0,0,0.1)',
    'border_radius': 4,
    'spacing': 8,
    'animation_duration': 200  # ms
}