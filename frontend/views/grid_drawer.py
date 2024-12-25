from PyQt5.QtWidgets import QOpenGLWidget, QPushButton
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import numpy as np

from frontend.utils.constants import (
    GRID_COLOR, 
    CONDUCTOR_COLOR, 
    DIELECTRIC_COLOR,
    GRID_VIEW_MIN_SIZE, 
    DrawingMode,
    DEFAULT_EPSILON_R,
    COLORS
)

class GridDrawer(QOpenGLWidget):
    """OpenGL-based grid visualization widget"""
    
    def __init__(self, grid_model):
        super().__init__()
        self.grid_model = grid_model
        
        # Set widget properties
        self.setMinimumSize(*GRID_VIEW_MIN_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Drawing state
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.last_pos = None
        self.is_panning = False
        # Drawing mode
        self.drawing_mode = DrawingMode.SELECT
        self.drawing_start = None
        self.current_conductor = None
        self.current_dielectric = None
        
        # Selection state
        self.selected_conductor = None
        self.selected_dielectric = None
        
    def _hit_test(self, pos):
        """Test if position hits any element"""
        x = (pos.x() - self.offset.x()) / self.scale_factor / self.width() * self.grid_model.lx
        y = (pos.y() - self.offset.y()) / self.scale_factor / self.height() * self.grid_model.ly
        
        # Check conductors first (they're on top)
        for conductor in self.grid_model.conductors:
            (x1, x2), (y1, y2) = conductor.geometry
            if x1 <= x <= x2 and y1 <= y <= y2:
                return ('conductor', conductor)
        
        # Check dielectrics
        for region in self.grid_model.dielectric_regions:
            (x1, x2), (y1, y2) = region.geometry
            if x1 <= x <= x2 and y1 <= y <= y2:
                return ('dielectric', region)
        
        return None
    def initializeGL(self):
        """Initialize OpenGL context"""
        # Set background color
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Light gray background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    
    def resizeGL(self, width, height):
        """Handle resize events"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)  # Maintain aspect ratio
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Paint the grid and elements"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # Set background color to dark black
        self.setStyleSheet("background-color:  #000000;")
        
        # Switch to QPainter for 2D drawing
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Apply transformations
        painter.translate(self.offset)
        painter.scale(self.scale_factor, self.scale_factor)
        
        # Draw components
        self._draw_grid(painter)
        self._draw_dielectrics(painter)
        self._draw_conductors(painter)

        #Draw Conductor/Dielectric preview during drawing
        if self.current_conductor is not None:
            self._draw_conductor_preview(painter)
        elif self.current_dielectric is not None:
            self._draw_dielectric_preview(painter)
        
        painter.end()

    def _draw_dielectric_preview(self, painter):
        """Draw Dielectric preview during drawing"""
        pen = QPen(QColor(DIELECTRIC_COLOR))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        painter.setPen(pen) 
        painter.setBrush(QColor(DIELECTRIC_COLOR).lighter(150)) 

        (x1, x2), (y1, y2) = self.current_dielectric
        x1_px = int(x1 / self.grid_model.lx * self.width())
        x2_px = int(x2 / self.grid_model.lx * self.width())
        y1_px = int(y1 / self.grid_model.ly * self.height())
        y2_px = int(y2 / self.grid_model.ly * self.height())
        
        painter.drawRect(x1_px, y1_px, x2_px - x1_px, y2_px - y1_px)    

    def _draw_conductor_preview(self, painter):
        "Draw Conductor preview during drawing"

        pen = QPen(QColor(CONDUCTOR_COLOR))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(CONDUCTOR_COLOR).lighter(150))


        (x1, x2), (y1, y2) = self.current_conductor
        x1_px = int(x1 / self.grid_model.lx * self.width())
        x2_px = int(x2 / self.grid_model.lx * self.width())
        y1_px = int(y1 / self.grid_model.ly * self.height())
        y2_px = int(y2 / self.grid_model.ly * self.height())
        
        painter.drawRect(x1_px, y1_px, x2_px - x1_px, y2_px - y1_px)
   ###############################################################################
   ###############################################################################
    def _draw_grid(self, painter):
        """Draw background grid"""
        if self.grid_model.use_nonuniform_grid and self.grid_model.coord_system:
            self._draw_nonuniform_grid(painter)
        else:
            self._draw_uniform_grid(painter)

    def _draw_nonuniform_grid(self, painter):
        """Draw non-uniform background grid with density visualization"""
        width = self.width()
        height = self.height()
        
        # Get physical coordinates
        x_coords = self.grid_model.coord_system.x_physical
        y_coords = self.grid_model.coord_system.y_physical
        
        # Calculate grid spacing and density
        x_spacing = np.diff(x_coords)
        y_spacing = np.diff(y_coords)
        
        # Normalize spacings (inverse relationship with density)
        x_density = 1 / (x_spacing / np.min(x_spacing))
        y_density = 1 / (y_spacing / np.min(y_spacing))
        
        # Draw vertical lines with density-based appearance
        for i, x in enumerate(x_coords[:-1]):
            x_px = int(x / self.grid_model.lx * width)
            next_x_px = int(x_coords[i+1] / self.grid_model.lx * width)
            
            # Calculate color based on density
            alpha = int(max(40, min(255, x_density[i] * 200)))
            color = QColor(COLORS['grid'])
            color.setAlpha(alpha)
            
            # Calculate line width based on density
            line_width = max(1, min(3, x_density[i] * 2))
            
            pen = QPen(color)
            pen.setWidth(int(line_width))
            painter.setPen(pen)
            painter.drawLine(x_px, 0, x_px, height)
            
            # Draw density gradient between lines
            if i < len(x_coords) - 2:
                gradient = QLinearGradient(x_px, 0, next_x_px, 0)
                start_color = QColor(COLORS['grid'])
                start_color.setAlpha(int(alpha * 0.3))
                end_color = QColor(COLORS['grid'])
                end_color.setAlpha(0)
                gradient.setColorAt(0, start_color)
                gradient.setColorAt(1, end_color)
                painter.fillRect(x_px, 0, next_x_px - x_px, height, gradient)
        
        # Draw horizontal lines with density-based appearance
        for i, y in enumerate(y_coords[:-1]):
            y_px = int(y / self.grid_model.ly * height)
            next_y_px = int(y_coords[i+1] / self.grid_model.ly * height)
            
            # Calculate color based on density
            alpha = int(max(40, min(255, y_density[i] * 200)))
            color = QColor(COLORS['grid'])
            color.setAlpha(alpha)
            
            # Calculate line width based on density
            line_width = max(1, min(3, y_density[i] * 2))
            
            pen = QPen(color)
            pen.setWidth(int(line_width))
            painter.setPen(pen)
            painter.drawLine(0, y_px, width, y_px)
            
            # Draw density gradient between lines
            if i < len(y_coords) - 2:
                gradient = QLinearGradient(0, y_px, 0, next_y_px)
                start_color = QColor(COLORS['grid'])
                start_color.setAlpha(int(alpha * 0.3))
                end_color = QColor(COLORS['grid'])
                end_color.setAlpha(0)
                gradient.setColorAt(0, start_color)
                gradient.setColorAt(1, end_color)
                painter.fillRect(0, y_px, width, next_y_px - y_px, gradient)

    def _draw_uniform_grid(self, painter):
        """Draw uniform grid with equal cell sizes"""
        width = self.width()
        height = self.height()
        
        # Calculate cell size
        dx = self.grid_model.lx / self.grid_model.nx
        dy = self.grid_model.ly / self.grid_model.ny
        
        # Draw vertical lines
        for i in range(self.grid_model.nx + 1):
            x = int(i * dx / self.grid_model.lx * width)  # Convert to int
            pen = QPen(QColor(255, 250, 235, 255))  # Light gray color
            pen.setWidth(1)  # Set a width for better visibility
            painter.setPen(pen)
            painter.drawLine(x, 0, x, height)

        # Draw horizontal lines
        for j in range(self.grid_model.ny + 1):
            y = int(j * dy / self.grid_model.ly * height)  # Convert to int
            pen = QPen(QColor(255, 250, 235, 255))  # Light gray color
            pen.setWidth(1)  # Set a width for better visibility
            painter.setPen(pen)
            painter.drawLine(0, y, width, y)

        
######################################################################################
    def _draw_conductors(self, painter):
        """Draw conductors"""
        for conductor in self.grid_model.conductors:
            is_selected = conductor == self.selected_conductor
            self._draw_single_conductor(painter, conductor, is_selected)

    def _draw_single_conductor(self, painter, conductor, is_selected):
        (x1, x2), (y1, y2) = conductor.geometry
        x1_px = int(x1 / self.grid_model.lx * self.width())
        x2_px = int(x2 / self.grid_model.lx * self.width())
        y1_px = int(y1 / self.grid_model.ly * self.height())
        y2_px = int(y2 / self.grid_model.ly * self.height())
        
        # Determine color based on potential
        color = QColor(255, 0, 0) if conductor.potential != 0 else QColor(0, 0, 255)  # Red or Gray
        
        # Draw selection highlight
        if is_selected:
            highlight_pen = QPen(QColor(COLORS['secondary']))
            highlight_pen.setWidth(3)
            highlight_pen.setStyle(Qt.DashLine)
            painter.setPen(highlight_pen)
            painter.drawRect(x1_px-2, y1_px-2, x2_px-x1_px+4, y2_px-y1_px+4)
        
        # Draw conductor with existing effects
        gradient = QLinearGradient(x1_px, y1_px, x1_px, y2_px)
        gradient.setColorAt(0, color)  # Use determined color
        gradient.setColorAt(1, color.darker(110))
        
        painter.setPen(QPen(color, 2))  # Use determined color for the pen
        painter.setBrush(gradient)
        painter.drawRect(x1_px, y1_px, x2_px-x1_px, y2_px-y1_px)
        
        # Annotate dimensions
        length = abs(x2 - x1) * 1e6  # Convert to micrometers
        width = abs(y2 - y1) * 1e6  # Convert to micrometers
        dimensions_text = f"{length:.1f} µm x {width:.1f} µm"
        painter.setPen(QColor(COLORS['surface']))  # Set text color
        # Convert positions to integers for drawText
        text_x = int((x1_px + x2_px) / 2)
        text_y = int((y1_px + y2_px) / 2)
        painter.drawText(text_x, text_y, dimensions_text)  # Centered text
    
    def _draw_dielectrics(self, painter):
        """Draw dielectric regions"""
        for region in self.grid_model.dielectric_regions:
            is_selected = region == self.selected_dielectric
            self._draw_single_dielectric(painter, region, is_selected)

    def _draw_single_dielectric(self, painter, region, is_selected):
        (x1, x2), (y1, y2) = region.geometry
        # Convert physical coordinates to widget coordinates
        x1_px = int(x1 / self.grid_model.lx * self.width())
        x2_px = int(x2 / self.grid_model.lx * self.width())
        y1_px = int(y1 / self.grid_model.ly * self.height())
        y2_px = int(y2 / self.grid_model.ly * self.height())
        
        # Draw selection highlight
        if is_selected:
            highlight_pen = QPen(QColor(COLORS['secondary']))
            highlight_pen.setWidth(3)
            highlight_pen.setStyle(Qt.DashLine)
            painter.setPen(highlight_pen)
            painter.drawRect(x1_px-2, y1_px-2, x2_px-x1_px+4, y2_px-y1_px+4)
        
        # Draw dielectric with gradient effect
        gradient = QLinearGradient(x1_px, y1_px, x1_px, y2_px)
        gradient.setColorAt(0, QColor(COLORS['dielectric']['fill']))
        gradient.setColorAt(1, QColor(COLORS['dielectric']['fill']).darker(110))
        
        painter.setPen(QPen(QColor(COLORS['dielectric']['stroke']), 2))
        painter.setBrush(gradient)
        painter.drawRect(x1_px, y1_px, x2_px-x1_px, y2_px-y1_px)

######################################################################################
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:

            self.ripple_pos = event.pos()
            self.ripple_animation = 0
            self.ripple_timer = self.startTimer(16)  # 60 FPS
            
            self.is_panning = True
            self.last_pos = event.pos()
    
        elif event.button() == Qt.LeftButton:
            if self.drawing_mode == DrawingMode.SELECT:
                hit = self._hit_test(event.pos())
                if hit:
                    element_type, element = hit
                    if element_type == 'conductor':
                        self.selected_conductor = element
                        self.selected_dielectric = None
                    else:
                        self.selected_conductor = None
                        self.selected_dielectric = element
                    self.grid_model.notify_property_panel()
                else:
                    self.selected_conductor = None
                    self.selected_dielectric = None
                    self.grid_model.notify_property_panel()
            else:
                # Existing drawing code...
                pos = event.pos()
                x = (pos.x() - self.offset.x()) / self.scale_factor / self.width() * self.grid_model.lx
                y = (pos.y() - self.offset.y()) / self.scale_factor / self.height() * self.grid_model.ly
                
                if self.drawing_mode == DrawingMode.CONDUCTOR:
                    self.drawing_start = (x, y)
                elif self.drawing_mode == DrawingMode.DIELECTRIC:
                    self.drawing_start = (x, y)
        
        self.update()
        event.accept()
    

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
        elif event.button() == Qt.LeftButton:
            if self.drawing_mode == DrawingMode.CONDUCTOR and self.current_conductor:
                result = self.grid_model.add_conductor(self.current_conductor)
                self.drawing_start = None
                self.current_conductor = None
                # Check for small features after drawing
                self.grid_model.check_small_feature()
            elif self.drawing_mode == DrawingMode.DIELECTRIC and self.current_dielectric:
                result = self.grid_model.add_dielectric(self.current_dielectric, DEFAULT_EPSILON_R)
                self.drawing_start = None
                self.current_dielectric = None
                # Check for small features after drawing
                self.grid_model.check_small_feature()
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        
        if self.is_panning and self.last_pos is not None:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.update()
        elif self.drawing_start is not None:
            pos = event.pos()
            x = (pos.x() - self.offset.x()) / self.scale_factor / self.width() * self.grid_model.lx
            y = (pos.y() - self.offset.y()) / self.scale_factor / self.height() * self.grid_model.ly
            
            if self.drawing_mode == DrawingMode.CONDUCTOR:
                self.current_conductor = (
                    (min(self.drawing_start[0], x), max(self.drawing_start[0], x)),
                    (min(self.drawing_start[1], y), max(self.drawing_start[1], y))
                )
            elif self.drawing_mode == DrawingMode.DIELECTRIC:
                self.current_dielectric = (
                    (min(self.drawing_start[0], x), max(self.drawing_start[0], x)),
                    (min(self.drawing_start[1], y), max(self.drawing_start[1], y))
                )
            self.update()
        event.accept()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        self.update()
        event.accept()
######################################################################################
    def timerEvent(self, event):
        """Handle ripple animation"""
        self.ripple_animation += 1
        if self.ripple_animation >= 15:  # ~250ms
            self.killTimer(self.ripple_timer)
            self.ripple_animation = 0
        self.update()
        
    def _draw_ripple(self, painter):
        """Draw ripple effect"""
        if hasattr(self, 'ripple_pos') and self.ripple_animation > 0:
            painter.save()
            color = QColor(COLORS['primary'])
            color.setAlpha(int(255 * (1 - self.ripple_animation / 15)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            radius = self.ripple_animation * 10
            painter.drawEllipse(self.ripple_pos, radius, radius)
            painter.restore()
####################################################################################
    def set_drawing_mode(self, mode: DrawingMode):
        """Set the current drawing mode for the grid drawer"""
        self.drawing_mode = mode
        self.update()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key_Delete:
            if self.selected_conductor:
                self.grid_model.remove_conductor(self.selected_conductor)
                self.selected_conductor = None
            elif self.selected_dielectric:
                self.grid_model.remove_dielectric(self.selected_dielectric)
                self.selected_dielectric = None
            self.update()
        event.accept()



