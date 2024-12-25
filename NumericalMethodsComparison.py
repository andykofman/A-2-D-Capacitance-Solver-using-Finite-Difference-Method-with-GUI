from manim import *
import numpy as np
from scipy.spatial import Delaunay

class EnhancedNumericalMethodsComparison(Scene):
    def construct(self):
        # Configuration
        SQUARE_SIZE = 4
        BOUNDARY_COLOR = "#2C3E50"
        
        # Enhanced square conductor with better styling
        square = Square(side_length=SQUARE_SIZE)
        square.set_stroke(color=BOUNDARY_COLOR, width=3)
        
        # Title with better typography
        title = Text("Numerical Methods in Electromagnetic Analysis", 
                    font="Sans Serif",
                    gradient=(BLUE, GREEN))
        title.scale(0.8)
        title.to_edge(UP, buff=0.5)

        def create_enhanced_fem_mesh():
            # Generate more natural point distribution
            n_points = 50  # interior points
            n_boundary = 20  # points per boundary edge
            
            # Interior points with controlled random distribution
            rng = np.random.default_rng(42)  # for reproducibility
            interior_points = []
            
            # Generate points using rejection sampling for better distribution
            while len(interior_points) < n_points:
                x = rng.uniform(-1.9, 1.9)
                y = rng.uniform(-1.9, 1.9)
                
                # Add some structure to the randomness
                if rng.random() < 0.7:  # 70% chance of structured points
                    x = round(x * 4) / 4
                    y = round(y * 4) / 4
                
                interior_points.append([x, y])
            
            # Boundary points (more densely packed near corners)
            boundary_points = []
            for i in range(n_boundary):
                t = i / (n_boundary - 1)
                # Add non-linear distribution for better corner resolution
                t = t ** 1.5 if t < 0.5 else 1 - (1 - t) ** 1.5
                
                boundary_points.extend([
                    [-2, -2 + 4 * t],  # left edge
                    [2, -2 + 4 * t],   # right edge
                    [-2 + 4 * t, -2],  # bottom edge
                    [-2 + 4 * t, 2]    # top edge
                ])
            
            # Combine points and create Delaunay triangulation
            all_points = np.array(interior_points + boundary_points)
            tri = Delaunay(all_points)
            
            # Create mesh visualization
            triangles = VGroup()
            for simplex in tri.simplices:
                # Convert 2D points to 3D by adding z=0 coordinate
                triangle_points = [np.append(all_points[i], [0]) for i in simplex]
                triangle = Polygon(
                    *triangle_points,
                    stroke_width=1,
                    stroke_color=BLUE_A,
                    fill_color=BLUE_E,
                    fill_opacity=0.3
                )
                triangles.add(triangle)
            
            # Add nodes (also need to convert to 3D points)
            nodes = VGroup(*[
                Dot(point=np.append(point, [0]), radius=0.03, color=BLUE_A)
                for point in all_points
            ])
            
            return VGroup(triangles, nodes)

        def create_enhanced_fdm_grid():
            lines = VGroup()
            
            # Use uniform spacing instead of refined positions
            positions = np.linspace(-2, 2, 15)
            
            # Create grid lines
            for x in positions:
                # Vertical lines
                line = Line(
                    start=[x, -2, 0],
                    end=[x, 2, 0],
                    stroke_width=2 if abs(x) > 1.9 else 1,  # thicker boundary lines
                    color=GREEN_A
                )
                lines.add(line)
                
                # Horizontal lines
                line = Line(
                    start=[-2, x, 0],
                    end=[2, x, 0],
                    stroke_width=2 if abs(x) > 1.9 else 1,  # thicker boundary lines
                    color=GREEN_A
                )
                lines.add(line)
            
            # Add intersection points
            points = VGroup(*[
                Dot(point=[x, y, 0], radius=0.02, color=GREEN)
                for x in positions
                for y in positions
            ])
            
            return VGroup(lines, points)

        def create_enhanced_bem_points():
            # Create boundary elements with varying density
            boundary_elements = VGroup()
            
            # Generate non-uniform distribution of points
            def get_boundary_points(n=40):
                t = np.linspace(0, 1, n)
                # Apply non-linear transformation for corner refinement
                t = np.power(t, 1.3)
                return t
            
            t = get_boundary_points()
            
            # Create boundary elements with size variation
            for i in range(len(t)-1):
                # Bottom edge
                element = Line(
                    start=[-2 + 4*t[i], -2, 0],
                    end=[-2 + 4*t[i+1], -2, 0],
                    stroke_width=3,
                    color=RED_A
                )
                boundary_elements.add(element)
                
                # Top edge
                element = Line(
                    start=[-2 + 4*t[i], 2, 0],
                    end=[-2 + 4*t[i+1], 2, 0],
                    stroke_width=3,
                    color=RED_A
                )
                boundary_elements.add(element)
                
                # Left edge
                element = Line(
                    start=[-2, -2 + 4*t[i], 0],
                    end=[-2, -2 + 4*t[i+1], 0],
                    stroke_width=3,
                    color=RED_A
                )
                boundary_elements.add(element)
                
                # Right edge
                element = Line(
                    start=[2, -2 + 4*t[i], 0],
                    end=[2, -2 + 4*t[i+1], 0],
                    stroke_width=3,
                    color=RED_A
                )
                boundary_elements.add(element)
            
            # Add nodes at element boundaries
            nodes = VGroup()
            for element in boundary_elements:
                start_dot = Dot(element.get_start(), radius=0.04, color=RED)
                end_dot = Dot(element.get_end(), radius=0.04, color=RED)
                nodes.add(start_dot, end_dot)
            
            return VGroup(boundary_elements, nodes)

        # Create enhanced visualizations
        fem = create_enhanced_fem_mesh()
        fdm = create_enhanced_fdm_grid()
        bem = create_enhanced_bem_points()

        # Enhanced labels with better typography and positioning
        def create_method_label(text, color):
            return Text(
                text,
                font="Arial",
                color=color
            ).scale(0.6).to_edge(LEFT, buff=1)

        fem_label = create_method_label("Finite Element Method", BLUE)
        fdm_label = create_method_label("Finite Difference Method", GREEN)
        bem_label = create_method_label("Boundary Element Method", RED)

        # Enhanced animation sequence
        self.play(
            Write(title),
            run_time=1.5
        )
        self.play(
            Create(square),
            run_time=1.5
        )
        self.wait(0.5)

        # FEM visualization with annotations
        self.play(Write(fem_label))
        self.play(
            *[Create(triangle) for triangle in fem[0]],
            *[GrowFromCenter(node) for node in fem[1]],
            run_time=2,
            lag_ratio=0.05
        )
        
        fem_notes = VGroup(
            Text("• Flexible mesh adaptation", color=BLUE_A).scale(0.4),
            Text("• Better for irregular geometries", color=BLUE_A).scale(0.4),
            Text("• Higher computational cost", color=BLUE_A).scale(0.4)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(square, RIGHT)
        
        self.play(Write(fem_notes))
        self.wait(2)
        self.play(
            *[Uncreate(mob) for mob in fem],
            FadeOut(fem_label),
            FadeOut(fem_notes),
            run_time=1
        )

        # FDM visualization with annotations
        self.play(Write(fdm_label))
        self.play(
            *[Create(line) for line in fdm[0]],
            *[GrowFromCenter(point) for point in fdm[1]],
            run_time=2,
            lag_ratio=0.05
        )
        
        fdm_notes = VGroup(
            Text("• Simple implementation", color=GREEN_A).scale(0.4),
            Text("• Regular grid structure", color=GREEN_A).scale(0.4),
            Text("• Efficient for simple geometries", color=GREEN_A).scale(0.4)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(square, RIGHT)
        
        self.play(Write(fdm_notes))
        self.wait(2)
        self.play(
            *[Uncreate(mob) for mob in fdm],
            FadeOut(fdm_label),
            FadeOut(fdm_notes),
            run_time=1
        )

        # BEM visualization with annotations
        self.play(Write(bem_label))
        self.play(
            *[Create(element) for element in bem[0]],
            *[GrowFromCenter(node) for node in bem[1]],
            run_time=2,
            lag_ratio=0.05
        )
        
        bem_notes = VGroup(
            Text("• Reduces problem dimension", color=RED_A).scale(0.4),
            Text("• Efficient for infinite domains", color=RED_A).scale(0.4),
            Text("• Complex implementation", color=RED_A).scale(0.4)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(square, RIGHT)
        
        self.play(Write(bem_notes))
        self.wait(2)
        self.play(
            *[Uncreate(mob) for mob in bem],
            FadeOut(bem_label),
            FadeOut(bem_notes),
            run_time=1
        )

        # Clear EVERYTHING from scene, including title
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1
        )
        self.wait(0.5)

        # Show all methods together with enhanced layout
        comparison_square = Square(side_length=4).set_stroke(color=BOUNDARY_COLOR, width=3)
        
        # Position labels with more space from the left edge
        fem_label.to_edge(LEFT, buff=1)
        fdm_label.to_edge(LEFT, buff=1).shift(UP * 0.8)
        bem_label.to_edge(LEFT, buff=1).shift(UP * 1.6)
        
        # self.play(Create(comparison_square))
        
        self.play(
        #     *[Create(mob) for mob in fem],
        #     *[Create(mob) for mob in fdm],
        #     *[Create(mob) for mob in bem],
            FadeIn(fem_label),
            FadeIn(fdm_label),
            FadeIn(bem_label),
            run_time=2
        )

        # Enhanced explanation text
        # Enhanced explanation text with more detailed descriptions
        explanation = VGroup(
            VGroup(
                Text("Finite Element Method (FEM):", color=BLUE).scale(0.5),
                Text("• Divides entire domain into triangular elements", color=BLUE_A).scale(0.4),
                Text("• Excellent for complex geometries & materials", color=BLUE_A).scale(0.4),
                Text("• High accuracy but computationally intensive", color=BLUE_A).scale(0.4)
            ).arrange(DOWN, aligned_edge=LEFT),
            
            VGroup(
                Text("Finite Difference Method (FDM):", color=GREEN).scale(0.5),
                Text("• Uses regular grid points for approximation", color=GREEN_A).scale(0.4),
                Text("• Simple to implement and understand", color=GREEN_A).scale(0.4),
                Text("• Best for rectangular domains", color=GREEN_A).scale(0.4)
            ).arrange(DOWN, aligned_edge=LEFT),
            
            VGroup(
                Text("Boundary Element Method (BEM):", color=RED).scale(0.5),
                Text("• Only discretizes the boundary", color=RED_A).scale(0.4),
                Text("• Ideal for infinite domain problems", color=RED_A).scale(0.4),
                Text("• Reduces computational dimensions", color=RED_A).scale(0.4)
            ).arrange(DOWN, aligned_edge=LEFT)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.8).to_edge(RIGHT, buff=0.5)
        
        self.play(
            Write(explanation),
            run_time=2
        )
        self.wait(2)

        # Smooth fadeout
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5
        )

# To render:
# manim -pqh EnhancedNumericalMethodsComparison.py EnhancedNumericalMethodsComparison




from manim import *
from manim import *
from manim import *

from manim import *
from manim import *
import numpy as np

class LaplaceEquationSolver(Scene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#1a1a1a"  # Darker background
        
        # Title sequence
        title = Text("Solving Laplace's Equation", font_size=48)
        subtitle = Text("Using Finite Difference Method", font_size=32)
        subtitle.next_to(title, DOWN)
        title_group = VGroup(title, subtitle)
        
        self.play(Write(title_group))
        self.wait()
        self.play(FadeOut(title_group))

        # Step 1: Grid Introduction
        self.show_grid_formation()
        
        # Step 2: Boundary Conditions
        self.show_boundary_conditions()
        
        # Step 3: Matrix Formation
        self.show_matrix_formation()
        
        # Step 4: Solution
        self.show_solution_process()

    def show_grid_formation(self):
        # Create equation
        laplace_eq = MathTex(
            r"\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = 0",
            font_size=36
        ).to_edge(UP)

        # Create and animate grid formation
        grid = self.create_enhanced_grid()
        
        self.play(Write(laplace_eq))
        self.wait(0.5)
        
        # Animate grid creation with sequential cell appearance
        for cell in grid:
            self.play(Create(cell), run_time=0.05)
        
        self.wait()
        
        # Fade out for next section
        self.play(
            FadeOut(laplace_eq),
            grid.animate.scale(0.8).to_edge(LEFT)
        )
        return grid

    def show_boundary_conditions(self):
        # Create text explanation
        explanation = Text(
            "Boundary Conditions",
            font_size=36,
            color=BLUE
        ).to_edge(UP)
        
        # Create boundary visualization
        boundary_points = self.create_boundary_points()
        
        self.play(Write(explanation))
        
        # Animate boundary points with glow effect
        for point in boundary_points:
            self.play(
                Create(point),
                Flash(point.get_center(), color=BLUE, line_length=0.2),
                run_time=0.2
            )
        
        self.wait()
        
        # Transition out
        self.play(
            FadeOut(explanation),
            *[FadeOut(point) for point in boundary_points]
        )

    def show_matrix_formation(self):
        # Matrix equation
        matrix_eq = MathTex(
            r"A\mathbf{x} = \mathbf{b}",
            font_size=36
        ).to_edge(UP)
        
        # Create sparse matrix with animated formation
        matrix = self.create_animated_sparse_matrix()
        
        self.play(Write(matrix_eq))
        self.wait()
        
        # Show matrix formation with highlighting
        for element in matrix:
            self.play(
                Create(element),
                run_time=0.05
            )
        
        self.wait()
        
        # Transition out
        self.play(
            FadeOut(matrix_eq),
            FadeOut(matrix)
        )

    def create_enhanced_grid(self):
        """Create a more visually appealing grid"""
        grid = VGroup()
        n_rows, n_cols = 5, 5
        cell_size = 0.5
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Create cell
                cell = Square(
                    side_length=cell_size,
                    stroke_width=2,
                    stroke_color=WHITE
                )
                
                # Position
                cell.move_to([i*cell_size, j*cell_size, 0])
                
                # Add coordinates
                coords = Text(
                    f"({i},{j})",
                    font_size=16,
                    color=LIGHT_GRAY
                ).move_to(cell.get_center())
                
                grid.add(VGroup(cell, coords))
        
        # Center the grid
        grid.move_to(ORIGIN)
        return grid

    def create_animated_sparse_matrix(self):
        """Create an animated sparse matrix with better visuals"""
        matrix = VGroup()
        size = 25
        
        # Create matrix elements with better styling
        for i in range(size):
            for j in range(size):
                if self.should_draw_element(i, j):
                    element = Square(
                        side_length=0.2,
                        stroke_width=2,
                        fill_color=ORANGE,
                        fill_opacity=0.8
                    )
                    element.move_to([-2 + j*0.25, 2 - i*0.25, 0])
                    
                    # Add value label
                    if i == j:
                        value = Text("-4", font_size=12)
                    else:
                        value = Text("1", font_size=12)
                    value.move_to(element.get_center())
                    
                    matrix.add(VGroup(element, value))
        
        return matrix

    def create_boundary_points(self):
        """Create enhanced boundary points with better visuals"""
        points = VGroup()
        n_rows, n_cols = 5, 5
        cell_size = 0.5
        
        for i in range(n_rows):
            for j in range(n_cols):
                if i in [0, n_rows-1] or j in [0, n_cols-1]:
                    point = Dot(
                        point=[i*cell_size, j*cell_size, 0],
                        color=BLUE,
                        radius=0.1
                    )
                    # Add glow effect
                    glow = point.copy().set_opacity(0.5).scale(1.5)
                    points.add(VGroup(point, glow))
        
        points.move_to(ORIGIN)
        return points

    def show_solution_process(self):
        # Create solution visualization
        solution_grid = self.create_enhanced_grid()
        
        # Add potential values visualization
        potentials = self.create_potential_visualization()
        
        # Animate solution process
        self.play(
            Create(solution_grid),
            run_time=1
        )
        
        self.play(
            *[Create(pot) for pot in potentials],
            run_time=2
        )
        
        self.wait()
        
        # Final fadeout
        self.play(
            FadeOut(solution_grid),
            *[FadeOut(pot) for pot in potentials]
        )

    def create_potential_visualization(self):
        """Create visualization of potential values"""
        potentials = VGroup()
        n_rows, n_cols = 5, 5
        
        # Create color gradient for potentials
        colors = color_gradient([BLUE, GREEN, YELLOW, RED], n_rows * n_cols)
        
        for i in range(n_rows):
            for j in range(n_cols):
                potential = Circle(
                    radius=0.2,
                    fill_color=colors[i*n_cols + j],
                    fill_opacity=0.5,
                    stroke_width=0
                )
                potential.move_to([i*0.5, j*0.5, 0])
                potentials.add(potential)
        
        return potentials
