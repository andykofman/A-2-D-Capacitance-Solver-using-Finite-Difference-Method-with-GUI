import sys
from PyQt5.QtWidgets import QApplication
from frontend.controllers.main_controller import MainController
# from qt_material import apply_stylesheet
def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Capacitance Solver")
    app.setOrganizationName("CapSolver")
    app.setOrganizationDomain("capsolver.org")

    # apply_stylesheet(app, theme='dark_cyan.xml')
    
    # Initialize main controller
    controller = MainController()

   
    
    # Show main window
    controller.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()