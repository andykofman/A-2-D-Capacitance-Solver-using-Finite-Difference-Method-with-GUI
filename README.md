# 2D Capacitance Solver

## Overview

The 2D Capacitance Solver is an open-source project designed to solve 2D capacitance problems using a custom mathematical engine. It includes a backend for performing calculations and a frontend for user interaction, following the Model-View-Controller (MVC) architecture. This tool is ideal for educational and research purposes in the field of electrical engineering and physics.


-------------
Under media/videos/How to use the program, you will find a quick tutorial for the program.
More details can be found in the documentation file.
-------------
## Features

- **Backend**: 
  - Solves 2D capacitance problems using a custom mathematical engine.
  - Supports non-uniform grid generation.
  - Provides detailed logging for debugging and performance analysis.
  - Capable of visualizing electric fields and potential distributions.

- **Frontend**:
  - User-friendly graphical user interface (GUI) for easy interaction.
  - Visualizes results with plots and diagrams.
  - Allows users to input parameters and view results dynamically.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/andykofman/2D-FieldSolver-for-Solving-ParasiticCapacitance
   cd 2d-capacitance-solver
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

To start the application, run the following command:
bash
python main.py




### Backend

The backend is responsible for all calculations and performance optimizations. It includes:

- **Coordinate System**: Manages grid points and physical dimensions.
- **Laplace Solver**: Solves the Laplace equation for potential distribution.
- **Capacitance Extraction**: Computes the capacitance matrix and charges.
- **Nonuniform Grid**: Generates non-uniform grids for enhanced accuracy.

### Frontend

The frontend provides a GUI for user interaction. It allows users to:

- Input physical dimensions and material properties.
- Define conductor geometries.
- Visualize potential distributions and electric fields.
- Export results for further analysis.

## Project Structure
2d-capacitance-solver/
  │
  ├── backend.py # Backend calculations and logic
  ├── frontend/ # Frontend GUI files
  │ ├── main.py # Entry point for the GUI
  │ ├── views.py # GUI layout and components
  │ ├── controllers.py # Handles user interactions
  │ └── models.py # Data models and logic
  │
  ├── requirements.txt # Python dependencies
  ├── README.md # Project documentation
  ├── LICENSE # License information
  └── .gitignore # Git ignore file



## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and users who have helped improve this project.
- Special thanks to the open-source community for providing valuable resources and tools.

## Contact

For questions or feedback, please contact [ali.a@aucegypt.edu](mailto:ali.a@aucegypt.edu).
