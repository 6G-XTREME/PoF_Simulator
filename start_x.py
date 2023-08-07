import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

class PoF_Simulator_App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load and set up the UI
        self.loader = QUiLoader()
        self.centralwidget = self.loader.load("./ui/simulator.ui", None)
        # Set the central widget for the QMainWindow
        self.setCentralWidget(self.centralwidget)
        # Set the window title and initial size to match the loaded UI
        self.setWindowTitle("User Data")
        self.setGeometry(100, 100, 800, 600)  # Set window position (x, y) and size (width, height)
        
        # Connections
        self.centralwidget.edit_simulator_input_checkBox.toggled.connect(self.toggle_editing_input_parameters)
        
    def toggle_editing_input_parameters(self, state):
        self.centralwidget.battery_capacity_edit.setReadOnly(state)
        # Apply style sheet to change appearance when disabled
        if state:
            self.centralwidget.battery_capacity_edit.setStyleSheet("QLineEdit:disabled { background-color: #2e2e2e; color: #808080; }")
        else:
            self.centralwidget.battery_capacity_edit.setStyleSheet("")  # Clear style sheet

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = PoF_Simulator_App()
    main_window.show()
    sys.exit(app.exec())