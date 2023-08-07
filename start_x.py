import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

class PoF_Simulator_App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load and set up the UI
        self.loader = QUiLoader()
        self.ui = self.loader.load("./ui/simulator.ui", None)
        self.centralwidget = self.ui.findChild(QtWidgets.QWidget, "centralwidget") 

        # Set the central widget for the QMainWindow
        self.setCentralWidget(self.ui)
        # Set the window title and initial size to match the loaded UI
        self.setWindowTitle("User Data")
        # Set the window size to match the specified geometry in Qt Designer
        self.setGeometry(self.ui.geometry())
        
        # Connections
        self.ui.edit_simulator_input_checkBox.toggled.connect(self.toggle_editing_input_parameters)
        
    def toggle_editing_input_parameters(self, state):
        self.ui.battery_capacity_edit.setReadOnly(state)
        # Apply style sheet to change appearance when disabled
        if state:
            self.ui.battery_capacity_edit.setStyleSheet("QLineEdit:disabled { background-color: #2e2e2e; color: #808080; }")
        else:
            self.ui.battery_capacity_edit.setStyleSheet("")  # Clear style sheet

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = PoF_Simulator_App()
    main_window.show()
    sys.exit(app.exec())