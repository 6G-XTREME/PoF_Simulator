import sys
import logging
from PySide6 import QtWidgets, QtCore
from PySide6.QtUiTools import QUiLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from simulator.launch import execute_simulator

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s',
                    level=logging.INFO)     # Change level to DEBUG if needed

class PoF_Executor_Thread(QtCore.QThread):
    def __init__(self, func, canvas_widget, progressbar_widget, input_parameters: dict, config_parameters: dict, custom_parameters: dict):
        super().__init__()
        self.func = func
        self.input_parameters : dict = input_parameters
        self.config_parameters : dict = config_parameters
        self.custom_parameters : dict = custom_parameters
        self.canvas_widget = canvas_widget
        self.progressbar_widget = progressbar_widget
    
    def run(self):
        logging.info("Execute simulator...")
        self.func(input_parameters=self.input_parameters, 
                  config_parameters=self.config_parameters, 
                  custom_parameters=self.custom_parameters,
                  canvas_widget=self.canvas_widget,
                  progressbar_widget=self.progressbar_widget)

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
        self.setWindowTitle("PoF Simulator v0.1")
        # Set the window size to match the specified geometry in Qt Designer
        self.setGeometry(self.ui.geometry())
        self.long_task_thread = None
        
        # Connections
        self.ui.edit_simulator_input_checkBox.toggled.connect(self.toggle_editing_input_parameters)
        self.ui.run_pushButton.clicked.connect(self.run_simulation_button)
        
        self.simulation_progressbar = self.findChild(QtWidgets.QWidget, "simulation_progressBar")
        
        ## MatPlotLib Things
        # Find the existing QWidget named "figure" created in Qt Designer
        self.figure = self.findChild(QtWidgets.QWidget, "figure")

        # Convert the "figure" widget to a Matplotlib canvas
        self.canvas_widget = FigureCanvas(Figure(figsize=(5, 3)))
        layout = QtWidgets.QVBoxLayout(self.figure)
        layout.addWidget(self.canvas_widget)
        
    def run_simulation_button(self):
        if self.long_task_thread and self.long_task_thread.isRunning():
            # Don't start a new thread if the previous one is still running
            return

        self.ui.run_pushButton.setDisabled(True)
        self.simulation_progressbar.setValue(0)
        simulator_input = self.get_input_parameters()
        simulator_config = self.get_config_parameters()
        custom_config = self.get_custom_config()
        self.long_task_thread = PoF_Executor_Thread(execute_simulator, 
                                               input_parameters=simulator_input, 
                                               config_parameters=simulator_config, 
                                               custom_parameters=custom_config,
                                               canvas_widget=self.canvas_widget,
                                               progressbar_widget=self.simulation_progressbar)
        self.long_task_thread.finished.connect(self.on_simulaton_finished)
        self.long_task_thread.start()
        
    def on_simulaton_finished(self):
        self.ui.run_pushButton.setEnabled(True)
        logging.info("UI: simulation done")
        
    def toggle_editing_input_parameters(self, state):
        self.ui.battery_capacity_edit.setReadOnly(state)
        # Apply style sheet to change appearance when disabled
        if state:
            self.ui.battery_capacity_edit.setStyleSheet("QLineEdit:disabled { background-color: #2e2e2e; color: #808080; }")
        else:
            self.ui.battery_capacity_edit.setStyleSheet("")  # Clear style sheet
            
    def get_input_parameters(self) -> dict :
        input_parameters = {}
        try:
            input_parameters['battery_capacity'] = float(self.ui.battery_capacity_edit.text())
            input_parameters['small_cell_consumption_on'] = float(self.ui.small_cell_consumption_on_edit.text())
            input_parameters['small_cell_consumption_sleep'] = float(self.ui.small_cell_consumption_sleep_edit.text())
            input_parameters['small_cell_voltage_min'] = 0.01 * float(self.ui.small_cell_voltage_min_edit.text())
            input_parameters['small_cell_voltage_max'] = 0.01 * float(self.ui.small_cell_voltage_max_edit.text())
            input_parameters['Maplimit'] = 1000
            input_parameters['Users'] = int(self.ui.users_edit.text())
            input_parameters['mean_user_speed'] = float(self.ui.mean_user_speed_edit.text())
            input_parameters['Simulation_Time'] = int(60 * float(self.ui.simTime_edit.text()))
            input_parameters['timeStep'] = float(self.ui.timeStep_edit.text())
            input_parameters['numberOfLasers'] = int(self.ui.numberOfLasers_edit.text())
            input_parameters['noise'] = float(self.ui.noise_edit.text())
            input_parameters['SMA_WINDOW'] = 5
            input_parameters['NMacroCells'] = int(self.ui.NMacroCells_edit.text())
            input_parameters['NFemtoCells'] = int(self.ui.NFemtoCells_edit.text())
            input_parameters['TransmittingPower'] = {}
            input_parameters['TransmittingPower']['PMacroCells'] = float(self.ui.PMacroCells_edit.text())
            input_parameters['TransmittingPower']['PFemtoCells'] = float(self.ui.PFemtoCells_edit.text())
            input_parameters['TransmittingPower']['PDevice'] = float(self.ui.PDevice_edit.text())
            input_parameters['TransmittingPower']['MacroCellDownlinkBW'] = float(self.ui.MacroCellDownlinkBW_edit.text())
            input_parameters['TransmittingPower']['FemtoCellDownlinkBW'] = float(self.ui.FemtoCellDownlinkBW_edit.text())
            input_parameters['TransmittingPower']['alpha_loss'] = float(self.ui.alpha_loss_edit.text())
        except ValueError:
            print("Unable to convert to number")
            
        print(input_parameters)
        return input_parameters
    
    def get_config_parameters(self) -> dict:
        config_parameters = {}
        try:
            config_parameters['algorithm'] = self.ui.algorithm_comboBox.currentText().lower().replace("-", "")
            config_parameters['use_nice_setup'] = self.ui.use_nice_setup_checkBox.isChecked()
            config_parameters['use_user_list'] = self.ui.use_user_list_checkBox.isChecked()
            config_parameters['show_plots'] = self.ui.show_plots_checkBox.isChecked()
            config_parameters['show_live_plots'] = False
            config_parameters['speed_live_plots'] = 0.05
            config_parameters['save_output'] = self.ui.save_output_checkBox.isChecked()
            config_parameters['output_folder'] = self.ui.output_folder_edit.text()
        except ValueError:
            print("Unable to convert to numbers")
        
        print(config_parameters)
        return config_parameters
    
    def get_custom_config(self) -> dict:
        custom_config = {}
        try:
            custom_config['user_report_position'] = self.ui.user_report_position_spinBox.value()
            custom_config['startup_max_tokens'] = self.ui.startup_max_tokens_spinBox.value()
            custom_config['poweroff_unused_cell'] = self.ui.poweroff_unused_cell_spinBox.value()
        except ValueError:
            print("Unable to convert to number")
            
        print(custom_config)
        return custom_config

if __name__ == "__main__":
    logging.info("Starting UI")
    app = QtWidgets.QApplication(sys.argv)
    main_window = PoF_Simulator_App()
    main_window.show()
    sys.exit(app.exec())