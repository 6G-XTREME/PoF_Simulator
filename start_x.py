import os
import sys
import csv
import re
import send2trash
import logging
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QMessageBox
from PySide6.QtUiTools import QUiLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        self.setWindowTitle("Power over Fiber (PoF) Polling Simulator - E-Lighthouse")
        # Set the window size to match the specified geometry in Qt Designer
        self.setGeometry(self.ui.geometry())
        
        self.long_task_thread = None
        self.output_folder_path = "./output"
        
        # Connections
        self.ui.edit_simulator_input_checkBox.toggled.connect(self.toggle_editing_input_parameters)
        self.ui.use_nice_setup_checkBox.toggled.connect(self.toggle_editing_cells)
        self.ui.run_pushButton.clicked.connect(self.run_simulation_button)
        self.ui.use_solar_harvesting_checkBox.toggled.connect(self.toggle_solar_harvesting)
        self.ui.algorithm_comboBox.currentIndexChanged.connect(self.update_custom_configuration_state)
        
        # Actions
        menu_bar = self.findChild(QtWidgets.QMenuBar, "menubar")
        menu_figures = menu_bar.findChild(QtWidgets.QMenu, "menuFigures")
        menu_help = menu_bar.findChild(QtWidgets.QMenu, "menuHelp")
        action_close_figures = menu_figures.addAction("Close All Figures")
        action_close_figures.triggered.connect(self.close_all_figures)
        action_purge_runs = menu_help.addAction("Purge Output Folder")
        action_purge_runs.triggered.connect(lambda: self.purge_output_folder(self.output_folder_path))
        action_about = menu_help.addAction("About")
        action_about.triggered.connect(self.show_help)
        action_exit = menu_help.addAction("Exit")
        action_exit.triggered.connect(self.exit)
        
        # Save progress bar for later
        self.simulation_progressbar = self.findChild(QtWidgets.QWidget, "simulation_progressBar")
        
        ## MatPlotLib Things
        # Find the existing QWidget named "figure" created in Qt Designer
        self.figure = self.findChild(QtWidgets.QWidget, "figure")

        # Convert the "figure" widget to a Matplotlib canvas
        self.canvas_widget = FigureCanvas(Figure(figsize=(5, 3)))
        layout = QtWidgets.QVBoxLayout(self.figure)
        layout.addWidget(self.canvas_widget)
        
        # Set table options
        self.ui.table_csv.setEditTriggers(QTableWidget.NoEditTriggers)  # Make the table read-only
        self.ui.table_csv.setSortingEnabled(True)
        
        # Start conditions
        self.ui.tabWidget.setCurrentIndex(0)
        self.toggle_editing_input_parameters(False)
        
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
        
        if self.ui.save_output_checkBox.isChecked():
            # Find the new folder created
            output_run = self.get_last_created_output_folder()
            message = f"Simulation had finished. The simulation data has been saved inside: output/{output_run}. View data inside Results Tab."
            self.update_results_table(f"./output/{output_run}/data/{output_run}-output.csv")
        else:
            message = "Simulation had finished."
        
        # Show a message box when the long-running task finishes
        QtWidgets.QMessageBox.information(self, "Simulation finished", message)
    
    def toggle_editing_cells(self, state):
        self.ui.NMacroCells_edit.setReadOnly(state)
        self.ui.NFemtoCells_edit.setReadOnly(state)
        if state:
            self.ui.NMacroCells_edit.setText("3")
            self.ui.NFemtoCells_edit.setText("20")
      
    def toggle_editing_input_parameters(self, state):
        for i in range(self.ui.InputParameters.count()):
            widget = self.ui.InputParameters.itemAt(i).widget()
            if widget:
                widget.setDisabled(not state)
                if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit)):
                    widget.setReadOnly(state)
    
    def toggle_solar_harvesting(self, state):
        self.ui.weather_comboBox.setEnabled(state)
        
    def update_custom_configuration_state(self):
        selected_value = self.ui.algorithm_comboBox.currentText()
        for i in range(self.ui.RightPanel.count()):
            widget = self.ui.RightPanel.itemAt(i).widget()
            if widget:
                widget.setDisabled(selected_value != "E-Lighthouse")
                
    def update_results_table(self, file: str):
        self.ui.results_msg.setText(f"Reading results of file : {file}")
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

            # Extract the column names from the first row of the CSV data
            column_names = data[0]

            # Remove the first row (column names) from the data
            data = data[1:]

            self.ui.table_csv.setRowCount(len(data))
            self.ui.table_csv.setColumnCount(len(column_names))

            # Set the extracted column names as column headers
            self.ui.table_csv.setHorizontalHeaderLabels(column_names)

            for row_index, row_data in enumerate(data):
                for col_index, cell_value in enumerate(row_data):
                    item = QTableWidgetItem(cell_value)
                    self.ui.table_csv.setItem(row_index, col_index, item)
    
    def purge_output_folder(self, directory_path, loop=False) -> None:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                uuid_pattern = r'^[0-9a-fA-F]+$'  # Check if the folder name matches the UUID pattern (e.g., "4877ba88")
                if re.match(uuid_pattern, item):
                    logging.info(f"Moving UUID folder to trash: {item}")
                    self.purge_output_folder(item_path, True)  # Recursively move contents of UUID folders to trash
                    send2trash.send2trash(item_path)  # Send the UUID folder to trash
                else:
                    logging.info(f"Skipping human-readable folder: {item}")
        if not loop:
            QtWidgets.QMessageBox.information(self, "Purge Output Folder", f"Succesfully trashed the content of the folder: {self.output_folder_path}")
    
    def close_all_figures(self) -> None:
        logging.info("Closing all figures...")
        try:
            plt.close('all')
        except Exception as ex:
            logging.error(ex)
        return
            
    def show_help(self) -> None:
        QtWidgets.QMessageBox.information(self, "Help", "This is the help content.")
        return
    
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.close_all_figures()    # Try to close all figures
            event.accept()
        else:
            event.ignore()

    def exit(self) -> None:
        logging.info("Exit simulator...")
        sys.exit(0)
        return
            
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
            config_parameters['use_user_list'] = False
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
            custom_config['use_harvesting'] = self.ui.use_solar_harvesting_checkBox.isChecked()
            custom_config['weather'] = str(self.ui.weather_comboBox.currentText()).upper()
            custom_config['MapScale'] = float(self.ui.MapScale_edit.text())
            custom_config['fiberAttdBperKm'] = float(self.ui.fiber_att_edit.text())
            try: 
                custom_config['extraPoFCharger'] = self.ui.extra_charger_checkBox.isChecked()
            except RuntimeError:
                custom_config['extraPoFCharger'] = False 
        except ValueError:
            print("Unable to convert to number")
            
        print(custom_config)
        return custom_config
    
    def get_last_created_output_folder(self) -> str:
        # Get a list of all directories in the output folder
        subdirectories = [d for d in os.listdir(self.output_folder_path) if os.path.isdir(os.path.join(self.output_folder_path, d))]
        # Sort the directories by creation time
        subdirectories.sort(key=lambda d: os.path.getctime(os.path.join(self.output_folder_path, d)))
        
        if subdirectories:
            last_created_directory = subdirectories[-1]
            return last_created_directory
        else:
            return ""

if __name__ == "__main__":
    logging.info("Starting UI")
    app = QtWidgets.QApplication(sys.argv)
    main_window = PoF_Simulator_App()
    main_window.show()
    sys.exit(app.exec())