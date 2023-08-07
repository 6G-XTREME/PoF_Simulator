import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

loader = QUiLoader()    # Set up a Loader object

app = QtWidgets.QApplication(sys.argv)
centralwidget = loader.load("./ui/simulator.ui", None)
centralwidget.setWindowTitle("User data")

centralwidget.show()
sys.exit(app.exec())