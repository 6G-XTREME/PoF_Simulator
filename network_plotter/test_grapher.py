from model.GraphComplete import GraphComplete
import pandas as pd
import scipy.io

def plot_test(mat_path, xlsx_path):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_path)
    distance_matrix = mat_contents['crossMatrix']

    # Load the .xlsx file
    xlsx_data = pd.read_excel(xlsx_path)

    graph = GraphComplete(distance_matrix, xlsx_data)
    graph.plot_graph()
    
    
def plot_test_2():
    graph = GraphComplete.of("Passion_Xtreme_III.mat", "NameTypes.xlsx")
    graph.plot_graph()


if __name__ == "__main__":
    plot_test_2()