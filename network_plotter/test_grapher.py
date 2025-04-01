from model.GraphComplete import GraphComplete
import pandas as pd
import scipy.io
import json
def plot_test(mat_path, xlsx_path):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_path)
    distance_matrix = mat_contents['crossMatrix']

    # Load the .xlsx file
    xlsx_data = pd.read_excel(xlsx_path)

    graph = GraphComplete(distance_matrix, xlsx_data)
    graph.plot_graph()
    
    
def plot_test_2():
    graph = GraphComplete.of("network_plotter/Passion_Xtreme_III.mat", "network_plotter/NameTypes.xlsx")
    graph.plot_graph()
    with open("graph.json", "w") as f:
        json.dump(graph.to_json(), f)


if __name__ == "__main__":
    plot_test_2()