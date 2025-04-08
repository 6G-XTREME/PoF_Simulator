from model.NetworkGraph import CompleteGraph, GraphCompletePlots
import pandas as pd
import scipy.io
import json
def plot_test(mat_path, xlsx_path):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_path)
    distance_matrix = mat_contents['crossMatrix']

    # Load the .xlsx file
    xlsx_data = pd.read_excel(xlsx_path)

    graph = CompleteGraph(distance_matrix, xlsx_data)
    graph.plot_graph()
    
    
def plot_test_2():
    graph = CompleteGraph.of("Passion_Xtreme_III.mat", "NameTypes.xlsx")
    graph.plot_graph()
    with open("graph.json", "w") as f:
        json.dump(graph.to_json(), f)
        
def plot_test_3():
    graph = CompleteGraph.of("Passion_Xtreme_III.mat", "NameTypes.xlsx")
    graph.plot_graph_with_map()
    
def plot_test_4():
    graph = CompleteGraph.of("Passion_Xtreme_III.mat", "NameTypes.xlsx")
    GraphCompletePlots.plot_without_map(graph)

def plot_test_5():
    graph = CompleteGraph.of("Passion_Xtreme_III.mat", "NameTypes.xlsx")
    GraphCompletePlots.plot_for_node_degree(graph, include_node_labes=False)


if __name__ == "__main__":
    # plot_test_2()
    # plot_test_3()
    # plot_test_4()
    plot_test_5()