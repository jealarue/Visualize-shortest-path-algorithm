import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import copy

# Graph definition
my_graph = {
    'A': [('B', 5), ('C', 3), ('E', 11)],
    'B': [('A', 5), ('C', 1), ('F', 2)],
    'C': [('A', 3), ('B', 1), ('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 9), ('F', 3)],
    'E': [('A', 11), ('C', 5), ('D', 9)],
    'F': [('B', 2), ('D', 3)]
}

def shortest_path_steps(graph, start, target=''):
    unvisited = list(graph)
    distances = {node: 0 if node == start else float('inf') for node in graph}
    paths = {node: [] for node in graph}
    paths[start].append(start)
    
    steps = []
    
    while unvisited:
        current = min(unvisited, key=distances.get)
        steps.append((current, copy.deepcopy(distances), copy.deepcopy(paths)))
        
        for node, distance in graph[current]:
            if distance + distances[current] < distances[node]:
                distances[node] = distance + distances[current]
                if paths[node] and paths[node][-1] == node:
                    paths[node] = paths[current][:]
                else:
                    paths[node].extend(paths[current])
                paths[node].append(node)
        
        unvisited.remove(current)
    
    steps.append((None, distances, paths))
    return steps

# Generate steps for the shortest path
steps = shortest_path_steps(my_graph, 'A', 'F')

# Create a NetworkX graph from the input graph
G = nx.Graph()
for node, edges in my_graph.items():
    for neighbor, weight in edges:
        G.add_edge(node, neighbor, weight=weight)

pos = nx.spring_layout(G)

def update_graph(step):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=16)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    
    if step < len(steps) - 1:
        current_node, distances, paths = steps[step]
        plt.title(f"Current Node: {current_node}")
        
        if current_node is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='red', node_size=1000)
        
        for target, path in paths.items():
            if len(path) > 1:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    else:
        _, distances, paths = steps[-1]
        plt.title("Final Distances and Paths")
        
        for target, path in paths.items():
            if len(path) > 1:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    
    plt.show()

# Create the interactive widget
step_slider = IntSlider(value=0, min=0, max=len(steps)-1, step=1, description='Step:', continuous_update=False)
interact(update_graph, step=step_slider)
