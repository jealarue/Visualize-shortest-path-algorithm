import matplotlib.pyplot as plt
import networkx as nx
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define the graph
my_graph = {
    'A': [('B', 3), ('D', 1)],
    'B': [('A', 3), ('C', 4)],
    'C': [('B', 4), ('D', 7)],
    'D': [('A', 1), ('C', 7)]
}

# Modified shortest path function to track steps
def shortest_path_steps(graph, start):
    steps = []
    unvisited = list(graph)
    distances = {node: 0 if node == start else float('inf') for node in graph}
    paths = {node: [] for node in graph}
    paths[start].append(start)
    
    while unvisited:
        current = min(unvisited, key=distances.get)
        steps.append((current, dict(distances), dict(paths), f"Current Node: {current}"))
        
        for neighbor, distance in graph[current]:
            new_distance = distances[current] + distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                paths[neighbor] = paths[current] + [neighbor]
        
        unvisited.remove(current)
    
    steps.append((None, dict(distances), dict(paths), "Final Distances and Paths"))
    return steps

# Generate steps for visualization
steps = shortest_path_steps(my_graph, 'A')

# Create a directed graph using networkx
G = nx.DiGraph()

# Add nodes and edges with weights
for node, neighbors in my_graph.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# Custom layout to set positions based on edge weights
def custom_layout(G, scale=1):
    pos = nx.spring_layout(G)
    for edge in G.edges(data=True):
        u, v, data = edge
        pos[v] = pos[u] + (pos[v] - pos[u]) * data['weight'] / max(nx.get_edge_attributes(G, 'weight').values()) * scale
    return pos

# Set positions for nodes
pos = custom_layout(G, scale=2)

# Function to update the graph visualization
def update_graph(step):
    clear_output(wait=True)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=15, font_weight='bold', arrowsize=20)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    
    if step < len(steps) - 1:
        current_node, distances, paths, description = steps[step]
        plt.title(description)
        
        if current_node is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='red', node_size=2000)
        
        for target, path in paths.items():
            if len(path) > 1:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.5)
    else:
        _, distances, paths, description = steps[step]
        plt.title(description)
    
    plt.show()
    print(description)
    print("Distances:", distances)
    print("Paths:", paths)

# Interactive widgets
step_slider = widgets.IntSlider(value=0, min=0, max=len(steps)-1, step=1, description='Step:', continuous_update=False)
play_button = widgets.Play(min=0, max=len(steps)-1, step=1, interval=2000)

widgets.jslink((play_button, 'value'), (step_slider, 'value'))

# Display the interactive plot
def display_graph(step):
    update_graph(step)
    
step_slider.observe(lambda change: display_graph(change.new), names='value')

display(widgets.HBox([play_button, step_slider]))
display_graph(0)
