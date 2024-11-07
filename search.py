import numpy as np
import yaml
import argparse

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

def create_graph(graph_data):
    num_cities = len(graph_data['problem']['cities'])
    graph = np.full((num_cities, num_cities), np.inf)
    for city, details in graph_data['problem'].items():
        if city.startswith('city_') and city not in ['city_start', 'city_end']:
            city_index = int(city.split('_')[1])
            for neighbor, cost in details['connects_to'].items():
                neighbor_index = int(neighbor)
                graph[city_index, neighbor_index] = cost
    return graph

def reconstruct(came_from, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def calculate_path(graph_data, heuristic):
    start = int(graph_data['problem']['city_start'])
    end = int(graph_data['problem']['city_end'])
    graph = create_graph(graph_data)
    num_nodes = graph.shape[0]

    open_set = [(0, start)]  # List of tuples (f_score, node)
    came_from = {}
    g_score = np.full(num_nodes, np.inf)
    g_score[start] = 0

    f_score = np.full(num_nodes, np.inf)
    f_score[start] = heuristic(start, end)  # Since heuristic is 0 for now, this is just formal.

    while open_set:
        # Find the node with the lowest f_score
        open_set.sort(key=lambda x: x[0])  # Sort by f_score
        current_f, current = open_set.pop(0)  # Pop the first element, the one with the smallest f_score

        if current == end:
            print(g_score[end])
            return reconstruct(came_from, start, end)

        for neighbor in range(num_nodes):
            if graph[current, neighbor] != np.inf:  # There is a connection
                tentative_g = g_score[current] + graph[current, neighbor]
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    # Only add to the open set if not already in or if found a better path
                    if neighbor not in [n for f, n in open_set]:
                        open_set.append((f_score[neighbor], neighbor))

    return None

def main():
    parser = argparse.ArgumentParser(description='Process yaml file for graph data.')
    parser.add_argument('yaml_file', help='Path to the YAML file containing graph data.')
    args = parser.parse_args()
    graph_data = load_yaml(args.yaml_file)
    path = calculate_path(graph_data, lambda x, y: 0)  # Using a lambda for zero heuristic
    print("Path:", path)

if __name__ == '__main__':
    main()
