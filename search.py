import numpy as np
import yaml
import argparse


def load_yaml(filepath):
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{filepath}'")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing file: {e}")


def create_graph(graph_data):
    graph = {}
    for city, details in graph_data['problem'].items():
        if city.startswith('city_') and city not in ['city_start', 'city_end']:
            city_name = city.split('_', 1)[1]
            graph[city_name] = details['connects_to']
    return graph


def reconstruct_path(came_from, start, end):
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def heuristic(info, graph, flag):
    heuristics = {}
    for city in graph:
        los = info[f'city_{city}']['line_of_sight_distance']
        alt_dif = info[f'city_{city}']['altitude_difference']
        if flag == 3:
            heuristics[f'city_{city}'] = float(np.sqrt(los ** 2 + alt_dif ** 2))
        else:
            heuristics[f'city_{city}'] = abs(los)
    return heuristics


def calculate_path_heuristic(graph_data, mode):
    start = graph_data['problem']['city_start']
    end = graph_data['problem']['city_end']
    if start is None or end is None:
        raise ValueError("Start or end not defined")
    graph = create_graph(graph_data)
    info = graph_data['additional_information']
    if not graph or start not in graph or end not in graph:
        raise ValueError("Graph is unsolvable")

    heuristics = heuristic(info, graph, mode)
    open_set = {start: 0}
    came_from = {}
    g_val = {node: float('inf') for node in graph}
    g_val[start] = 0
    f_val = {node: float('inf') for node in graph}
    f_val[start] = heuristics[f'city_{start}']
    expanded = 0
    visited = set()

    while open_set:
        current_node, current_f = min(open_set.items(), key=lambda x: x[1])
        del open_set[current_node]

        expanded += 1
        if current_node == end:
            return reconstruct_path(came_from, start, end), g_val[end], expanded, heuristics

        visited.add(current_node)

        for neighbor, cost in graph[current_node].items():
            if neighbor in visited:
                continue
            new_g_val = g_val[current_node] + cost
            new_f_val = new_g_val + heuristics[f'city_{neighbor}']
            if new_f_val < f_val[neighbor]:
                came_from[neighbor] = current_node
                g_val[neighbor] = new_g_val
                f_val[neighbor] = new_f_val
                open_set[neighbor] = new_f_val

    raise ValueError("Graph is unsolvable: No path!")


def calculate_path_no_heuristic(graph_data):
    start = graph_data['problem']['city_start']
    end = graph_data['problem']['city_end']
    if start is None or end is None:
        raise ValueError("Start or end not defined")

    graph = create_graph(graph_data)
    if not graph or start not in graph or end not in graph:
        raise ValueError("Graph is unsolvable")

    open_set = [(0, start)]
    came_from = {}
    g_val = {node: float('inf') for node in graph}
    g_val[start] = 0
    expanded = 0
    heuristics = {'city_' + city: 0 for city in graph}
    visited = {}

    while open_set:
        open_set.sort(key=lambda x: x[0])
        current_f, current = open_set.pop(0)
        expanded += 1

        if current == end:
            return reconstruct_path(came_from, start, end), g_val[end], expanded, heuristics

        for neighbor, cost in graph[current].items():
            new_g_val = g_val[current] + cost
            if new_g_val < g_val[neighbor]:
                came_from[neighbor] = current
                g_val[neighbor] = new_g_val
                if neighbor not in visited:
                    open_set.append((g_val[neighbor], neighbor))
                    visited[neighbor] = g_val[neighbor]

    raise ValueError("Graph is unsolvable: No path!")


def a_star_search(graph_data, mode):
    if mode == 1:
        path, cost, exp_nodes, heuristics = calculate_path_no_heuristic(graph_data)
    else:
        path, cost, exp_nodes, heuristics = calculate_path_heuristic(graph_data, mode)
    output = {
        "solution": {
            "cost": cost,
            "path": path,
            "expanded_nodes": exp_nodes,
            "heuristic": heuristics
        }
    }
    with open(f'output_{mode}.yaml', 'w') as file: # 1: h(n) = 0, 2: h(n) = los_dist, 3: h(n) = sqrt(los^2 + alt^2)
        yaml.dump(output, file, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Process yaml file for graph data.')
    parser.add_argument('yaml_file', help='Path to the YAML file containing graph data.')
    args = parser.parse_args()

    try:
        graph_data = load_yaml(args.yaml_file)
        a_star_search(graph_data, 1)
        a_star_search(graph_data, 2)
        a_star_search(graph_data, 3)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
