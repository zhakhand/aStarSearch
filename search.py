import numpy as np
import yaml
import argparse


def load_yaml(filepath):
    """
    Load and parse a YAML file from the given filepath.

    Args:
        filepath (str): The path to the YAML file to be loaded.

    Returns:
        dict: The parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the specified file cannot be found.
        ValueError: If there is an error parsing the YAML file.
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{filepath}'")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing file: {e}")


def create_graph(graph_data):
    """
    Create a graph representation from the input graph data.

    This function extracts city connections from the graph data, 
    excluding the start and end cities.

    Args:
        graph_data (dict): A dictionary containing graph configuration data.

    Returns:
        dict: A dictionary representing the graph, where each key is a city 
              and the value is a dictionary of connected cities and their connection costs.
    """
    graph = {}
    for city, details in graph_data['problem'].items():
        if city.startswith('city_') and city not in ['city_start', 'city_end']:
            city_name = city.split('_', 1)[1]
            graph[city_name] = details['connects_to']
    return graph


def reconstruct_path(came_from, end):
    """
    Reconstruct the path from start to end using the came_from dictionary.

    Args:
        came_from (dict): A dictionary tracking the previous node for each city.
        end (str): The destination city.

    Returns:
        list: A list of cities representing the path from start to end, 
              in order from start to end.
    """
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def heuristic(info, graph, flag):
    """
    Calculate heuristic values for each city based on the specified mode.

    Args:
        info (dict): Additional information about cities, including 
                     line of sight distance and altitude difference.
        graph (dict): The graph representation of cities.
        flag (int): Heuristic calculation mode:
            - 3: Euclidean distance (line of sight + altitude)
            - Other: Line of sight distance

    Returns:
        dict: A dictionary of heuristic values for each city.
    """
    heuristics = {}
    for city in graph:
        los = info[f'city_{city}']['line_of_sight_distance']
        alt_dif = info[f'city_{city}']['altitude_difference']
        if flag == 3:
            heuristics[f'city_{city}'] = float(np.sqrt(los ** 2 + alt_dif ** 2))
        else:
            heuristics[f'city_{city}'] = abs(los)
    return heuristics


def calculate_path(graph_data, mode=1):
    """
    Calculate the optimal path using A* search algorithm with different heuristic modes.
    
    Args:
        graph_data (dict): A dictionary containing graph information.
        mode (int, optional): Heuristic calculation mode:
            1: No heuristic (Dijkstra's algorithm)
            2: Line of sight distance heuristic
            3: Euclidean distance heuristic (line of sight + altitude)
    
    Returns:
        tuple: A tuple containing:
            - Optimal path from start to end
            - Total path cost
            - Number of expanded nodes
            - Heuristic values for each city
    
    Raises:
        ValueError: If start or end cities are not defined, or graph is unsolvable
    """
    # Validate start and end cities
    start = graph_data['problem']['city_start']
    end = graph_data['problem']['city_end']
    if start is None or end is None:
        raise ValueError("Start or end not defined")
    
    # Create graph and validate its structure
    graph = create_graph(graph_data)
    info = graph_data['additional_information']
    if not graph or start not in graph or end not in graph:
        raise ValueError("Graph is unsolvable")

    # Calculate heuristics based on the selected mode
    if mode == 1:
        # No heuristic (Dijkstra's algorithm)
        heuristics = {'city_' + city: 0 for city in graph}
        frontier = [(0, start)]
        g_val = {node: float('inf') for node in graph}
        g_val[start] = 0
    else:
        # A* search with heuristics
        heuristics = heuristic(info, graph, mode)
        frontier = {start: heuristics[f'city_{start}']}
        g_val = {node: float('inf') for node in graph}
        g_val[start] = 0
        f_val = {node: float('inf') for node in graph}
        f_val[start] = heuristics[f'city_{start}']

    # Common search variables
    came_from = {}
    visited = set()

    # Search algorithm
    while frontier:
        # Different node selection based on mode
        if mode == 1:
            # Dijkstra's algorithm: sort and pop lowest cost
            frontier.sort(key=lambda x: x[0])
            current_cost, current = frontier.pop(0)
        else:
            # A* search: select lowest f-value
            current = min(frontier, key=frontier.get)
            frontier.pop(current)

        # Skip already visited nodes
        if current in visited:
            continue
        
        # Goal check: if we've reached the end city
        if current == end:
            return reconstruct_path(came_from, end), g_val[end], len(visited), heuristics
        
        # Mark current node as visited
        visited.add(current)

        # Explore neighbors
        for neighbor, cost in graph[current].items():
            # Skip already visited nodes
            if neighbor in visited:
                continue
            
            # Compute new path cost
            new_g_val = g_val[current] + cost

            # Different path update logic based on mode
            if mode == 1:
                # Dijkstra's algorithm
                if new_g_val < g_val[neighbor]:
                    came_from[neighbor] = current # Update path
                    g_val[neighbor] = new_g_val # Update g-value
                    frontier.append((g_val[neighbor], neighbor)) # Add neighbor to frontier
            else:
                # A* search with heuristic
                neighbor_city = f'city_{neighbor}'
                new_f_val = new_g_val + heuristics[neighbor_city]
                
                # Update path if a better route is found
                if new_f_val < f_val[neighbor]:
                    came_from[neighbor] = current # Update path
                    g_val[neighbor] = new_g_val # Update g-value
                    f_val[neighbor] = new_f_val # Update f-value
                    frontier[neighbor] = new_f_val # Add neighbor to frontier

    # No path found
    raise ValueError("Graph is unsolvable: No path!")


def a_star_search(graph_data, mode):
    """
    Perform A* search with different heuristic modes and output results.

    This function runs the search algorithm for a given graph and mode,
    then writes the solution to a YAML output file.

    Args:
        graph_data (dict): A dictionary containing graph configuration data.
        mode (int): Heuristic mode:
            - 1: No heuristic (Dijkstra's algorithm)
            - 2: Line of sight distance heuristic
            - 3: Euclidean distance heuristic

    Writes:
        YAML file: Outputs the solution details to 'output_{mode}.yaml'
    """
    path, cost, exp_nodes, heuristics = calculate_path(graph_data, mode)
    output = {
        "solution": {
            "cost": cost,
            "path": path,
            "expanded_nodes": exp_nodes,
            "heuristic": heuristics
        }
    }
    with open(f'output_{mode}.yaml', 'w') as file:
        yaml.dump(output, file, allow_unicode=True, sort_keys=False)


def main():
    """
    Main entry point of the script.

    Parses command-line arguments, loads the YAML graph data,
    and runs A* search with different heuristic modes.

    Handles and prints any exceptions that occur during execution.
    """
    parser = argparse.ArgumentParser(description='Process yaml file for graph data.')
    parser.add_argument('yaml_file', help='Path to the YAML file containing graph data.')
    args = parser.parse_args()

    try:
        graph_data = load_yaml(args.yaml_file)  # Load graph data
        a_star_search(graph_data, 1)          # Run A* search with different heuristic modes
        a_star_search(graph_data, 2)
        a_star_search(graph_data, 3)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
