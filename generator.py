import numpy as np
import random
import yaml

FILE_NAME = 'large_problem.yaml'

N_CITIES = 10_000
MIN_CONNECTIONS = 5
MAX_CONNECTIONS = min(int(np.log(N_CITIES)*5),N_CITIES)
X_SIZE = 10_000
Y_SIZE = 10_000
Z_SIZE = 10_000
IS_SOLVABLE = True

def generate_large_prism_problem(
    n_cities=N_CITIES,
    x_size=X_SIZE,
    y_size=Y_SIZE,
    z_size=Z_SIZE,
    min_connections=MIN_CONNECTIONS,
    max_connections=MAX_CONNECTIONS,
    is_solvable = IS_SOLVABLE):

    print('Generating city names')
    cities = [f"{i}" for i in range(n_cities)]
    city_connections = {}
    city_metadata = {}

    print('Placing cities randomly in the 3D space')
    coordinates = {city: np.array(
        [np.random.uniform(0, x_size),
         np.random.uniform(0, y_size),
         np.random.uniform(0, z_size)])
        for city in cities}

    print('Choosing start and end city')
    start_city = str(np.random.choice(cities))
    while True:
        end_city = str(np.random.choice(cities))
        if end_city != start_city:
            break
    end_city_coords = coordinates[end_city]

    if is_solvable:
        print('Establishing guaranteed path from start to end')
        n_connections = random.randint(min_connections, max_connections)
        guaranteed_path = random.sample(cities, k = n_connections) # Samples without repetition
        for i in range(1, len(guaranteed_path)):
            city_a = guaranteed_path[i-1]
            city_b = guaranteed_path[i]
            distance = np.linalg.norm(coordinates[city_b] - coordinates[city_a]) # Calculate 3D distance
            distance = float(round(distance, 2))

            if f"city_{city_a}" not in city_connections:
                city_connections[f"city_{city_a}"] = {"connects_to": {}}
            if f"city_{city_b}" not in city_connections:
                city_connections[f"city_{city_b}"] = {"connects_to": {}}

            city_connections[f"city_{city_a}"]["connects_to"][city_b] = distance
            city_connections[f"city_{city_b}"]["connects_to"][city_a] = distance

    cities_set = set(cities)
    if not is_solvable:
        print('Making sure the graph is not solvable.')
        cities_set -= {end_city}

    print('Adding random connections and calculating LOS and altitude differences')
    for city, coord in coordinates.items():

        if not is_solvable: # Edit to actually ensure unsolvability
            if end_city == city:
                continue

        cities_subset = cities_set - {city}
        n_connections = random.randint(min_connections, max_connections)
        connected_cities = random.sample(list(cities_subset), n_connections)

        connections = {c: np.linalg.norm(coordinates[c] - coord) for c in connected_cities} # Calculate 3D distance

        # Update or add to the city_connections dictionary
        if f"city_{city}" not in city_connections:
            city_connections[f"city_{city}"] = {"connects_to": {}}

        for connected_city, distance in connections.items():
            distance = float(round(distance, 2))
            # Only add the connection if it doesn't already exist
            if connected_city not in city_connections[f"city_{city}"]["connects_to"]:
                city_connections[f"city_{city}"]["connects_to"][connected_city] = distance

            # Ensure bidirectional connection is added without duplication
            if f"city_{connected_city}" not in city_connections:
                city_connections[f"city_{connected_city}"] = {"connects_to": {}}
            if city not in city_connections[f"city_{connected_city}"]["connects_to"]:
                city_connections[f"city_{connected_city}"]["connects_to"][city] = distance

        # Calculate line of sight and altitude difference
        los_distance = float(np.linalg.norm(coord[:2] - end_city_coords[:2]))  # (on the X-Y plane, on purpose)
        altitude_difference = float(abs(coord[2] - end_city_coords[2]))        # (on the Z plane)

        # Add to city metadata
        city_metadata[f"city_{city}"] = {
            "line_of_sight_distance": round(los_distance, 3),
            "altitude_difference": round(altitude_difference, 3)}

    print('Creating the result dictionary')
    return {
        "problem": {
            "cities": cities,
            **city_connections,
            "city_start": start_city,
            "city_end": end_city},
        "additional_information": city_metadata}

# Write to a YAML file
with open(FILE_NAME, 'w') as outfile:
    data = generate_large_prism_problem()
    print('Exporting .yaml file')
    yaml.dump(data, outfile)