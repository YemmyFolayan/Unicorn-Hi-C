import random
import numpy as np


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        

def initialize_structure_c_style(sequence_length, min_dist=2.0, max_dist=6.0, use_lattice=True):
    
    #Initialize the 3D structure with sequence_length and dynamically computed cube size.
    
    size_cube = int(sequence_length * 5)  # Adjust multiplier for larger space
    nodes = []
    global_restart_attempts = 10  # Increase restart attempts
    restart_count = 0
    lattice_spacing = min_dist

    # Gradual relaxation parameters
    min_dist_step = 0.5
    max_dist_step = 1.0

    while len(nodes) < sequence_length and restart_count < global_restart_attempts:
        #print(f"Restarting initialization. Attempt {restart_count + 1}/{global_restart_attempts}")
        nodes = [Node(size_cube // 2, size_cube // 2, size_cube // 2)]  # Start at center

        # Sequential placement with adjacency constraints
        for i in range(1, sequence_length):
            candidates = find_candidate_positions(nodes[-1], size_cube, max_dist)
            valid_candidates = [c for c in candidates if is_valid_position(c, nodes, min_dist, max_dist)]
            if valid_candidates:
                nodes.append(random.choice(valid_candidates))
            else:
                adjacent_node = Node(nodes[-1].x + min_dist, nodes[-1].y, nodes[-1].z)
                nodes.append(adjacent_node)

        if len(nodes) < sequence_length:
            print(f"Failed to place all nodes. Adjusting constraints and restarting...")
            restart_count += 1
            min_dist = max(0.5, min_dist - min_dist_step)  # Relax min_dist
            max_dist += max_dist_step  # Relax max_dist

    if len(nodes) < sequence_length:
        raise RuntimeError(f"Failed to initialize {sequence_length} nodes after {global_restart_attempts} attempts.")

    coordinates = np.array([[node.x, node.y, node.z] for node in nodes])

    # Apply continuity refinement
    coordinates = refine_path_continuity(coordinates)

    return scale_coordinates(coordinates, size_cube)


def find_candidate_positions(last_node, size_cube, max_dist, direction=(1, 1, 1)):
    
    #Generate candidate positions with directional bias.
    
    candidates = []
    for _ in range(200):  # Generate multiple candidates
        dx = random.uniform(-max_dist, max_dist) * direction[0]
        dy = random.uniform(-max_dist, max_dist) * direction[1]
        dz = random.uniform(-max_dist, max_dist) * direction[2]
        x, y, z = last_node.x + dx, last_node.y + dy, last_node.z + dz
        if 0 <= x < size_cube and 0 <= y < size_cube and 0 <= z < size_cube:
            candidates.append(Node(x, y, z))
    return candidates


def is_valid_position(new_node, nodes, min_dist, max_dist):
    
    #Validate the new node based on distance constraints with all previously placed nodes.
    
    for node in nodes:
        dist = np.sqrt((new_node.x - node.x) ** 2 + (new_node.y - node.y) ** 2 + (new_node.z - node.z) ** 2)
        if dist < min_dist or dist > max_dist:
            return False
    return True


def scale_coordinates(coordinates, size_cube):
    
    #Scale the coordinates to fit within the size_cube using the ratio logic.
    
    max_val = np.max(np.abs(coordinates))
    scaling_factor = size_cube / max_val if max_val != 0 else 1
    return (coordinates * scaling_factor).reshape(-1, 3)


def refine_path_continuity(coordinates, max_step=3):
    
    #Ensure path-like continuity by limiting step sizes between nodes.
    
    for i in range(1, len(coordinates)):
        step = coordinates[i] - coordinates[i - 1]
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            coordinates[i] = coordinates[i - 1] + (step / step_norm) * max_step
    return coordinates
