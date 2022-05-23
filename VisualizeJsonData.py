import matplotlib.pyplot as plt
import numpy as np
import json
from DataLoader import DrawPath
import cv2

def load_json(location = "final_runs.json"):
    data = None
    with open(location, 'r') as f:
        data = json.loads(f.read())
    if data is not None:
        return data
    raise Exception("Could not Load Json")

if __name__ == "__main__":

    # Load Json
    data = load_json(location = "project_used_data/33mm/final_runs.json")

    '''
        Plots End Distribution
    '''
    # Get Ending X Coordinates
    ending_x_coordinates = []
    for coords in data.values():
        end_coord = coords[-1]
        end_x = end_coord[0]
        ending_x_coordinates.append(end_x)
    plt.hist(ending_x_coordinates, bins=np.arange(np.min(ending_x_coordinates), np.max(ending_x_coordinates)+1), rwidth=10)
    plt.show()



    '''
        Plots Overlay of Runs
    '''
    # Get All Images of Paths
    boards = []
    for coords in data.values():
        boards.append(DrawPath(coords, show = False).board)
    
    # Average Images
    avg_image = boards[0]
    for i in range(len(boards)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(boards[i], alpha, avg_image, beta, 0.0)

    cv2.imshow('Final Image', avg_image)
    # cv2.imwrite('average_image.jpg', avg_image)
    cv2.waitKey(0)


    '''
        Shows Distribution as Chips Fall
    '''
    rows = {}
    all_chains = data.values()
    all_chains = list(all_chains)
    # Calculate Minimum Nodes Reached by a Chain
    min_nodes = 255
    for chain in all_chains:
        if len(chain) < min_nodes:
            min_nodes = len(chain)
    # Create an Array of Node Prevelence Per 'Row'
    node_tracker = {} # Dictionary Like {'Row 1': {'[x1, y1]': 10, '[x2, y2]': 2, ...}, 'Row 2'...}\
    print(f"Min Nodes: {min_nodes}")
    for chain in all_chains:
        for i in range(min_nodes):
            # Check if Row Exists Already
            if f'Row {i}' not in node_tracker:
                node_tracker[f'Row {i}'] = {}
            # Check if Coords Already In This Row as Key and Add if Not
            if str(chain[i]) not in node_tracker[f'Row {i}']:
                node_tracker[f'Row {i}'][str(chain[i])] = 0
                continue
            # If Already In, Add One
            node_tracker[f'Row {i}'][str(chain[i])] += 1

    # print(node_tracker)


    '''
        Unique Combinations Within X Number of Moves (Like Chess Openings) -> The less here the more consistent
    '''
    # Initialize moves_unique
    max_depth = 15
    moves_unique = {} # Like {'combinations depth 1': [[[x, y]]], 'combinations depth 2': [[[x, y], [x, y]], [[x, y], [x, y]], ...], ...}
    for i in range(min(max_depth, len(all_chains[i]))):
        moves_unique[f'Combinations Depth {i}'] = []
    # Iterate Through all_chains
    for i in range(len(all_chains)):
        for j in range(min(max_depth, len(all_chains[i]))):
            # Declare Chain so Far
            chain_so_far = all_chains[i][0:j]
            if chain_so_far not in moves_unique[f"Combinations Depth {j}"]:
                moves_unique[f"Combinations Depth {j}"].append(chain_so_far)
    for key, value in moves_unique.items():
        value = list(value)
        print(f"{key} Length: {len(value)}")

            

