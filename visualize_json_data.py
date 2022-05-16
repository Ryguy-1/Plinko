import matplotlib.pyplot as plt
import numpy as np
import json


def load_json(location = "final_runs.json"):
    data = None
    with open(location, 'r') as f:
        data = json.loads(f.read())
    if data is not None:
        return data
    raise Exception("Could not Load Json")

if __name__ == "__main__":
    # Load Json
    data = load_json()
    # Get Ending X Coordinates
    ending_x_coordinates = []
    for coords in data.values():
        end_coord = coords[-1]
        end_x = end_coord[0]
        ending_x_coordinates.append(end_x)
    plt.hist(ending_x_coordinates, bins=np.arange(np.min(ending_x_coordinates), np.max(ending_x_coordinates)+1), rwidth=10)
    plt.show()