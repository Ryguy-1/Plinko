from audioop import avg
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



    '''
        Plots End Distribution
    '''
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
    cv2.imwrite('average_image.jpg', avg_image)
    cv2.waitKey(0)

