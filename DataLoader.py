import pickle
import os
import glob
import numpy as np
import cv2
import math
import json
import re

def remove_duplicate_points(points):
    new_points = []
    for point in points:
        if len(new_points) == 0:
            new_points.append(point)
            continue
        found_match = False
        for new_point in new_points:
            if np.array_equal(point, new_point):
                found_match = True
                break
        if not found_match:
            new_points.append(point)
    return np.array(new_points)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class DataLoader:

    def __init__(self, data_folder = "piece_trials"):
        # Load Image Locations
        piece_trials_locations = glob.glob(data_folder + "/*")
        # Sort Properly
        piece_trials_locations.sort(key=natural_keys)
        # Load Piece Trials
        self.trials_loaded = self.load_trials(piece_trials_locations)

    def __len__(self):
        return len(self.trials_loaded)

    # Loads Single Trial
    def load_array(self, location):
        if location is None:
            raise Exception("No Location Loaded")
        with open(location, "rb") as f:
            return pickle.load(f)

    # Loads All trialls
    def load_trials(self, piece_trials_locations):
        trials_loaded = []
        for trial in piece_trials_locations:
            trials_loaded.append(self.load_array(trial))
        return trials_loaded


class ClusterPointsToIntersections:

    # Max Line Length
    max_line_length = 300

    # Vertical Horizontal Threshold Distance
    vh_threshold = 300

    def __init__(self, points, lines_location = "cached_data/lines.pkl"):
        # Lines Location
        self.lines_location = lines_location
        # Points
        self.points = np.array(points)
        # Load Lines
        with open(self.lines_location, 'rb') as f:
            self.loaded_lines = pickle.load(f)
        # Get Intersection Points
        self.intersection_points = self.get_intersection_points_from_lines(self.loaded_lines)
        # Cluster Points to Intersection Points
        self.clustered_points = self.cluster(self.points, self.intersection_points)
        # Chain Points a Max Distance Away
        self.chained_points = self.chain_points(self.clustered_points)
        print(f"Chained Points Shape: {self.chained_points.shape}")

    # Chains Points A Max Distance Away
    def chain_points(self, points):
        chained_points = []
        for point1, point2 in zip(points, points[1:]):
            if math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) < self.max_line_length:
                chained_points.append(point1)
                chained_points.append(point2)
        return remove_duplicate_points(chained_points)

    def cluster(self, points, intersection_points):
        def closest_node(node, nodes):
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return nodes[np.argmin(dist_2)]

        clustered_points = []
        for point in points:
            clustered_points.append(closest_node(point, intersection_points))
        return remove_duplicate_points(clustered_points)


    # Get Intersection Points of Lines
    def get_intersection_points_from_lines(self, lines):

        # Took From Online
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        # Took From Online
        def find_intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x,y
            else:
                return False

        intersection_points = [] # Like [(x1, y1), ...]

        # Sort Lines by Horizontal and Vertical
        vertical_lines = []
        horizontal_lines = []
        # Get Y Coordinates of Horizontal Lines / X Coordinates of Vertical Lines
        for line_coords in lines:
            # Lines Coords
            x1, y1 = line_coords[0]
            x2, y2 = line_coords[1]
            # Color Depends on Orientation
            if abs(y1-y2) > self.vh_threshold: # Vertical Lines
                vertical_lines.append(line_coords)
            elif abs(x1-x2) > self.vh_threshold: # Horizontal Lines
                horizontal_lines.append(line_coords)

        # Get Y Coordinates of Horizontal Lines / X Coordinates of Vertical Lines
        for horizontal in horizontal_lines:
            for vertical in vertical_lines:
                # Get Lines
                L1 = line(horizontal[0], horizontal[1])
                L2 = line(vertical[0], vertical[1])
                # Find Intersection Point
                intersection = find_intersection(L1, L2)
                # Check if there is intesection point and not already added
                if intersection is not False and intersection not in intersection_points:
                    # Check if is within bounds
                    x1 = horizontal[0][0]; y1 = horizontal[0][1]
                    x2 = horizontal[1][0]; y2 = horizontal[1][1]

                    x3 = vertical[0][0]; y3 = vertical[0][1]
                    x4 = vertical[1][0]; y4 = vertical[1][1]

                    x = round(intersection[0])
                    y = round(intersection[1])

                    if (x1 < x < x2 or x2 < x < x1) and (y3 < y < y4 or y4 < y < y3):
                        intersection_points.append((x, y))
                    
        return intersection_points



class DrawPath:

    def __init__(self, points, board_image = 'cached_data/board.jpg', show = True):
        # Points
        points = points
        # Board Jpg
        self.board = cv2.imread(board_image)
        # Draw Lines
        self.board = self.draw_path(self.board, points)
        # Plot Points
        self.board = self.draw_points(self.board, points)
        # Show Image
        if show:
            cv2.imshow("Plotted Route", self.board)
            cv2.waitKey(0)
            cv2.destroyWindow("Plotted Route")
        # Can not show and get image from self.board

    def draw_path(self, image, points):
        for point1, point2 in zip(points, points[1:]):
            image = cv2.line(image, point1, point2, [0, 255, 0], 2)
        return image

    def draw_points(self, image, points, color = (0, 0, 255)):
        for point in points:
            image = cv2.circle(image, point, radius=2, color = color, thickness=2)
        return image


def visualize_last_run():
     # Loads All Data
    data_loader = DataLoader()
    print(f"{len(data_loader)} files Loaded")
    # Get Last Run
    point = data_loader.trials_loaded[-1]
    # Get Processed Point
    processed_points = ClusterPointsToIntersections(point).chained_points
    # Show Chained Points
    DrawPath(processed_points)


def visualize_all_runs():
    # Loads All Data
    data_loader = DataLoader()
    print(f"{len(data_loader)} files Loaded")
    # Process Points For Each File
    for point in data_loader.trials_loaded:
        # Get Processed Point
        processed_points = ClusterPointsToIntersections(point).chained_points
        # Show Chained Points
        DrawPath(processed_points)

# Brings Trial Files to Json Analyze File
if __name__ == "__main__":
    
    # Loads All Data
    data_loader = DataLoader()
    print(f"{len(data_loader)} files Loaded")

    # Initialize Processed Points
    data_total = []
    # Process Points For Each File
    for point in data_loader.trials_loaded:
        # Get Processed Point
        processed_points = ClusterPointsToIntersections(point).chained_points
        # # Show Chained Points
        DrawPath(processed_points)
        data_total.append(processed_points.tolist())

    # Open Json Data
    json_loaded = None
    if os.path.exists('final_runs.json'):
        with open('final_runs.json', 'r') as f:
            json_loaded = json.loads(f.read())
    else:
        json_loaded = {}
    if json_loaded is None:
        raise Exception("Json Not Loaded Correctly")
    
    # Start Json Save Number
    file_counter = 0
    # Append New Data and Save Json
    for data in data_total:
        print(data)
        json_loaded[f'Run {file_counter}'] = data
        file_counter += 1
    
    # Save Json
    with open('final_runs.json', 'w') as f:
        json.dump(json_loaded, f, ensure_ascii=False, indent=4)
