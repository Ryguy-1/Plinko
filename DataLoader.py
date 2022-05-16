import pickle
import os
import glob
import numpy as np
import cv2
import math

class DataLoader:

    def __init__(self, data_folder = "piece_trials"):
        # Load Image Locations
        piece_trials_locations = glob.glob(data_folder + "/*")
        # Load Piece Trials
        self.trials_loaded = self.load_trials(piece_trials_locations)

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
        return np.array(trials_loaded)


class ClusterPointsToIntersections:

    # Vertical Horizontal Threshold Distance
    vh_threshold = 300

    def __init__(self, points, lines_location = "cached_data/lines.pkl"):
        # Lines Location
        self.lines_location = lines_location
        # Points
        self.points = points
        # Load Lines
        with open(self.lines_location, 'rb') as f:
            self.loaded_lines = pickle.load(f)
        # Get Intersection Points
        self.intersection_points = self.get_intersection_points_from_lines(self.loaded_lines)
        # Cluster Points to Intersection Points
        self.clustered_points = self.cluster(self.points, self.intersection_points)

    def cluster(self, points, intersection_points):
        def closest_node(node, nodes):
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return nodes[np.argmin(dist_2)]

        clustered_points = []
        for point in points:
            clustered_points.append(closest_node(point, intersection_points))
        return clustered_points


    def get_intersection_points_from_lines(self, lines):
        intersection_points = [] # Like [(x1, y1), ...]

        x_coords_vertical_lines = []; y_coords_horizontal_lines = []
        # Get Y Coordinates of Horizontal Lines / X Coordinates of Vertical Lines
        for line_coords in lines:
            # Lines Coords
            x1, y1 = line_coords[0]
            x2, y2 = line_coords[1]
            # Color Depends on Orientation
            if abs(y1-y2) > self.vh_threshold: # Vertical Lines
                x_coords_vertical_lines.append(round(np.mean([x1, x2])))
            elif abs(x1-x2) > self.vh_threshold: # Horizontal Lines
                y_coords_horizontal_lines.append(round(np.mean([y1, y2])))

        # Get Intersection Points
        for x in x_coords_vertical_lines:
            for y in y_coords_horizontal_lines:
                intersection_points.append((x, y))
            
        return np.array(intersection_points)

class DrawPath:

    # Max Line Length
    max_line_length = 300

    def __init__(self, position_vs_time, intersection_points, unclustered_points, board_image = 'cached_data/board.jpg'):
        if position_vs_time is None or len(position_vs_time) == 0:
            raise Exception("No Path to Trace")
        # (x, y) array
        self.position_vs_time = position_vs_time
        # Board Jpg
        self.board = cv2.imread(board_image)
        # Draw Lines
        self.board = self.draw_path(self.board, self.position_vs_time)
        # Plot Points
        # self.board = self.draw_points(self.board, intersection_points, color = (0, 255, 0))
        # self.board = self.draw_points(self.board, unclustered_points, color = (255, 0, 0))
        self.board = self.draw_points(self.board, self.position_vs_time)
        # Show Image
        cv2.imshow("Plotted Route", self.board)
        cv2.waitKey(0)

    def draw_path(self, image, points):
        for point1, point2 in zip(points, points[1:]):
            if math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) < self.max_line_length:
                image = cv2.line(image, point1, point2, [0, 255, 0], 2) 
        return image

    def draw_points(self, image, points, color = (0, 0, 255)):
        for point in points:
            image = cv2.circle(image, point, radius=2, color = color, thickness=2)
        return image


if __name__ == "__main__":
    data_loader = DataLoader(data_folder = "piece_trials")
    for trial in data_loader.trials_loaded:
        clustered_points = ClusterPointsToIntersections(trial)
        draw_path = DrawPath(clustered_points.clustered_points, clustered_points.intersection_points, trial)