import pickle
import os
import glob
import numpy as np
import cv2

class DataLoader:

    def __init__(self, data_folder = "piece_trials"):
        # Load Image Locations
        piece_trials_locations = glob.glob(data_folder + "/*")
        # Load Piece Trials
        self.trials_loaded = self.load_trials(piece_trials_locations)
        print(self.trials_loaded.shape)

    # Loads Single Trial
    def load_array(self, location):
        print(location)
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

class DrawPath:

    def __init__(self, position_vs_time, board_image = 'cached_data/board.jpg'):
        if position_vs_time is None or len(position_vs_time) == 0:
            raise Exception("No Path to Trace")
        # (x, y) array
        self.position_vs_time = position_vs_time
        # Filter Redundant Points
        self.position_vs_time = self.remove_same_points(self.position_vs_time)
        # Board Jpg
        self.board = cv2.imread(board_image)
        # Draw Lines
        image = self.draw_path(self.board, self.position_vs_time)
        # Plot Points
        image = self.draw_points(image, self.position_vs_time)
        # 
        # Show Image
        image = cv2.imshow("Plotted Route", image)
        cv2.waitKey(0)

    def remove_same_points(self, points):
        new_points = []
        for point in points:
            if len(new_points) == 0:
                new_points.append(point)
                continue
            if point[0] not in [p[0] for p in new_points] and point[1] not in [p[1] for p in new_points]:
                new_points.append(point)
        return np.array(new_points)

    def draw_path(self, image, points):
        for point1, point2 in zip(points, points[1:]): 
            image = cv2.line(image, point1, point2, [0, 255, 0], 2) 
        return image

    def draw_points(self, image, points):
        for point in points:
            image = cv2.circle(image, point, radius=2, color = (0, 0, 255), thickness=2)
        return image
if __name__ == "__main__":
    data_loader = DataLoader(data_folder = "piece_trials")
    draw_path = DrawPath(data_loader.trials_loaded[0])