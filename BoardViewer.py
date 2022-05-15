import pickle
from re import L
import numpy as np
from WebCamFeed import WebCamFeed
import cv2
import threading
import time
from datetime import datetime

# Analyzes Webcam Feed. Draws Contours. Updates Member Variable: self.board_representation based on Web Cam Feed
class BoardViewer:

    # Purpose: Initializes Individual Webcam Feed. Hold/Update Board Representation. Start Analyze Board Thread.
    def __init__(self, frame_width=1000, frame_height=1000):
        # WebCamFeed for each BoardViewer
        self.webcam_feed = WebCamFeed(frame_width, frame_height)

        '''
            Holds Current Piece Location Information    
        '''
        # Array Like [(x1, y1), (x2, y2), ...]
        self.current_piece_location_over_time = []
        self.current_piece_num = 0 # Update This for Starting Save Index

        # Canny Edge Detection Lower, Upper
        self.canny_lower = 70
        self.canny_upper = 90
        # Minimum Area Considered as Piece
        self.contour_area_cutoff_min = 100
        # Maximum Area Considered as Piece
        self.contour_area_cutoff_max = 100000
        # Saturation Minimum
        self.saturation_cutoff = 120
        # Vertical Horizontal Threshold Distance
        self.vh_threshold = 300
        # Frame Delay
        self.frame_delay = 5
        # Thread which takes info from the webcam feed and constantly updates contour and board information
        self.analyze_thread = threading.Thread(target=self.analyze_board).start()


    # Purpose: Main Update Loop. Takes Recent frame from self.webcam_feed and acts on it.
    def analyze_board(self):

        while self.webcam_feed.is_running:
            # If the current frame isn't empty (on initialization)
            if self.webcam_feed.current_frame.all() is not None:
                # Grab current frame from WebCamFeed
                image = self.webcam_feed.current_frame
                # Convert to HSV (Hue, Saturation, Value) -> Value
                hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                #Threshold Saturation
                _, saturation = cv2.threshold(saturation, self.saturation_cutoff, 255, cv2.THRESH_BINARY)
                # cv2.imshow('saturation', saturation)
                # cv2.imshow('value', value)
                # Find Contours
                (contours, hierarchy) = cv2.findContours(saturation.copy(), cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_SIMPLE)
                # Filter Contours By Area
                contours, hierarchy = self.filter_contours_by_area(contours, hierarchy)
                # Do Contour Things
                if len(contours) > 0:
                    # Get Center of Contour
                    x_center, y_center = self.get_center_of_contour(contour=contours[0])
                    # Put Text Above Piece
                    cv2.putText(image, "Piece :)", (x_center-25, y_center-25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                    # Draw Circle Around Piece
                    cv2.circle(image, (x_center, y_center), radius=15, color = (0, 0, 255), thickness=3)

                # Check if is Timing /do Appropriate Actions
                if self.webcam_feed.is_timing:
                    # Put Timing Text
                    image = cv2.putText(image.copy(), str((datetime.now()-self.webcam_feed.start_time).total_seconds())[:5], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
                    self.current_piece_location_over_time.append((x_center, y_center))
                else: # Timer Turned Off
                    # If Still Items This is First Call Turned Off
                    if len(self.current_piece_location_over_time) > 0:
                        # Save
                        with open(f'piece_trials/piece{self.current_piece_num}.pkl', 'wb') as f:
                            pickle.dump(self.current_piece_location_over_time, f)
                        # Empty Array
                        self.current_piece_location_over_time = []
                        self.current_piece_num += 1

                # Combine Images To Track Piece
                # mask = saturation
                # image[mask==255] = (36, 12, 255)
                # Draw Lines
                image = self.draw_lines(image)
                # Show the image
                cv2.imshow(self.webcam_feed.frame_title, image)
                # Wait in between frames
                cv2.waitKey(self.frame_delay)

    # Draws Lines
    def draw_lines(self, image):
        image = image.copy()
        for line_coords in self.webcam_feed.lines_coords:
            # Lines Coords
            x1, y1 = line_coords[0]
            x2, y2 = line_coords[1]
            # Color Depends on Orientation
            if abs(y1-y2) > self.vh_threshold: # Vertical Lines
                cv2.line(image, line_coords[0], line_coords[1], color = (36, 255, 12), thickness=1)
            elif abs(x1-x2) > self.vh_threshold:
                cv2.line(image, line_coords[0], line_coords[1], color = (255, 36, 12), thickness=1)
        return image

    # Purpose: Filter Contours by Area (Closed Mandatory)
    # Input: Contour List, Hierarchy List
    # Output: Contours Filtered by Area, Corresponding Hierarchy
    def filter_contours_by_area(self, contours, hierarchy):
        # New Empty Lists for Contours and Hierarchy
        new_contours = []
        new_hierarchy = [[]]
        # Iterate through Contours
        for i in range(len(contours)):
            # If Contour Area (closed) is greater than value 1 and less than value 2, it is a piece
            if (cv2.contourArea(contours[i]) > self.contour_area_cutoff_min) and \
                    (cv2.contourArea(contours[i]) < self.contour_area_cutoff_max):
                # Append the Piece Contours to the new_contours and new_hierarchy lists
                new_contours.append(contours[i])
                new_hierarchy[0].append(hierarchy[0][i])
        # Return Contour and Hierarchy Lists
        return new_contours, new_hierarchy


    # Purpose: Finds Centroid using Pixel Values
    # Input: Contour
    # Output: Centroid Coordinates of Contour
    def get_center_of_contour(self, contour):
        # cv2.moments read data
        m = cv2.moments(contour)
        # Get X Coordinate, Get Y Coordinate
        contour_center_x = int(m["m10"] / m["m00"])
        contour_center_y = int(m["m01"] / m["m00"])
        # return the centroid x and y coordinates
        return contour_center_x, contour_center_y

if __name__ == "__main__":
    BoardViewer()