import pickle
from WebCamFeed import WebCamFeed
import cv2
import threading
from datetime import datetime
from DataLoader import visualize_last_run

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
        self.current_piece_num = 91 # Update This for Starting Save Index

        # Minimum Area Considered as Piece
        self.contour_area_cutoff_min = 200
        # Maximum Area Considered as Piece
        self.contour_area_cutoff_max = 100000
        # Saturation Minimum
        self.saturation_cutoff = 120
        # Value Minimum (For Shadows)
        self.value_cutoff = 140
        # Vertical Horizontal Threshold Distance
        self.vh_threshold = 300
        # Frame Delay
        self.frame_delay = 1
        # Thread which takes info from the webcam feed and constantly updates contour and board information
        self.analyze_thread = threading.Thread(target=self.analyze_board).start()

    '''
        All For Sliders
    '''
    def saturation_cutoff_change(self, val):
        self.saturation_cutoff = val

    def value_cutoff_change(self, val):
        self.value_cutoff = val

    def contour_area_cutoff_min_change(self, val):
        self.contour_area_cutoff_min = val

    def contour_area_cutoff_max_change(self, val):
        self.contour_area_cutoff_max = val
    
    '''
        Main Loop
    '''
    # Purpose: Main Update Loop. Takes Recent frame from self.webcam_feed and acts on it.
    def analyze_board(self):
        # Used for Initialization of Sliders
        is_first_show = True
        while self.webcam_feed.is_running:
            # If the current frame isn't empty (on initialization)
            if self.webcam_feed.current_frame.all() is not None:
                # Grab current frame from WebCamFeed
                image = self.webcam_feed.current_frame

                '''
                    Saturation Method (All Bright Colored Chips)
                '''
                # Convert to HSV (Hue, Saturation, Value) -> Value
                hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                #Threshold Saturation
                _, saturation = cv2.threshold(saturation, self.saturation_cutoff, 255, cv2.THRESH_BINARY)
                # Set Saturation Greyscale to 0 where value is less than value cutoff
                # Get pixels lower than threshold (value)
                _, value = cv2.threshold(value, self.value_cutoff, 255, cv2.THRESH_BINARY)
                # Mask Saturation with Value
                saturation = cv2.bitwise_and(saturation, saturation, mask=value)

                '''
                    Draw Circle Around Piece
                '''
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
                        
                        # See Chained Points
                        visualize_last_run()

                # Combine Images To Track Piece
                # mask = saturation
                # image[mask==255] = (36, 12, 255)

                '''
                    Draw Intersection Points (Kinda Slow)
                '''
                # Get Intersection Points
                intersection_points = self.get_intersection_points_from_lines(lines=self.webcam_feed.lines_coords)
                # Draw Intersection Points
                image = self.draw_points(image, intersection_points)

                # Draw Lines
                image = self.draw_lines(image)
                # Show the image
                cv2.imshow(self.webcam_feed.frame_title, image)
                # If First Run, Add Sliders
                if is_first_show:
                    # Saturation
                    cv2.createTrackbar('Saturation Cutoff', self.webcam_feed.frame_title, self.saturation_cutoff, 255,
                                       self.contour_area_cutoff_min_change)
                    # Value
                    cv2.createTrackbar('Value Cutoff', self.webcam_feed.frame_title, self.value_cutoff, 255,
                                       self.contour_area_cutoff_min_change)
                    # Min Area Slider
                    cv2.createTrackbar('Min Area', self.webcam_feed.frame_title, self.contour_area_cutoff_min, 400,
                                       self.contour_area_cutoff_min_change)
                    # Max Area Slider
                    cv2.createTrackbar('Max Area', self.webcam_feed.frame_title, self.contour_area_cutoff_max, 400,
                                       self.contour_area_cutoff_max_change)
                    is_first_show = False
                # Wait in between frames
                cv2.waitKey(self.frame_delay)

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

    # Draw Points
    def draw_points(self, image, points, color = (0, 0, 255)):
        for point in points:
            image = cv2.circle(image, point, radius=2, color = color, thickness=2)
        return image

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