from datetime import datetime
import glob
import cv2
import keyboard
import threading
import numpy as np
import time
import pickle
import os
import re

# Encapsulates WebCam Feed. (Get Current Frame through web_cam_feed.current_frame)
class WebCamFeed:

    # Title of Frame
    frame_title = "Plinko Board Viewer [('q') to Quit, ('r') to Reset Mask, ('u') to Undo Line, ('s') to Start/Stop Trial, ('d') to Delete Last Trial]"

    # Purpose: Initialize Video Capture / Member Variables
    def __init__(self, frame_width, frame_height):
        print("Initializing Webcam Stream...")
        # Video Capture Variable initialize and set size(0 = webcam live feed, cv2.CAP_DSHOW = Direct Show (video input)
        # (also makes loading much faster)
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Done.")
        # Frame width and height (not image capture height/width)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        # Running Live Capture
        self.is_running = True
        # Current Frame initialize to empty
        self.current_frame = np.array([None])
        # Initialize Capture Thread and Start it
        print("Starting Live Capture...")
        self.capture_thread = threading.Thread(target=self.run_live_feed).start()
        print("Running.")
        # Cropping
        self.cropping = False
        self.crop_start = (0, 0)  # Reset when lift mouse button
        self.crop_end = (0, 0)  # Reset when lift mouse button
        self.mask = None  # Calculated Mask to Apply to Each Frame

        # Lines Draw
        # Array for lines like [[(x1, y1), (x2, y2)], [...]...]
        self.lining = False
        self.line_start = (0, 0)
        self.line_end = (0, 0)
        self.lines_coords = []

        # Timer (Start and Stop Trials)
        self.is_timing = False
        self.start_time = 0

        # Load Lines
        if os.path.exists('cached_data/lines.pkl'):
            self.load_lines()

        # Initialize Cropping Feature
        self.prompt_crop = threading.Thread(target=self.prompt_crop).start()

    # Purpose: Camera Loop.
    # Output: Set Member (self.current_frame) equal to most recent frame (read later in Board Viewer)
    def run_live_feed(self):
        # While is running
        while self.is_running:
            # ret = True if frame read correctly, frame = numpy array of frame read
            ret, frame = self.vid.read()
            # Save Image for Use Later if Wanted
            cv2.imwrite("cached_data/board.jpg", frame)
            # If the frame is not read correctly, stop frame reading (ret==False if frame not red correctly)
            if not ret:
                print("Frame not Read Correctly. Please Check Camera is Plugged in Correctly. Quitting Frame Read")
                # If not read correctly, stop trying (no camera, etc)
                self.is_running = False
                # Close Device (Best Practice)
                self.vid.release()
                # Break Running Loop
                break

            # Set Member variable current frame to frame (to be accessed elsewhere when requested)
            if self.mask is not None:  # If there is a mask
                frame = cv2.bitwise_and(frame, frame, mask=self.mask)
                # set Current Frame Member variable equal to current frame with crop mask
                self.current_frame = frame
            else:  # If there isn't a mask, just set member variable equal to current frame
                self.current_frame = frame
            # Wait 16ms in between frames (a little more than 60fps (ideally))
            cv2.waitKey(16)
            if keyboard.is_pressed('q'):
                self.is_running = False
                # Close Device (Best Practice)
                self.vid.release()
                break
            elif keyboard.is_pressed('r'):  # Clear Mask
                self.mask = None
            elif keyboard.is_pressed('u'):
                self.lines_coords = self.lines_coords[:-1]
                time.sleep(0.1)
                # Save Lines
                self.save_lines()
            elif keyboard.is_pressed('s'):
                if not self.is_timing:
                    self.start_time = datetime.now()
                    self.is_timing = True
                    time.sleep(0.3)
                else:
                    self.start_time = 0
                    self.is_timing = False
                    time.sleep(0.3)
            elif keyboard.is_pressed('d'):
                trial_locations = glob.glob('piece_trials/*')
                trial_locations.sort(key=self.natural_keys)
                if len(trial_locations) > 0:
                    os.remove(trial_locations[-1])
                time.sleep(0.3)


    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]


    # Purpose: Make Left Click Bind to Crop Function
    def prompt_crop(self):
        # While the current frame isn't None -> Then wait an additional 0.3 seconds (make sure loaded)
        while True:
            if self.current_frame.all() is not None:
                break
        time.sleep(0.3)

        # Function linked to left click on frame (through cv2)
        def lc_callback(event, x, y, flags, param):
            # If Left MB Down
            if event == cv2.EVENT_LBUTTONDOWN:
                # Cropping is True
                self.cropping = True
                # Record x,y for start_pos
                self.crop_start = (x, y)
                # Print Crop Start Pos
                print(self.crop_start)
            elif event == cv2.EVENT_LBUTTONUP:
                # Cropping is False
                self.cropping = False
                # Record x, y for end_pos
                self.crop_end = (x, y)
                # Print Crop End Pos
                print(self.crop_end)
                # Update member variable Mask
                self.update_mask()
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Lining is True
                self.lining = True
                # Record x,y for start_pos
                self.line_start = (x, y)
                print(self.line_start)
            elif event == cv2.EVENT_RBUTTONUP:
                # Cropping is False
                self.lining = False
                # Record x, y for end_pos
                self.line_end = (x, y)
                # Print Crop End Pos
                print(self.line_end)
                # Update member variable Mask
                self.lines_coords.append([self.line_start, self.line_end])
                # Reset Line Start and End
                self.line_start = (0, 0)
                self.line_end = (0, 0)
                # Print Line Coords
                print(self.lines_coords)
                # Save Lines
                self.save_lines()

        # Sets the webcam feed window's callback function when lc is pressed to lc_callback
        cv2.setMouseCallback(self.frame_title, lc_callback)

    # Purpose: Update Member Variables for Crop and self.mask
    def update_mask(self):
        # Set the Mask Here
        self.mask = np.zeros(self.current_frame.shape[:2], dtype="uint8")
        # Retrieve Start and End Crops
        (start_x, start_y) = self.crop_start
        (end_x, end_y) = self.crop_end
        # Get the mask rectangle using the start and end positions
        # (255=mask color, -1 = fill it in so it's a mask, not outline)
        cv2.rectangle(self.mask, (start_x, start_y), (end_x, end_y), 255, -1)

    def save_lines(self):
        with open('cached_data/lines.pkl', 'wb') as f:
            pickle.dump(self.lines_coords, f)
    
    def load_lines(self):
        with open('cached_data/lines.pkl', 'rb') as f:
            self.lines_coords = pickle.load(f)
            print(f'Lines Loaded: {self.lines_coords}')