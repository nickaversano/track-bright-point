import cherrypy
import transformers
import cv2
import threading
import sys
import pdb
import numpy as np
import random
import json

def transform(x, y):
    return transformers.Transformer.evaluate([x,y])

STOP_TRACKING = False
TRACKED_POSITION = [-1,-1]

token = lambda:''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for c in range(15)])

def mark(img, location):
    temp = img.copy()
    location = tuple(map(int,location))
    cv2.circle(temp, location, 5, (0,), 20)
    cv2.circle(temp, location, 2, (255,), 10)
    height, width = temp.shape
    cv2.putText(temp, str(location), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    display(temp)

def display(im):
    MULTIPLIER = 1000.
    T = 'x'#token()
    cv2.imshow(T, cv2.resize(im,(int(MULTIPLIER),int(im.shape[0]/(im.shape[1]/MULTIPLIER)))))
    def callback_func(click_type, x, y, mouse_down, _):
        real_mult = im.shape[1]/MULTIPLIER
        if mouse_down >= 1:# mouse is down
            print x*real_mult,y*real_mult
    #cv2.setMouseCallback(T, callback_func)
    #cv2.moveWindow(T, 0, 0)
    if cv2.waitKey(1) & 0xFF == ord('q'):sys.exit(-1)
    #cv2.destroyAllWindows()
    #cv2.waitKey()

def mass_threshold(image, percentage = 0.035):
    ravel = np.sort(np.ravel(image))
    threshold = ravel[ravel.shape[0]*percentage]
    threshold = ravel[ravel.searchsorted(threshold) - 1]
    return cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)[1]

def find_bright_spot(image):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc

ALPHA = 0.5#0.2
BETA = 1.0#0.5
GAMMA = 33

def track_forever():
    # initial set of tracked_position
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_x, max_y = find_bright_spot(gray)#find_bright_spot(gray)
    TRACKED_POSITION[0] = max_x
    TRACKED_POSITION[1] = max_y
    velocities = [0.0,0.0]
    while not STOP_TRACKING:
        ret, frame = cap.read()
        #169, 9, 67
        #150-160
        #15-50
        #80-100
        lower_green = np.array([100, 0, 200])
        upper_green = np.array([255, 255, 255])
        #lower_green = np.array([50, 100, 100])
        #upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(frame, lower_green, upper_green)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        #gray = np.multiply(mass_threshold(gray, 0.99), gray)
        gray = cv2.GaussianBlur(gray, (GAMMA, GAMMA), 0)
        #display(gray)
        max_x, max_y = find_bright_spot(gray)
        new_x_position = TRACKED_POSITION[0]*(1-ALPHA) + max_x*ALPHA
        measured_velocity = new_x_position - TRACKED_POSITION[0]
        velocities[0] = velocities[0]*(1-BETA) + measured_velocity*BETA
        TRACKED_POSITION[0] = new_x_position + velocities[0]
        new_y_position = TRACKED_POSITION[1]*(1-ALPHA) + max_y*ALPHA
        measured_velocity = new_y_position - TRACKED_POSITION[1]
        velocities[1] = velocities[1]*(1-BETA) + measured_velocity*BETA
        TRACKED_POSITION[1] = new_y_position + velocities[1]
        mark(gray, TRACKED_POSITION)
    cap.release()
    cv2.destroyAllWindows()

import cherrypy
class HelloWorld(object):
    def index(self):
        tup = transform(TRACKED_POSITION[0], TRACKED_POSITION[1])
        return str(tup[0]) + ", " + str(tup[1])
    index.exposed = True

if __name__ == '__main__':
    def thread_thing():
        cherrypy.quickstart(HelloWorld())
    threading.Thread(target = thread_thing).start()
    track_forever()
    sys.exit(-1)
    threading.Thread(targ = track_forever).start()
    raw_input()
    STOP_TRACKING = True
