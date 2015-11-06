from tps import from_control_points
import os
import shutil
import pdb
import time
# install from https://github.com/olt/thinplatespline

class transformer:
    def __init__(self, initial_points = []):
        self.points = initial_points
        if initial_points:
            self.control_points = from_control_points(self.points)
        else:
            self.control_points = None
    def update(self, new_point):
        self.points.append(new_point)
        self.control_points = from_control_points(self.points)
    def reset(self):
        self.points = []
        self.control_points = None
    def evaluate(self, pair):
        return self.control_points.transform(pair[0], pair[1])
    def to_string(self):
        return str(self.points)

def transformer_from_string(string):
    return transformer(eval(string))

if 'transformer_cache' in os.listdir('.'):
    Transformer = transformer_from_string(open('transformer_cache').read())
else:
    Transformer = transformer()
    open('transformer_cache','w').close()

import shutil

def forever_dump():
    while True:
        time.sleep(5.0)
        s = Transformer.to_string()
        shutil.copy('transformer_cache','cache_bkp')
        open('transformer_cache','w').write(s)

import threading
T = threading.Thread(target = forever_dump)
T.daemon = True
T.start()
