import os
from timeit import default_timer as timer
import configparser as cp
import tensorflow as tf
from time import gmtime, strftime

class Session:
    def __init__(self):
        self.current_epoch = 0
        self.current_batch = 0
    def start_timer(self):
        self.duration = timer()

    def stop_timer(self):
        self.duration = str(timer() - self.duration)
        print("Duration : " + self.duration)

    def load_from_file(self, file_config):
        config = cp.ConfigParser()
        if (os.path.exists(file_config)):
            config.read(file_config)
            self.current_epoch = int(config['iteration']['epoch'])
            self.current_batch = int(config['iteration']['batch'])
        else:
            fp = open(file_config,'w')
            config.add_section('iteration')
            config.set('iteration','epoch','0')
            config.set('iteration','batch','0')
            config.write(fp)
            fp.close()

    def now(self):
        return strftime("%Y-%m-%d-%H-%M-%S", gmtime())

session = Session()