import pyaudio
import threading
import numpy as np
import atexit


RATE = 16000
FORMAT = pyaudio.paInt16 #conversion format for PyAudio stream
CHANNELS = 1 #microphone audio channels
CHUNK_SIZE = 8192 #number of samples to take per read
SAMPLE_LENGTH = int(CHUNK_SIZE*1000/RATE) #length of each sample in ms

def open_mic():
        pa = pyaudio.PyAudio()
        stream = pa.open(format = pyaudio.paInt16,
                         channels = CHANNELS,
                         rate = RATE,
                         input = True,
                         frames_per_buffer = CHUNK_SIZE)
        return stream, pa

def get_data(stream, pa):
        input_data = stream.read(CHUNK_SIZE)
        data = np.fromstring(input_data, np.int16)
        return data
    

class Microphone(object):

    def __init__(self, rate=4000,chunksize=1024, channels=1, width=2):
        super(Microphone,self).__init__()
        self.rate = rate
        self.chunksize = chunksize
        self.channels = channels
        self.width = width

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                        format = self.p.get_format_from_width(self.width),
                        channels = self.channels,
                        rate = self.rate,
                        input = True,
                        output = True,
                        stream_callback = self.callback
            )

        self.lock = threading.Lock()
        self.frames = []
        self.stop = False
        atexit.register(self.stop_listening)

##        while self.stream.is_active:
##            time.sleep(0.1)
            
##        self.stream.stop_stream()

    def callback(self,in_data, frame_count, time_info, status):
        data = np.fromstring(in_data, 'int16')
        self.frames.append(data)
        if self.stop:
            return None, pyaudio.paComplete
        return (in_data, pyaudio.paContinue)

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start_listening(self):
        with self.lock:
            self.stop= False
        self.stream.start_stream()

    def stop_listening(self):
        with self.lock:
            self.stop = True
        self.stream.stop_stream()
##        self.p.terminate()
        
   
