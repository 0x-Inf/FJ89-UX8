import os
import sys
import traceback
import time
from multiprocessing import Pool
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
matplotlib.use('Qt5Agg')

import speech_recognition as sr

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from mic import Microphone, open_mic, get_data 
from animated_toggle import AnimatedToggle
from models import SpeechRecognitionTextModel
from audio_preprocessing import Transforms

WIDTH = 2
CHANNELS = 1
RATE = 44100
CHUNKSIZE = 1024
SAMPLES_PER_FRAME = 4

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are

    finished
        No data

    error
        tuple(exctype, value, trcaback.format_exec() )
        
    result
        object data returned from processing, anything
    '''

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)
    speech = pyqtSignal(object)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up

    :param callback: The function callback to run in this worker thread. Supplied args and
                    kwargs will be passed through to the runner
    :type callback: function

    :param args: Arguments to make available to the run code
    :param kwargs: Keywords arguments to make available to the run code
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.is_killed = False
        # store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
##        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['speech_recognition'] = self.signals.speech

    def run(self):
        '''
        Initialize the runner function with passed self.args, self.kwargs
        '''

        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result) # Return the result of the process
        finally:
            self.signals.finished.emit() # Done

        if self.is_killed:
            self.signals.finished.emit()

    def kill(self):
        self.is_killed = True

        

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
##        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow,self).__init__(*args, **kwargs)
##        self.trayIcon = SystemTrayIcon(QIcon("ear-listen.png"),self)
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.worker = None

        self.audio_transforms = Transforms()
        
        self.start_system_tray()

        
        try:
            # Init the model for displaying text on screen
            self.text_model = SpeechRecognitionTextModel()
            self.textView = QListView()
            self.textView.setModel(self.text_model)
        except Exception as err:
            print(err)
        # Create the matplotlib FigureCanvas Object,
        # which defines a single set of axes as self.axes
        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        self.time_amp_axes = self.canvas.figure.add_subplot(211)
        self.freq_amp_axes = self.canvas.figure.add_subplot(212)

        self.canvas_specgram = MplCanvas(self, width=10, height=5, dpi=100)
        self.specgram_axis = self.canvas_specgram.figure.add_subplot(111)

        # Navigation bar for matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        
        sub_widget = QWidget()
        main_widget = QWidget()
        horizontal_layout = QHBoxLayout()
        vertical_layout = QVBoxLayout()

        self.microphone = None

        self.speech_recognition_toggle = AnimatedToggle(
            checked_color = "#FFB000",
            pulse_checked_color = "#50FFB000"
        )
        self.speech_recognition_toggle.setFixedSize(self.speech_recognition_toggle.sizeHint())

        self.recognition_label = QLabel("Speech Recognition")
        
        startButton = QPushButton("Start Listening")
        startButton.clicked.connect(self.start_listening)

        stopButton = QPushButton("Stop Listening")
        stopButton.clicked.connect(self.stop_listening)
        
        vertical_layout.addWidget(startButton)
        vertical_layout.addWidget(stopButton)
        vertical_layout.addWidget(self.recognition_label)
        vertical_layout.addWidget(self.speech_recognition_toggle)
        vertical_layout.addWidget(self.textView)
        vertical_layout.addWidget(self.toolbar)
        vertical_layout.addWidget(self.canvas)
        sub_widget.setLayout(vertical_layout)


        horizontal_layout.addWidget(sub_widget)
        horizontal_layout.addWidget(self.canvas_specgram)
        main_widget.setLayout(horizontal_layout)

        self.setup_spec_plot()
        

        self.setWindowTitle("FJ89-UX8")

        self.speech_recognition_toggle.stateChanged.connect(self.toggle_speech_recognition)

        self.initDataAndGraphs()
        self.timer = QTimer()
        self.setCentralWidget(main_widget)

    def start_system_tray(self):
        menu = QMenu(self)
        actionStartListening = menu.addAction("Start Listening")
        actionStartListening.triggered.connect(self.start_listening)
        menu.addAction(actionStartListening)

        actionStopListening = menu.addAction("Stop Listening")
        actionStopListening.triggered.connect(self.stop_listening)
        menu.addAction(actionStopListening)

        quit = menu.addAction("Quit")
        quit.triggered.connect(self.quit)
        menu.addAction(quit)

        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(QIcon("ear-listen.png"))
                
        self.tray_icon.setContextMenu(menu)
##        self.tray_icon.setVisible(True)
        self.tray_icon.show()
##        self.tray_icon.setToolTip("")

    def initDataAndGraphs(self):
        microphone = Microphone(RATE,CHUNKSIZE,CHANNELS,WIDTH)
        microphone.stop_listening()
        
        self.freq_vect = rfftfreq(microphone.chunksize,
                                         1./microphone.rate)
        self.time_vect = np.arange(microphone.chunksize, dtype = np.float32) / microphone.rate * 1000

        # time amp plot
        self.time_amp_axes.set_xlim(0, self.time_vect.max())
        self.time_amp_axes.set_ylim(-32768, 32768)
        self.time_amp_axes.set_xlabel(u'time (ms)', fontsize=6)
        
        # line Object for time amp
        self.line_time_amp, = self.time_amp_axes.plot(self.time_vect,
                                                      np.ones_like(self.time_vect))


        #freq amp plot
        self.freq_amp_axes.set_xlim(0, self.freq_vect.max())
        self.freq_amp_axes.set_ylim(0, 1)
        self.freq_amp_axes.set_xlabel(u'freq (Hz)', fontsize=6)

        # line object
        self.line_freq_amp, = self.freq_amp_axes.plot(self.freq_vect,
                                                      np.ones_like(self.freq_vect))

    def setup_spec_plot(self):
        spec_microphone = Microphone(RATE,CHUNKSIZE,CHANNELS,WIDTH)
        spec_microphone.stop_listening()
        
        """
        Launch the stream and the original spectrogram
        """
        
        stream, pa = open_mic()
        data = get_data(stream, pa)
        print(data)
        arr2D,freqs,bins = self.audio_transforms.get_specgram(data,16000)

        """Setup the plot parameters"""
        
        extent =(bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
        self.im = self.specgram_axis.imshow(arr2D,aspect='auto',extent = extent,
                                            interpolation="none",cmap='jet',
                                            norm = LogNorm(vmin=.01,vmax=1))
        self.specgram_axis.set_xlabel('Time (s)')
        self.specgram_axis.set_ylabel('Frequency (Hz)')
        self.specgram_axis.set_title('Real Time Spectogram')
##        self.specgram_axis.gca().invert_yaxis()
        anim = animation.FuncAnimation(self.canvas_specgram.figure, self.update_specgram_fig,
                                        blit=False, interval = 8192/1000)
        
        


    def update_specgram_fig(self, n):
        print("running update specgram")
        data = get_data()
        arr2D,freqs,bins = self.audio_transforms.get_specgram(data, 16000)
        spec_im_data = self.im.get_array()
        if n < SAMPLES_PER_FRAME:
            spec_im_data = np.hstack((spec_im_data, arr2D))
            self.im.set_array(spec_im_data)
        else:
            keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
            spec_im_data = np.delete(spec_im_data, np.s_[:-keep_block],1)
            spec_im_data = np.hstack((spec_im_data,arr2D))
            self.im.set_array(spec_im_data)
            
        return self.im




    def toggle_speech_recognition(self,b):
        if b:
            self.spawn_speech_recognition_process()
        else:
           self.end_speech_recognition_process()

    def start_listening(self):
        dialog = QDialog()
        label = QLabel("I am listening")
        try:
            if self.microphone == None:
                self.microphone = Microphone(RATE,CHUNKSIZE,CHANNELS,WIDTH)
                self.microphone.start_listening()
                self.createPlotsAndUpdate()
            else:
                self.microphone.start_listening()
                self.createPlotsAndUpdate()
        except Exception as err:
            print(err)

    def stop_listening(self):
        try:
            self.microphone.stop_listening()
            self.timer.setInterval(0)
            self.timer.stop()
        except Exception as err:
            print(err)
        
    def createPlotsAndUpdate(self):
        if self.timer.isActive():
            pass
        else:
            print(self.timer.interval())
            self.timer.setInterval(1)
            self.timer.timeout.connect(self.handleNewAudioData)
            self.timer.start()

    def spawn_speech_recognition_process(self):
        print("Starting sr worker")
        ## Spawn separate processes for speech recognition
        worker = Worker(self.setup_sr)
        self.worker = worker
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        worker.signals.speech.connect(self.speech_processing)

        #Execute
        self.threadpool.start(worker)

    def end_speech_recognition_process(self):
        self.worker.kill
        
    def speech_processing(self, s):
        self.text_model.s_texts.append(s["transcription"])
        self.text_model.layoutChanged.emit()
##        print(s["transcription"])

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def progress_fn(self, n):
        print("%d%% done" % n)

    def setup_sr(self, speech_recognition):
        
##        for index, name in enumerate(sr.Microphone.list_microphone_names()):
##            print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
        print("Setting up Speech recognition")
##        progress_callback.emit(100/9)

        # set up response object
        response = {
            "success": True,
            "error":None,
            "transcription":None
        }

        
        r = sr.Recognizer()
        def callback(recognizer,audio):   #this is called from the background thread
            try:
                recognizer.dynamic_energy_threshold = True
                print("You said " + recognizer.recognize_google(audio))  # received audio now need to recognize it
                response["transcription"] = recognizer.recognize_google(audio)
                speech_recognition.emit(response)
                return recognizer.recognize_google(audio)
            except LookupError:
                print("Oops! Didn't catch that!")
            except sr.RequestError:
                # API was unreachable or unresponsive
                response["success"] = False
                response["error"] = "API unavailable"
            except sr.UnknownValueError:
                # speech was unintelligible
                response["error"] = "Unable to recognize speech"
        
        r.listen_in_background(sr.Microphone(), callback)

        while True: time.sleep(0.1)    

        
    def handleNewAudioData(self):
        # Gets the latest frames
        frames = self.microphone.get_frames()

        if len(frames) > 0:
            # Keeps only the last frame
            current_frame = frames[-1]
            
            fft_frame = np.fft.rfft(current_frame)
            fft_frame /= np.abs(fft_frame).max()
            #plots the time and amp signal
            try:
                self.line_time_amp.set_data(self.time_vect, current_frame)
                self.canvas.draw()
            except Exception as err:
                print(err)

            #plots the freq and amp
            try:
                self.line_freq_amp.set_data(self.freq_vect, np.abs(fft_frame))
            except Exception as err:
                print(err)
                
                
    
            
        

    def quit(self):
        if app != None:
            app.quit()    

def main():
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    

