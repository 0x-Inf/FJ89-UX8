class SystemTrayIcon(QSystemTrayIcon):

    def __init__(self, icon, parent=None):
        super(SystemTrayIcon,self).__init__(icon, parent)

        menu = QMenu(parent)
        actionStartListening = QAction("Start Listening")
        actionStartListening.triggered.connect(self.start_listening)
        menu.addAction(actionStartListening)

        quit = QAction("Quit")
        quit.triggered.connect(self.quit)
        menu.addAction(quit)

        self.setContextMenu(menu)
        self.setVisible(True)
        self.show()

    def start_listening(self):
        dialog = QDialog()
        label = QLabel("I am listening")
        microphone = Microphone(RATE,CHUNKSIZE,CHANNELS,WIDTH)
        microphone.start_listening()

    def quit(self):
        if app is not None:
            app.quit()
        print()
