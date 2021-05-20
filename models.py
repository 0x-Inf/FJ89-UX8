from PyQt5.QtCore import Qt
from PyQt5 import QtCore

class SpeechRecognitionTextModel(QtCore.QAbstractListModel):
    def __init__(self, *args, texts=None, **kwargs):
        super(SpeechRecognitionTextModel, self).__init__(*args, **kwargs)
        self.s_texts = texts or []

    def data(self, index, role):
        if role == Qt.DisplayRole:
            text = self.s_texts[index.row()]

            return text

    def rowCount(self, index):
        return len(self.s_texts)

    
