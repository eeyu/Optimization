from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget

class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        
        self.verticalLayout = QVBoxLayout(self)
        self.horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.horizontalLayout.addWidget(self.label)
        
        # spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # self.horizontalLayout.addItem(spacerItem)
        
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setTickInterval(1)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.x = int(self.x)
        self.label.setText("{0:.4g}".format(self.x))