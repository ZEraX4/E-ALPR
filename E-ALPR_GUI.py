import os
import sys
from collections import Counter

import cv2
import joblib
import numpy as np
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QSettings, QThread, Signal
from PySide2.QtGui import QMouseEvent, QImage, QPixmap
from PySide2.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMainWindow
from qimage2ndarray import array2qimage

alphabet = {
    "a": "أ", "b": "ب", "t": "ت", "th": "ث", "g": "ج", "hh": "ح", "kh": "خ", "d": "د", "the": "ذ",
    "r": "ر", "z": "ز", "c": "س", "sh": "ش", "s": "ص", "dd": "ض", "tt": "ط", "zz": "ظ", "i": "ع",
    "gh": "غ", "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و",
    "y": "ي", "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥", "6": "٦", "7": "٧",
    "8": "٨", "9": "٩"
}
YoloClasses = ['LP']
classes = list(alphabet.keys())

MODEL = None


class LineEdit(QtWidgets.QLineEdit):
    def mouseDoubleClickEvent(self, arg__1: QMouseEvent):
        self.setText(QFileDialog.getOpenFileName(self,
                                                 "Open Image", os.getcwd(),
                                                 "Image Files (*.png *.jpg *.mp4);;"
                                                 "Model Files (*.cfg *.h5 *.pkl *.tflite *.backup)")[0])


class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.setObjectName("Dialog")
        self.resize(845, 453)
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_2.addWidget(self.graphicsView)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.imageCheck = QtWidgets.QRadioButton(self.groupBox)
        self.imageCheck.setObjectName("imageCheck")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.imageCheck)
        self.imageEdit = LineEdit(self.groupBox)
        self.imageEdit.setEnabled(False)
        self.imageEdit.setObjectName("imageEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.imageEdit)
        self.videoCheck = QtWidgets.QRadioButton(self.groupBox)
        self.videoCheck.setObjectName("videoCheck")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.videoCheck)
        self.videoEdit = LineEdit(self.groupBox)
        self.videoEdit.setEnabled(False)
        self.videoEdit.setObjectName("videoEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.videoEdit)
        self.cameraCheck = QtWidgets.QRadioButton(self.groupBox)
        self.cameraCheck.setObjectName("cameraCheck")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.cameraCheck)
        self.cameraEdit = QtWidgets.QLineEdit(self.groupBox)
        self.cameraEdit.setEnabled(False)
        self.cameraEdit.setObjectName("cameraEdit")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.cameraEdit)
        self.tfModelLabel = QtWidgets.QLabel(self.groupBox)
        self.tfModelLabel.setObjectName("tfModelLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.tfModelLabel)
        self.tfModelEdit = LineEdit(self.groupBox)
        self.tfModelEdit.setObjectName("tfModelEdit")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.tfModelEdit)
        self.yoloWeightLabel = QtWidgets.QLabel(self.groupBox)
        self.yoloWeightLabel.setObjectName("yoloWeightLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.yoloWeightLabel)
        self.yoloWeightsEdit = LineEdit(self.groupBox)
        self.yoloWeightsEdit.setObjectName("yoloWeightsEdit")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.yoloWeightsEdit)
        self.yoloConfLabel = QtWidgets.QLabel(self.groupBox)
        self.yoloConfLabel.setObjectName("yoloConfLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.yoloConfLabel)
        self.yoloConfEdit = LineEdit(self.groupBox)
        self.yoloConfEdit.setObjectName("yoloConfEdit")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.yoloConfEdit)
        self.verticalLayout.addLayout(self.formLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.predictButton = QtWidgets.QPushButton(self.groupBox)
        self.predictButton.setObjectName("predictButton")
        self.horizontalLayout.addWidget(self.predictButton)
        self.stopButton = QtWidgets.QPushButton(self.groupBox)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.outLabel = QtWidgets.QLabel(self.groupBox)
        self.outLabel.setMinimumSize(QtCore.QSize(300, 100))
        font = QtGui.QFont()
        font.setFamily("Simplified Arabic")
        font.setPointSize(36)
        font.setWeight(75)
        font.setBold(True)
        self.outLabel.setFont(font)
        self.outLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.outLabel.setLineWidth(1)
        self.outLabel.setScaledContents(True)
        self.outLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.outLabel.setObjectName("outLabel")
        self.verticalLayout.addWidget(self.outLabel)
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox)
        self.tableWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.verticalLayout.addWidget(self.tableWidget)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.setCentralWidget(self.centralWidget)

        self.readSetting()
        self.thread = None

        self.retranslateUi(self)
        QtCore.QObject.connect(self.imageCheck, QtCore.SIGNAL("toggled(bool)"), self.imageEdit.setEnabled)
        QtCore.QObject.connect(self.videoCheck, QtCore.SIGNAL("toggled(bool)"), self.videoEdit.setEnabled)
        QtCore.QObject.connect(self.cameraCheck, QtCore.SIGNAL("toggled(bool)"), self.cameraEdit.setEnabled)
        QtCore.QObject.connect(self.predictButton, QtCore.SIGNAL("clicked()"), self.predict)
        QtCore.QObject.connect(self.stopButton, QtCore.SIGNAL("clicked()"), self.stop)
        QtCore.QObject.connect(self.tfModelEdit, QtCore.SIGNAL("editingFinished()"), resetModel)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtWidgets.QApplication.translate("Dialog", "E-ALPR", None, -1))
        self.groupBox_2.setTitle(QtWidgets.QApplication.translate("Dialog", "Frame", None, -1))
        self.groupBox.setTitle(QtWidgets.QApplication.translate("Dialog", "Tools", None, -1))
        self.imageCheck.setText(QtWidgets.QApplication.translate("Dialog", "Image", None, -1))
        self.videoCheck.setText(QtWidgets.QApplication.translate("Dialog", "Video", None, -1))
        self.cameraCheck.setText(QtWidgets.QApplication.translate("Dialog", "Camera", None, -1))
        self.tfModelLabel.setText(QtWidgets.QApplication.translate("Dialog", "Tensorflow Model", None, -1))
        self.yoloWeightLabel.setText(QtWidgets.QApplication.translate("Dialog", "YOLO Weights", None, -1))
        self.yoloConfLabel.setText(QtWidgets.QApplication.translate("Dialog", "YOLO Config", None, -1))
        self.predictButton.setText(QtWidgets.QApplication.translate("Dialog", "Predict", None, -1))
        self.stopButton.setText(QtWidgets.QApplication.translate("Dialog", "Stop", None, -1))
        self.outLabel.setText(QtWidgets.QApplication.translate("Dialog", "OUT", None, -1))
        self.tableWidget.horizontalHeaderItem(0).setText(
            QtWidgets.QApplication.translate("Dialog", "Car Owner", None, -1))
        self.tableWidget.horizontalHeaderItem(1).setText(QtWidgets.QApplication.translate("Dialog", "Car ID", None, -1))
        self.tableWidget.horizontalHeaderItem(2).setText(QtWidgets.QApplication.translate("Dialog", "Date", None, -1))

    def predict(self):
        self.progressBar.setValue(0)
        tfModel = self.tfModelEdit.text()
        yoloConf = self.yoloConfEdit.text()
        yoloWeights = self.yoloWeightsEdit.text()
        if self.imageCheck.isChecked() and os.path.exists(self.imageEdit.text()):
            cap = cv2.VideoCapture(self.imageEdit.text())
        elif self.videoCheck.isChecked() and os.path.exists(self.videoEdit.text()):
            cap = cv2.VideoCapture(self.videoEdit.text())
        elif self.cameraCheck.isChecked():
            cap = cv2.VideoCapture(self.cameraEdit.text())
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Check the paths.")
            return

        net = cv2.dnn.readNetFromDarknet(yoloConf, yoloWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        if self.cameraCheck.isChecked():
            total = -1
        else:
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.progressBar.setMaximum(total)

        self.thread = Thread(cap, net, tfModel)
        self.thread.change.connect(self.change)
        self.thread.finished.connect(self.predictButton.setEnabled)
        self.thread.frame_done.connect(self.inc)
        self.predictButton.setEnabled(False)
        self.thread.start()

    def inc(self):
        self.progressBar.setValue(self.progressBar.value() + 1)

    def stop(self):
        self.thread.exiting = True

    def change(self, p, out):
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap.fromImage(p.scaled(self.graphicsView.size() - QtCore.QSize(2, 2))))
        self.graphicsView.setScene(scene)
        if out:
            self.outLabel.setText(out[::-1])
            rows = self.tableWidget.rowCount()
            self.tableWidget.setRowCount(rows + 1)
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(rows, item)
            self.tableWidget.verticalHeaderItem(rows).setText(str(rows))
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setItem(rows, 0, item)
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setItem(rows, 1, item)
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setItem(rows, 2, item)
            self.tableWidget.item(rows, 0).setText("N/A")  # TODO: get the name from a database
            self.tableWidget.item(rows, 1).setText(out[::-1])
            self.tableWidget.item(rows, 2).setText(QtCore.QDateTime().currentDateTime().toString())
            self.tableWidget.scrollToBottom()

    def readSetting(self):
        settings = QSettings("ZEraX", "E-ALPR")
        settings.beginGroup("Config")
        self.tfModelEdit.setText(settings.value("tfModel").__str__())
        self.yoloConfEdit.setText(settings.value("yoloConf").__str__())
        self.yoloWeightsEdit.setText(settings.value("yoloWeights").__str__())
        self.imageEdit.setText(settings.value("image").__str__())
        self.videoEdit.setText(settings.value("video").__str__())
        self.cameraEdit.setText(settings.value("camera").__str__())
        settings.endGroup()

    def closeEvent(self, arg__1: QtGui.QCloseEvent):
        settings = QSettings("ZEraX", "E-ALPR")
        settings.beginGroup("Config")
        settings.setValue("tfModel", self.tfModelEdit.text())
        settings.setValue("yoloConf", self.yoloConfEdit.text())
        settings.setValue("yoloWeights", self.yoloWeightsEdit.text())
        settings.setValue("image", self.imageEdit.text())
        settings.setValue("video", self.videoEdit.text())
        settings.setValue("camera", self.cameraEdit.text())
        settings.endGroup()


class Thread(QThread):
    change = Signal(QImage, str)
    frame_done = Signal()
    finished = Signal(bool)

    def __init__(self, cap, net, tfModel):
        super(Thread, self).__init__()
        self.exiting = False
        self.cap = cap
        self.net = net
        self.model = tfModel

    def run(self):
        while self.cap and not self.exiting:
            out = None
            ret, frame = self.cap.read()
            if not ret:
                break
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(getOutputsNames(self.net))
            rec, img = postprocess(frame, outs, .5, .5)
            if rec and img is not None:
                out = predict_image(img, self.model)
            p = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p = array2qimage(p)
            self.change.emit(p, out)
            self.frame_done.emit()
        self.finished.emit(True)


# Get the names of the output layers
def getOutputsNames(n):
    # Get the names of all the layers in the network
    layersNames = n.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in n.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(fr, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(fr, (left, top), (right, bottom), (255, 255, 255), 3)
    # Get the label for the class name and its confidence
    lab = '%s:%.2f' % (YoloClasses[classId], conf)
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(fr, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (0, 0, 255), cv2.FILLED)
    cv2.putText(fr, lab, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(fr, outs, confT, nmsT):
    frameHeight = fr.shape[0]
    frameWidth = fr.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for o in outs:
        for detection in o:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confT:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confT, nmsT)
    cropped = None
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = max(box[0], 0)
        top = max(box[1], 0)
        width = max(box[2], 0)
        height = max(box[3], 0)
        if height > width:
            continue
        cropped = fr[top:(top + height), left:(left + width)]
        drawPred(fr, classIds[i], confidences[i], left, top, left + width, top + height)

    return len(indices) > 0, cropped


def resetModel():
    global MODEL
    MODEL = None


def square(img):
    assert type(img) == np.ndarray
    d, r = divmod(abs(img.shape[0] - img.shape[1]), 2)
    if img.shape[0] > img.shape[1]:
        return cv2.copyMakeBorder(img, 0, 0, d if not r else d + 1, d, cv2.BORDER_CONSTANT, 0)
    else:
        return cv2.copyMakeBorder(img, d if not r else d + 1, d, 0, 0, cv2.BORDER_CONSTANT, 0)


def mark(img):
    chars = {}
    digits = []

    copy = img.copy()
    t1, t2, _ = img.shape
    gray = cv2.inRange(img, (0, 0, 0), (150, 70, 255))

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=3 if t1 > 200 else 1)

    cnt, he = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    k = [key for (key, value) in Counter([x[3] for x in he[0]]).items() if value >= 5]
    for r in k:
        for i, v in enumerate(cnt):
            if he[0][i][3] == r:
                x, y, w, h = cv2.boundingRect(v)
                if .5 * t1 > h > .1 * t1 and .3 * t2 > w > .01 * t2:
                    chars[x] = opening[y:y + h, x:x + w]
                    cv2.rectangle(copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        if len(chars) < 5:
            chars = {}
            copy = img.copy()
        else:
            break
    if len(chars) >= 1:
        for i, key in enumerate(sorted(chars.keys())):
            digits.append(chars[key])
    return digits


def predict_image(image, modelFile):
    prediction = {}
    out = ''
    digits = mark(image)
    t = modelFile.split('.')[-1] == 'pkl'

    global MODEL
    if t:
        if MODEL is None:
            MODEL = joblib.load(modelFile)
        for i in range(len(digits)):
            prediction[i] = {}
            resized = cv2.resize(square(digits[i]), (40, 40))
            result = MODEL.predict([np.array(resized).ravel()])
            out += (alphabet[classes[result[0]]] + ' ')
    else:
        from tensorflow import lite
        if MODEL is None:
            MODEL = lite.Interpreter(model_path=modelFile)
            MODEL.allocate_tensors()
        for i in range(len(digits)):
            prediction[i] = {}
            resized = cv2.resize(square(digits[i]), (40, 40))
            resized = np.expand_dims(np.array(resized, dtype='f').ravel(), axis=0) / 255.0
            # Get input and output tensors.
            input_details = MODEL.get_input_details()
            output_details = MODEL.get_output_details()

            # Test the TensorFlow Lite model on random input data.
            MODEL.set_tensor(input_details[0]['index'], resized)

            MODEL.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            result = MODEL.get_tensor(output_details[0]['index'])
            prediction[i][classes[int(np.argmax(result))]] = float(np.max(result) * 100)
            out += (alphabet[classes[int(np.argmax(result))]] + ' ')
    return out


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    form = Form()
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
