import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
from copy import deepcopy

class WebcamThread(QtCore.QThread):
    frameCaptured = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.blur_value = 0
        self.brightness_value = 0
        self.sharpness_value = 0
        self.hough_value = 0
        self.perspective_value = 0
        self.threshold_value = 0
        self.edge_detection_value = 0
        self.frame = None

    def set_parameters(self, blur, brightness, sharpness, hough, perspective, threshold, edge_detection):
        self.blur_value = blur
        self.brightness_value = brightness
        self.sharpness_value = sharpness
        self.hough_value = hough
        self.perspective_value = perspective
        self.threshold_value = threshold
        self.edge_detection_value = edge_detection

    def run(self):
        cap = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if ret:
                self.frame = frame.copy()

                frame = cv2.GaussianBlur(frame, (self.blur_value * 2 + 1, self.blur_value * 2 + 1), 0)
                frame = cv2.convertScaleAbs(frame, alpha=1 + self.brightness_value / 100.0, beta=0)

                frame = self.apply_hough(frame, self.hough_value)
                frame = self.apply_perspective(frame, self.perspective_value)
                frame = self.apply_threshold(frame, self.threshold_value)
                frame = self.apply_edge_detection(frame, self.edge_detection_value)

                if self.sharpness_value > 0:
                    kernel = np.array([[-1, -1, -1], [-1, 9 + self.sharpness_value / 10.0, -1], [-1, -1, -1]])
                    frame = cv2.filter2D(frame, -1, kernel)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.frameCaptured.emit(qt_image)

        cap.release()

    def apply_hough(self, frame, value):
        if value > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, value)
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame

    def apply_perspective(self, frame, value):
        if value > 0:
            height, width = frame.shape[:2]
            pts1 = np.float32([[50, 50], [width - 50, 50], [50, height - 50], [width - 50, height - 50]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, matrix, (width, height))
        return frame

    def apply_threshold(self, frame, value):
        if value > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        return frame

    def apply_edge_detection(self, frame, value):
        if value > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return frame

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1122, 742)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_main.setObjectName("verticalLayout_main")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.verticalLayout_main.addWidget(self.tabWidget)

        # Camera Tab
        self.camera_tab = QtWidgets.QWidget()
        self.camera_tab.setObjectName("camera_tab")
        self.tabWidget.addTab(self.camera_tab, "Kamera")
        self.webcamThread = WebcamThread()
        self.webcamThread.frameCaptured.connect(self.update_camera)
        self.webcamThread.start()
        # Use a horizontal layout for the camera tab
        self.horizontalLayout_camera_tab = QtWidgets.QHBoxLayout(self.camera_tab)

        # Use a vertical layout for the sliders
        self.verticalLayout_sliders = QtWidgets.QVBoxLayout()

        # Add the sliders to the vertical layout
        self.verticalSlider_blur = self.create_slider("Blur", self.verticalLayout_sliders)
        self.verticalSlider_brightness = self.create_slider("Brightness", self.verticalLayout_sliders)
        self.verticalSlider_sharpness = self.create_slider("Sharpness", self.verticalLayout_sliders)
        self.verticalSlider_hough = self.create_slider("Hough", self.verticalLayout_sliders)
        self.verticalSlider_perspective = self.create_slider("Perspective", self.verticalLayout_sliders)
        self.verticalSlider_threshold = self.create_slider("Threshold", self.verticalLayout_sliders)
        self.verticalSlider_edge_detection = self.create_slider("Edge", self.verticalLayout_sliders)

        # Add the vertical layout with sliders to the horizontal layout of the camera tab
        self.horizontalLayout_camera_tab.addLayout(self.verticalLayout_sliders)

        # Add the camera label to the right side of the horizontal layout
        self.kameragiris = QtWidgets.QLabel(self.camera_tab)
        self.kameragiris.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        self.horizontalLayout_camera_tab.addWidget(self.kameragiris)

        # Add a spacer to push the camera label to the rightmost side
        self.horizontalLayout_camera_tab.addStretch()
        self.horizontalLayout_camera_tab.addSpacing(10)  #
        # Add the save and reset buttons to the horizontal layout
        self.save_photo_button = QtWidgets.QPushButton("Kaydet", self.camera_tab)
        self.save_photo_button.setFixedSize(80, 30)
        self.save_photo_button.clicked.connect(self.save_photo)

        self.reset_effects_button = QtWidgets.QPushButton("Sıfırla", self.camera_tab)
        self.reset_effects_button.setFixedSize(80, 30)
        self.reset_effects_button.clicked.connect(self.reset_effects)

        # Add the camera label to the right side of the horizontal layout
        self.kameragiris = QtWidgets.QLabel(self.camera_tab)
        self.kameragiris.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        # Add the vertical layout with sliders to the horizontal layout of the camera tab
        self.horizontalLayout_camera_tab.addLayout(self.verticalLayout_sliders)

        # Add the camera label to the right side of the horizontal layout
        self.horizontalLayout_camera_tab.addWidget(self.kameragiris)

        # Add a spacer to push the camera label to the rightmost side
        self.horizontalLayout_camera_tab.addStretch()
        self.horizontalLayout_camera_tab.addSpacing(10)

        # Add the save and reset buttons to the horizontal layout
        self.horizontalLayout_camera_tab.addWidget(self.save_photo_button)
        self.horizontalLayout_camera_tab.addWidget(self.reset_effects_button)

        # Set the horizontal layout as the layout for the camera tab
        self.camera_tab.setLayout(self.horizontalLayout_camera_tab)

        # Image Tab
        self.image_tab = QtWidgets.QWidget()
        self.image_tab.setObjectName("image_tab")
        self.tabWidget.addTab(self.image_tab, "Görüntü")

        # Vertical layout for the entire image tab
        self.verticalLayout_image_tab = QtWidgets.QVBoxLayout(self.image_tab)

        # Horizontal layout for loaded image label and sliders
        self.horizontalLayout_image = QtWidgets.QHBoxLayout()

        # Loaded image label
        self.loaded_image_label = QtWidgets.QLabel(self.image_tab)
        self.loaded_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.loaded_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_image.addWidget(self.loaded_image_label, 1)

        # Vertical layout for sliders
        self.verticalLayout_sliders = QtWidgets.QVBoxLayout()

        # Add vertical sliders to the layout
        self.verticalSlider_blur_image = self.create_slider("Blur", self.verticalLayout_sliders)
        self.verticalSlider_brightness_image = self.create_slider("Brightness", self.verticalLayout_sliders)
        self.verticalSlider_sharpness_image = self.create_slider("Sharpness", self.verticalLayout_sliders)
        self.verticalSlider_hough_image = self.create_slider("Hough", self.verticalLayout_sliders)
        self.verticalSlider_perspective_image = self.create_slider("Perspective", self.verticalLayout_sliders)
        self.verticalSlider_threshold_image = self.create_slider("Threshold", self.verticalLayout_sliders)
        self.verticalSlider_edge_detection_image = self.create_slider("Edge", self.verticalLayout_sliders)

        # Add sliders layout to the main horizontal layout
        self.horizontalLayout_image.addLayout(self.verticalLayout_sliders)

        # Add the main horizontal layout to the vertical layout of the image tab
        self.verticalLayout_image_tab.addLayout(self.horizontalLayout_image)

        # Add Yükle button
        self.pushButton_2 = QtWidgets.QPushButton("Yükle", self.image_tab)
        self.pushButton_2.setFixedSize(80, 30)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.load_image)
        self.verticalLayout_image_tab.addWidget(self.pushButton_2)

        # Add Uygula button
        self.pushButton_3 = QtWidgets.QPushButton("Uygula", self.image_tab)
        self.pushButton_3.setFixedSize(80, 30)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.apply_effects_to_loaded_image)
        self.verticalLayout_image_tab.addWidget(self.pushButton_3)

        # Add Sıfırla button
        self.reset_effects_save_button = QtWidgets.QPushButton("Sıfırla", self.image_tab)
        self.reset_effects_save_button.setFixedSize(80, 30)
        self.reset_effects_save_button.clicked.connect(self.reset_effects)
        self.verticalLayout_image_tab.addWidget(self.reset_effects_save_button)

        # Add Kaydet button
        self.save_processed_image_button = QtWidgets.QPushButton("Kaydet", self.image_tab)
        self.save_processed_image_button.setFixedSize(80, 30)
        self.save_processed_image_button.clicked.connect(self.save_processed_image)
        self.verticalLayout_image_tab.addWidget(self.save_processed_image_button)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def save_processed_image(self):
        if self.loaded_image is not None:
            file_dialog = QtWidgets.QFileDialog()
            options = file_dialog.Options()
            options |= file_dialog.DontUseNativeDialog
            file_path, _ = file_dialog.getSaveFileName(self.centralwidget, "Save Processed Image", "",
                                                       "Images (*.png *.jpg)",
                                                       options=options)
            if file_path:
                processed_pixmap = self.loaded_image_label.pixmap().toImage()
                processed_pixmap.save(file_path)
                print(f"Processed image saved to: {file_path}")
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Efekt uygulama"))


    def update_camera(self, image):
        self.kameragiris.setPixmap(QtGui.QPixmap.fromImage(image))

    def create_slider(self, label_text, layout):
        label = QtWidgets.QLabel(label_text)
        verticalSlider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        verticalSlider.setMaximum(100)
        verticalSlider.valueChanged.connect(self.update_parameters)
        verticalSlider.sliderReleased.connect(self.apply_effects_to_camera)
        layout.addWidget(label)
        layout.addWidget(verticalSlider)
        return verticalSlider


    def update_parameters(self):
        blur_value = self.verticalSlider_blur.value()
        brightness_value = self.verticalSlider_brightness.value()
        sharpness_value = self.verticalSlider_sharpness.value()
        hough_value = self.verticalSlider_hough.value()
        perspective_value = self.verticalSlider_perspective.value()
        threshold_value = self.verticalSlider_threshold.value()
        edge_detection_value = self.verticalSlider_edge_detection.value()

        self.webcamThread.set_parameters(
            blur_value, brightness_value, sharpness_value,
            hough_value, perspective_value,
            threshold_value, edge_detection_value
        )

    def apply_effects_to_camera(self):
        # Triggered when any slider is released
        if self.webcamThread.frame is not None:
            self.webcamThread.start()

    def save_photo(self):
        if self.webcamThread.frame is not None:
            file_dialog = QtWidgets.QFileDialog()
            options = file_dialog.Options()
            options |= file_dialog.DontUseNativeDialog
            file_path, _ = file_dialog.getSaveFileName(self.camera_tab, "Save Photo", "", "Images (*.png *.jpg)", options=options)
            if file_path:
                pixmap = QtGui.QPixmap.fromImage(self.kameragiris.pixmap().toImage())
                pixmap.save(file_path)
                print(f"Photo saved to: {file_path}")

    def reset_effects(self):
        self.verticalSlider_blur.setValue(0)
        self.verticalSlider_brightness.setValue(0)
        self.verticalSlider_sharpness.setValue(0)
        self.verticalSlider_hough.setValue(0)
        self.verticalSlider_perspective.setValue(0)
        self.verticalSlider_threshold.setValue(0)
        self.verticalSlider_edge_detection.setValue(0)
    def load_image(self):
        file_dialog = QtWidgets.QFileDialog()
        options = file_dialog.Options()
        options |= file_dialog.DontUseNativeDialog
        file_path, _ = file_dialog.getOpenFileName(self.centralwidget, "Open Image", "", "Images (*.png *.jpg)", options=options)
        if file_path:
            loaded_image = cv2.imread(file_path)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
            qt_image = QtGui.QImage(loaded_image.data, loaded_image.shape[1], loaded_image.shape[0],
                                    QtGui.QImage.Format_RGB888)
            self.loaded_image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            self.loaded_image = loaded_image
            self.original_loaded_image = deepcopy(loaded_image)
            self.webcamThread.frame = None

    def apply_effects_to_loaded_image(self):
        if self.loaded_image is not None:
            self.webcamThread.set_parameters(
                self.verticalSlider_blur_image.value(),
                self.verticalSlider_brightness_image.value(),
                self.verticalSlider_sharpness_image.value(),
                self.verticalSlider_hough_image.value(),
                self.verticalSlider_perspective_image.value(),
                self.verticalSlider_threshold_image.value(),
                self.verticalSlider_edge_detection_image.value()
            )

            processed_image = self.loaded_image.copy()
            processed_image = cv2.GaussianBlur(processed_image, (
                self.webcamThread.blur_value * 2 + 1, self.webcamThread.blur_value * 2 + 1), 0)
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1 + self.webcamThread.brightness_value / 100.0,
                                                  beta=0)

            processed_image = self.webcamThread.apply_hough(processed_image, self.webcamThread.hough_value)
            processed_image = self.webcamThread.apply_perspective(processed_image, self.webcamThread.perspective_value)
            processed_image = self.webcamThread.apply_threshold(processed_image, self.webcamThread.threshold_value)
            processed_image = self.webcamThread.apply_edge_detection(processed_image,
                                                                     self.webcamThread.edge_detection_value)

            if self.webcamThread.sharpness_value > 0:
                kernel = np.array([[-1, -1, -1], [-1, 9 + self.webcamThread.sharpness_value / 10.0, -1], [-1, -1, -1]])
                processed_image = cv2.filter2D(processed_image, -1, kernel)

            qt_image = QtGui.QImage(processed_image.data, processed_image.shape[1], processed_image.shape[0],
                                    QtGui.QImage.Format_RGB888)
            self.loaded_image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

            self.loaded_image = processed_image
            self.loaded_image_label.repaint()

    def reset_effects(self):

        self.verticalSlider_blur.setValue(0)
        self.verticalSlider_brightness.setValue(0)
        self.verticalSlider_sharpness.setValue(0)
        self.verticalSlider_hough.setValue(0)
        self.verticalSlider_perspective.setValue(0)
        self.verticalSlider_threshold.setValue(0)
        self.verticalSlider_edge_detection.setValue(0)

        if hasattr(self, 'original_loaded_image'):
            qt_image = QtGui.QImage(self.original_loaded_image.data, self.original_loaded_image.shape[1],
                                    self.original_loaded_image.shape[0], QtGui.QImage.Format_RGB888)
            self.loaded_image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            self.loaded_image = deepcopy(self.original_loaded_image)
            self.loaded_image_label.repaint()

    def save_loaded_image(self):
        if self.loaded_image is not None:
            file_dialog = QtWidgets.QFileDialog()
            options = file_dialog.Options()
            options |= file_dialog.DontUseNativeDialog
            file_path, _ = file_dialog.getSaveFileName(self.centralwidget, "Save Image", "", "Images (*.png *.jpg)",
                                                       options=options)
            if file_path:
                pixmap = QtGui.QPixmap.fromImage(self.loaded_image_label.pixmap().toImage())
                pixmap.save(file_path)
                print(f"Image saved to: {file_path}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
