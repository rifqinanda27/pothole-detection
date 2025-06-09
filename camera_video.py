import sys
import cv2 as cv
import time
import os
import geocoder
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class PotholeDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pothole Detection")
        self.setGeometry(100, 100, 960, 720)

        # Folder hasil
        self.result_path = "pothole_coordinates"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Label video
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(900, 600)
        self.video_label.setStyleSheet("background-color: black;")

        # Dropdown untuk camera device
        self.cam_selector = QComboBox()
        self.cam_selector.addItem("Select camera")
        self.detecting = False

        # Tombol
        self.btn_open_video = QPushButton("Open Video")
        self.btn_start_cam = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop")

        # Layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.cam_selector)
        h_layout.addWidget(self.btn_start_cam)
        h_layout.addWidget(self.btn_open_video)
        h_layout.addWidget(self.btn_stop)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label)
        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)

        # Timer untuk loop frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Variables
        self.cap = None
        self.frame_counter = 0
        self.starting_time = None
        self.i = 0
        self.b = 0
        self.location = ["Unknown", "Unknown"]

        # Load YOLO model
        self.net = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
        self.model = cv.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

        # Gunakan CPU saja (OpenCV default)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU backend")

        # Bind buttons
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_detection)

        # Cari kamera yang tersedia
        self.enumerate_cameras()

        # Coba dapatkan lokasi
        try:
            g = geocoder.ip('me')
            if g.latlng:
                self.location = g.latlng
        except:
            pass

    def enumerate_cameras(self):
        # Scan camera index 0-5
        self.cam_selector.clear()
        self.cam_selector.addItem("Select camera")
        for i in range(6):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                self.cam_selector.addItem(f"Camera {i}", i)
                cap.release()

    def open_video(self):
        if self.detecting:
            QMessageBox.warning(self, "Warning", "Stop current detection first.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.cap = cv.VideoCapture(filename)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Cannot open video file!")
                return
            self.start_detection()

    def start_camera(self):
        if self.detecting:
            QMessageBox.warning(self, "Warning", "Stop current detection first.")
            return
        index = self.cam_selector.currentData()
        if index is None:
            QMessageBox.warning(self, "Warning", "Please select a camera!")
            return
        self.cap = cv.VideoCapture(index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open selected camera!")
            return
        self.start_detection()

    def start_detection(self):
        self.frame_counter = 0
        self.starting_time = time.time()
        self.i = 0
        self.b = 0

        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width, self.height = width, height

        fps = self.cap.get(cv.CAP_PROP_FPS)
        if fps == 0 or fps is None or fps != fps:  # cek NaN juga
            fps = 25  # default fallback

        print(f"Camera FPS: {fps}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = os.path.join(self.result_path, f"output_{timestamp}.avi")

        self.result = cv.VideoWriter(
            output_filename,
            cv.VideoWriter_fourcc(*'MJPG'),
            fps,
            (width, height)
        )

        if not self.result.isOpened():
            QMessageBox.critical(self, "Error", "❌ Failed to open video writer.")
            return

        self.detecting = True
        # Timer interval = 1000ms / fps, pakai integer biar gak error
        self.timer.start(int(1000 / fps))


    def stop_detection(self):
        if not self.detecting:
            return
        self.timer.stop()
        self.detecting = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'result') and self.result.isOpened():
            self.result.release()
        QMessageBox.information(self, "Stopped", "Detection stopped, results saved.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        self.frame_counter += 1

        # Detection thresholds
        Conf_threshold = 0.5
        NMS_threshold = 0.4

        classes, scores, boxes = self.model.detect(frame, Conf_threshold, NMS_threshold)

        for (classid, score, box) in zip(classes, scores, boxes):
            label = "pothole"
            x, y, w, h = box
            recarea = w * h
            area = self.width * self.height

            if len(scores) != 0 and scores[0] >= 0.7:
                if (recarea / area) <= 0.1 and y < 600:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label,
                               (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                    # Save image and coordinates every 2 seconds
                    if self.i == 0:
                        cv.imwrite(os.path.join(self.result_path, f'pothole{self.i}.jpg'), frame)
                        with open(os.path.join(self.result_path, f'pothole{self.i}.txt'), 'w') as f:
                            f.write(str(self.location))
                        self.i += 1
                        self.b = time.time()
                    elif time.time() - self.b >= 2:
                        cv.imwrite(os.path.join(self.result_path, f'pothole{self.i}.jpg'), frame)
                        with open(os.path.join(self.result_path, f'pothole{self.i}.txt'), 'w') as f:
                            f.write(str(self.location))
                        self.b = time.time()
                        self.i += 1

        # Show FPS on frame
        ending_time = time.time() - self.starting_time
        fps = self.frame_counter / ending_time if ending_time > 0 else 0
        cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        # Show in QLabel
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

        # Save video frame
        self.result.write(frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PotholeDetector()
    window.show()
    sys.exit(app.exec_())
