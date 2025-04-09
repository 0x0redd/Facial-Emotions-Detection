import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QStackedLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QTransform, QMovie
from tensorflow.keras.models import load_model

# Load model and emotion mappings
model = load_model("emotiondetector.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        self.setFixedSize(1200, 900)
        self.setWindowTitle("Facial Emotion Detection")

        # Set background
        palette = QPalette()
        bg_image = QPixmap("UIfacialapp.jpg")
        palette.setBrush(QPalette.Window, QBrush(bg_image))
        self.setPalette(palette)

        # Camera view
        self.image_label = QLabel(self)
        self.image_label.setGeometry(335, 240, 530, 420)
        self.image_label.setStyleSheet("background-color: transparent;")

        # Result label
        self.result_label = QLabel("Emotion: 🤔", self)
        self.result_label.move(513, 693)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: transparent;")

        # Emoji visuals
        self.reaction_label1 = QLabel(self)
        self.reaction_label1.setGeometry(50, 90, 200, 200)
        self.reaction_label1.setStyleSheet("background-color: transparent;")
        self.reaction_label1.setVisible(False)

        self.reaction_label2 = QLabel(self)
        self.reaction_label2.setGeometry(970, 90, 200, 200)
        self.reaction_label2.setStyleSheet("background-color: transparent;")
        self.reaction_label2.setVisible(False)

        # Camera init
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        emotion_text = "Emotion: 🤔"
        emotion_image_filename = None

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            roi = cv2.resize(face, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            try:
                prediction = model.predict(roi, verbose=0)
                label_index = np.argmax(prediction)
                label = emotion_labels[label_index]
                emoji = emotion_emojis.get(label, '❓')
                emotion_text = f"{emoji} {label}"
                emotion_image_filename = f"{label.lower()}.png"
            except Exception as e:
                print("Prediction error:", e)
                emotion_text = "🤔 Error"
                emotion_image_filename = None

            break  # Only process the first detected face

        self.result_label.setText(emotion_text)

        if emotion_image_filename and os.path.exists(emotion_image_filename):
            original_pixmap = QPixmap(emotion_image_filename).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            rotated_pixmap1 = original_pixmap.transformed(QTransform().rotate(0), Qt.SmoothTransformation)
            rotated_pixmap2 = original_pixmap.transformed(QTransform().rotate(0), Qt.SmoothTransformation)
            self.reaction_label1.setPixmap(rotated_pixmap1)
            self.reaction_label1.setVisible(True)
            self.reaction_label2.setPixmap(rotated_pixmap2)
            self.reaction_label2.setVisible(True)
        else:
            self.reaction_label1.setVisible(False)
            self.reaction_label2.setVisible(False)

        frame_resized = cv2.resize(frame, (530, 420))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()

class IntroWidget(QWidget):
    def __init__(self, movie_path, switch_callback, duration=3000):  # duration in ms
        super().__init__()
        self.setFixedSize(1200, 900)
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie(movie_path)
        self.movie.setScaledSize(QSize(1200, 900))
        self.label.setMovie(self.movie)
        self.layout.addWidget(self.label)

        # Timer to switch after X milliseconds
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(switch_callback)

        self.movie.start()
        self.timer.start(duration)  # For example, 3000ms = 3s


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Emotion Detection App")
        self.setFixedSize(1200, 900)

        self.stack = QStackedLayout()
        self.setLayout(self.stack)

        self.intro = IntroWidget("intro.gif", self.show_main_app)
        self.stack.addWidget(self.intro)

        self.emotion_app = None  # Delay init

    def show_main_app(self):
        if not self.emotion_app:
            self.emotion_app = EmotionApp()
            self.stack.addWidget(self.emotion_app)
        self.stack.setCurrentWidget(self.emotion_app)

if __name__ == "__main__":
    import PyQt5.QtCore
    PyQt5.QtCore.QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
