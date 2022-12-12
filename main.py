import sys
import os
import json 
import tensorflow as tf
import cv2 
import numpy as np
import time

from gtts import gTTS
from playsound import playsound
from PyQt5.QtWidgets import QApplication
from speech_ui import Ui_MainWindow

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMovie

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from matplotlib import pyplot as plt

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('models/no_yes/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('models/no_yes/ckpt-2').expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

class listen_thread(QThread):
    answer = pyqtSignal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        category_index = label_map_util.create_category_index_from_labelmap('annotations/label_map.pbtxt')
        accurancy_count = 0
        previous_bool = False

        while cap.isOpened(): 
            ret, frame = cap.read()
            image_np = np.array(frame)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            label, _ = viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False)
            
            

            cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

            if label == 'yes':
                if previous_bool:
                    accurancy_count += 1
                else:
                    accurancy_count = 0
                previous_bool = True
                if accurancy_count > 50:
                    self.answer.emit('Evet')
                    accurancy_count = 0
            elif label == 'no':
                if not previous_bool:
                    accurancy_count += 1
                else:
                    accurancy_count = 0
                
                previous_bool = False
                if accurancy_count > 50:
                    self.answer.emit('Hayır')
                    accurancy_count = 0

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

class predict_thread(QThread):
    results = pyqtSignal(list)

    def run(self):
        from predict_new import predict_desease
        list_of_dis = predict_desease()
        self.results.emit(list_of_dis)
        self.exec_()

class ui_windows(QMainWindow):
    def __init__(self):
        super(ui_windows, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.movie = QMovie('resources/rec.gif')
        self.ui.record_label.setMovie(self.movie)
        # self.ui.stackedWidget.setCurrentIndex(1)
        # self.predicted = [0.0001, 0.01, 0.01, 0.01, 0.99, 0.01, 0.008027299613902613, 0.01]
        self.perc_count = 0.0
        self.frame_index = 1
        self.answered_bool = False
        # self.animate_results()

        for i in self.ui.frame.children():
            if i.isWidgetType():
                i.setStyleSheet('''
                QFrame#{frame_name}{
                background-color: #d3e1f7;
                border: 0px solid #d3e1f7;
                border-radius: 10px;
                }
                '''.replace('{frame_name}', i.objectName()))
        

        self.datas = {
            'sicaklik': '36.3',
            'yas': '20',
            'agri': '2',
            'bolge': '0',
            'oksuruk': '2',
            'kilogram': '55',
            'boy': '170',
            'sector': 'isci'
        }
        self.questions = [
            'Burun akıntısı var mı?',
            'Geniz akıntısı var mı?',
            'Göz kızarıklıgınız var mı?',
            'Hastalık sürecinde bilinç kaybı yaşadınız mı?',
            'Kolesterol hastalığınız var mı?',
            'Şeker hastalığınız var mı?',
            'Tat ve kok kaybınız var mı?',
            'Hasta sigara kullanıyor mu?',
            'Hasta cinsiyetini söyleyin.',
            'Alkol kullanıyor musunuz?',
            'Sonuçları doktora gönderilecektir.'
        ]

        self.keys = [
            'burun',
            'geniz_akinti',
            'goz',
            'bilinc',
            'kalestrol',
            'seker_hasta',
            'tat_koku',
            'sigara',
            'cinsyet',
            'alkol',
            'sector'
        ]

        self.pictures = [
            'sneeze',
            'nasal',
            'conjunctivitis',
            'loss-of-consciousness',
            'fat',
            'glucose-meter',
            'loss-of-sense-of-taste',
            'no-smoking',
            'equality',
            'no-drinks',
            'engineers',
            'doctor'
        ]

        self.question_index = 0


        QTimer.singleShot(1000, lambda: self.ask(self.questions[self.question_index]))
        self.start_listen()
    
    def speak(self, txt):
        speech = gTTS(text=txt, lang='tr', slow=False)
        speech.save('output.mp3')
        playsound('output.mp3', False)
        os.remove('output.mp3')

        if self.question_index != len(self.questions) - 1:
            QTimer.singleShot(self.delay_time(), self.set_answered)

        else:
            with open('questions.json', 'w') as outputfile:
                json.dump(self.datas, outputfile, indent=4)
            self.start_predict()
    
    def set_answered(self):
        self.answered_bool = False

    
    def ask(self, qsn):
        self.ui.question.setText(qsn)
        self.ui.pic_label.setStyleSheet('''
        image: url(resources/{pic}.png);
        '''.replace("{pic}", self.pictures[self.question_index]))
        self.ui.answer.setText('')
        self.ui.record_label.hide()
        self.movie.stop()
        self.speak(qsn)
        self.question_index += 1
    
    def answered(self, value):
        if not self.answered_bool:
            self.answered_bool = True
            if value == 'false':
                self.question_index -= 1
                QTimer.singleShot(1000, lambda: self.ask(self.questions[self.question_index]))
            
            else:
                if self.question_index < len(self.questions):
                    self.ans = value
                    self.ui.answer.setText(self.ans)
                    self.data_collection(self.keys[self.question_index - 1], self.ans)
                    QTimer.singleShot(1000, lambda: self.ask(self.questions[self.question_index]))
    
    def data_collection(self, key, value):
        if value == 'Evet' or value == 'var' or value == 'erkek':
            value = '1'
        if value == 'Hayır' or value == 'yok' or value == 'kadın':
            value = '0'
        if value == 'Baş':
            value = 'bas'
        if value == 'sırt':
            value = 'sirt'
        if value == 'öğrenci':
            value = 'ogrenci'
        if value == 'Tarım':
            value = 'tarım'
        if value == 'işçi':
            value = 'isci'
        if value == 'bir':
            value = '1'
        if self.question_index == 1:
            try:
                float(value)
            except:
                value = '18'
            
        if self.question_index != 4 and self.question_index != 17:
            try:
                float(value)
            except:
                value = '0'
        
        if self.question_index == 15:
            if value == '0' or value == '1':
                value = '50'
        if self.question_index == 16:
            if value == '0' or value == '1':
                value = '150'
        
        if self.question_index == 4:
            if value not in ['bacak', 'bel', 'genel', 'bas', 'sirt']:
                value = '0'
        
        if self.question_index == 17:
            if value not in ['emekli', 'hizmet', 'tarım', 'ogrenci', 'isci']:
                value = 'isci'

        self.datas[key] = value
        print(self.datas)
    
    def start_listen(self):
        self.listen_thread = listen_thread()
        self.listen_thread.answer.connect(self.answered)
        self.listen_thread.start()
        self.ui.record_label.show()
        self.movie.start()
    
    def delay_time(self):
        return len(self.questions[self.question_index]) * 95
    
    def start_predict(self):
        self.predict_thread = predict_thread()
        self.predict_thread.results.connect(self.results)
        self.predict_thread.start()
    
    def results(self, value):
        self.predicted = value
        self.animate_results()
        self.ui.stackedWidget.setCurrentIndex(1)
        print(value)
    
    def animate_results(self):
        if self.frame_index <= len(self.predicted):
            pers = self.predicted[self.frame_index - 1]
            frames = self.ui.frame.children()
            if self.frame_index < len(frames):
                frames[self.frame_index].setStyleSheet('''
                QFrame#{frame_name}{
                background-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:{STOP_2} #6d97ff, stop:{STOP_1} #d3e1f7);
                border: 0px solid #d3e1f7;
                border-radius: 10px;
                }
                '''.replace('{frame_name}', frames[self.frame_index].objectName()).replace('{STOP_1}', str(self.perc_count)).replace('{STOP_2}', str(self.perc_count - 0.01)))

                frames[self.frame_index].children()[2].setText(f'%{round((self.perc_count * 100), 1)}')

                QTimer.singleShot(10, self.animate_results)
                if self.perc_count >= pers:
                    self.frame_index += 1
                    self.perc_count = 0.0
                self.perc_count += 0.01


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ui_windows()

    win.show()
    sys.exit(app.exec_())