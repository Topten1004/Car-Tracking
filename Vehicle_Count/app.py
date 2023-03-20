from pickle import FALSE
from turtle import position
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from gui import *
import copy
from counter import CounterThread
from utils.sort import *
from models import *
from utils.utils import *
from utils.datasets import *
from config import *
from functools import partial
from shapely.geometry import LineString

class App(QMainWindow,Ui_mainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.setupUi(self)
        self.label_image_size = (self.label_image.geometry().width(),self.label_image.geometry().height())
        self.video = None
        self.exampleImage = None
        self.imgScale = None
        self.countLane = {}
        self.PB_lane_flag = 0
        self.running_flag = 0
        self.counter_thread_start_flag = 0
        for item in Lane_name:
            vars(self)[f"countLane['{item}']"] = []
            vars(self)[f"get_points_flag['{item}']"] = 0
        
        #button function
        self.pushButton_openVideo.clicked.connect(self.open_video)
        self.pushButton_start.clicked.connect(self.start_count)
        self.label_image.mouseDoubleClickEvent = self.get_points

        for item in Lane_name:            
            vars(self)[f"PB_{item}"].clicked.connect(partial(self.select_area, item))
            vars(self)[f"PB_{item}"].setEnabled(False)
        self.pushButton_start.setEnabled(False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_config = "config/coco.data"
        weights_path = "weights/yolov.weights"
        model_def = "config/yolov.cfg"
        data_config = parse_data_config(data_config)
        self.yolo_class_names = load_classes(data_config["names"])

        # Initiate model
        print("Loading model ...")
        self.yolo_model = Darknet(model_def).to(self.device)
        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.yolo_model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.yolo_model.load_state_dict(torch.load(weights_path))


        # counter Thread
        self.counterThread = CounterThread(self.yolo_model,self.yolo_class_names,self.device)
        self.counterThread.sin_counterResult.connect(self.show_image_label)
        self.counterThread.sin_done.connect(self.done)

    def open_video(self):
        for item in Lane_name:
            vars(self)[f"countLane['{item}']"] = []
            vars(self)[f"get_points_flag['{item}']"] = 0
        openfile_name = QFileDialog.getOpenFileName(self,'Open video','','Video files(*.avi , *.mp4)')
        self.videoList = [openfile_name[0]]

        vid = cv2.VideoCapture(self.videoList[0])
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break

        for item in Lane_name:
            vars(self)[f"PB_{item}"].setEnabled(True)
        self.pushButton_start.setText("Start")
        self.pushButton_start.setEnabled(False)

        #clear counting results
        KalmanBoxTracker.count = 0

    def get_points(self, event):
        exampleImageWithArea = copy.deepcopy(self.exampleImage)
        for item in Lane_name:
            if len(vars(self)[f"countLane['{item}']"])>1:
                    for i in range(len(vars(self)[f"countLane['{item}']"])):
                        cv2.line(exampleImageWithArea, tuple(vars(self)[f"countLane['{item}']"][i]), tuple(vars(self)[f"countLane['{item}']"][(i + 1) % (len(vars(self)[f"countLane['{item}']"]))]), (255, 0, 0), 1)
                    cv2.putText(exampleImageWithArea, item, tuple(vars(self)[f"countLane['{item}']"][0]), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0, 0, 255), thickness=1)
        for item in Lane_name:
            if vars(self)[f"get_points_flag['{item}']"]:
                x = event.x()
                y = event.y()
                vars(self)[f"countLane['{item}']"].append([int(x*self.imgScale[1]),int(y*self.imgScale[0])])
                for point in vars(self)[f"countLane['{item}']"]:
                    exampleImageWithArea[point[1]-2:point[1]+2,point[0]-2:point[0]+2] = (0,255,0)
                cv2.fillConvexPoly(exampleImageWithArea, np.array(vars(self)[f"countLane['{item}']"]), (0,255,25))
                self.show_image_label(exampleImageWithArea)
                print(item, vars(self)[f"countLane['{item}']"])

    def select_area(self, lane_name):
        #change Area needs update exampleImage
        if self.counter_thread_start_flag:
            ret, frame = self.videoCapture.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
        
        if not self.PB_lane_flag:
            for item in Lane_name:
                vars(self)[f"PB_{item}"].setEnabled(False)
            vars(self)[f"PB_{lane_name}"].setEnabled(True)
        else:
            for item in Lane_name:
                vars(self)[f"PB_{item}"].setEnabled(True)
        
        if not self.PB_lane_flag: # button flag 1 or 0
            self.PB_lane_flag = 1
        else:
            self.PB_lane_flag = 0

        linestring_data = None
        if not vars(self)[f"get_points_flag['{lane_name}']"]:
            vars(self)[f"PB_{lane_name}"].setText(f"Submit")
            vars(self)[f"get_points_flag['{lane_name}']"] = 1
            vars(self)[f"countLane['{lane_name}']"] = []
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_start.setEnabled(False)

        else:
            vars(self)[f"PB_{lane_name}"].setText(lane_name)
            vars(self)[f"get_points_flag['{lane_name}']"] = 0
            exampleImage = copy.deepcopy(self.exampleImage)
            # painting area
            for item in Lane_name:
                if len(vars(self)[f"countLane['{item}']"])>1:
                    for i in range(len(vars(self)[f"countLane['{item}']"])):
                        cv2.line(exampleImage, tuple(vars(self)[f"countLane['{item}']"][i]), tuple(vars(self)[f"countLane['{item}']"][(i + 1) % (len(vars(self)[f"countLane['{item}']"]))]), (255, 0, 0), 1)
                    cv2.putText(exampleImage, item, tuple(vars(self)[f"countLane['{item}']"][0]), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0, 255, 0), thickness=1)
                    linestring_data = LineString(vars(self)[f"countLane['{item}']"])
            self.show_image_label(exampleImage)
            #enable start button
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(True)

    def show_image_label(self, img_np):
        img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.label_image_size)
        frame = QImage(img_np, self.label_image_size[0], self.label_image_size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.label_image.setPixmap(pix)
        self.label_image.repaint()

    def start_count(self):
        if self.running_flag == 0:
            #clear count and display
            KalmanBoxTracker.count = 0
            #start
            self.running_flag = 1
            self.pushButton_start.setText("Pause")
            self.pushButton_openVideo.setEnabled(False)
            for item in Lane_name:
                vars(self)[f"PB_{item}"].setEnabled(False)
            #emit new parameter to counter thread
            for item in Lane_name:
                self.countLane[f'{item}'] = vars(self)[f"countLane['{item}']"]

            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.counterThread.sin_countLane.emit(self.countLane)
            self.counterThread.sin_videoList.emit(self.videoList)
            #start counter thread
            self.counterThread.start()


        elif self.running_flag == 1:  #push stop button
            #stop system
            self.running_flag = 0
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.pushButton_openVideo.setEnabled(True)
            for item in Lane_name:
                vars(self)[f"PB_{item}"].setEnabled(True)
            self.pushButton_start.setText("Start")
                

    def done(self,sin):
        if sin == 1:
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(False)
            self.pushButton_start.setText("Start")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())
