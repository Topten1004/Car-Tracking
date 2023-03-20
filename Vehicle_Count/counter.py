# coding:utf-8
from enum import unique
import cv2
from utils.sort import *
from PyQt5.QtCore import  QThread, pyqtSignal
import predict
from config import *
import pandas as pd

class CounterThread(QThread):
    sin_counterResult = pyqtSignal(np.ndarray)
    sin_runningFlag = pyqtSignal(int)
    sin_videoList = pyqtSignal(list)
    sin_countLane = pyqtSignal(dict)
    sin_done = pyqtSignal(int)
    sin_counter_results = pyqtSignal(list)

    def __init__(self, model,class_names, device):
        super(CounterThread,self).__init__()  
        self.model = model
        self.class_names = class_names
        self.device = device
        self.permission = names
        self.colorDict = color_dict

        # create instance of SORT
        self.mot_tracker = Sort(max_age=10, min_hits=2)
        self.countLane = {}
        self.running_flag = 0
        self.pause_flag = 0
        self.videoList = []
        self.history = {}  #save history
        for item in Lane_name:
            vars(self)[f"countLane['{item}']"] = None
            vars(self)[f"tracking_count_on_lane_{item}"] = []
            vars(self)[f"history_lane_{item}"] = {}
            vars(self)[f"unique_{item}"] = []
            vars(self)[f"{item}_CAR"] = 0
            vars(self)[f"{item}_BUS"] = 0
            vars(self)[f"{item}_TRUCK"] = 0
            vars(self)[f"{item}_MOTORBIKE"] = 0
            vars(self)[f"{item}_BICYCLE"] = 0
            vars(self)[f"{item}_PERSON"] = 0
            

        self.sin_runningFlag.connect(self.update_flag)
        self.sin_videoList.connect(self.update_videoList)
        self.sin_countLane.connect(self.update_countLanes)

        self.save_dir = "results"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

    def run(self):
        video = self.videoList[0]
        cap = cv2.VideoCapture(video)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("video fps: {}".format(fps))
        codec = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(os.path.join(self.save_dir,"output.avi"), codec, fps/4, (width, height), True)
        frame_count = 0
        while True:
            if self.running_flag:
                ret, frame = cap.read()
                if frame_count % 4 == 0:
                    if ret:
                        a1 = time.time()
                        frame_v, objects = self.counter(frame)
                        for item in Lane_name:
                            if len(vars(self)[f"countLane['{item}']"]) > 2:
                                self.counter_lane(objects, np.array(vars(self)[f"countLane['{item}']"]), item)
                        self.sin_counterResult.emit(frame_v)
                        out.write(frame_v)
                        self.write_to_excel()
                        a2 = time.time()
                        print(f"fps: {1 / (a2 - a1):.2f}")
                    else:
                        break
                frame_count += 1
            else:
                time.sleep(0.1)
                # break

        #restart count for each video
        KalmanBoxTracker.count = 0
        cap.release()
        out.release()

        if self.running_flag:
            self.sin_done.emit(1)

    def update_pauseFlag(self, flag):
        self.pause_flag = flag

    def update_flag(self,flag):
        self.running_flag = flag

    def update_videoList(self, videoList):
        print("Update videoPath!")
        self.videoList = videoList

    def update_countLanes(self, Lanes):
        print("Update countLanes!")
        for item in Lane_name:
            vars(self)[f"countLane['{item}']"] = Lanes[f"{item}"]

    def counter_lane(self, objects, CountArea, lane_name):
        
        # painting area
        AreaBound = [min(CountArea[:, 0]), min(CountArea[:, 1]), max(CountArea[:, 0]), max(CountArea[:, 1])]
        painting = np.zeros((AreaBound[3] - AreaBound[1], AreaBound[2] - AreaBound[0]), dtype=np.uint8)
        CountArea_mini = CountArea - AreaBound[0:2]
        cv2.fillConvexPoly(painting, CountArea_mini, (1,))
        # check out object in count area
        objects = list(filter(lambda x: pointInCountArea(painting, AreaBound, [int(x[2][0]), int(x[2][1])]),objects))
        for item in objects:
            if item[0] not in vars(self)[f"unique_{lane_name}"]:
                vars(self)[f"unique_{lane_name}"].append(item[0])
                vars(self)[f"tracking_count_on_lane_{lane_name}"].append([item[0], item[1], [int(item[2][0]), int(item[2][1])]])
                # print(lane_name, [item[0], item[1], [int(item[2][0]), int(item[2][1])]])
                print(vars(self)[f"unique_{lane_name}"])
                car_list = list(filter(lambda x: x[1] == 'car', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                bus_list = list(filter(lambda x: x[1] == 'bus', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                truck_list = list(filter(lambda x: x[1] == 'truck', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                motorbike_list = list(filter(lambda x: x[1] == 'motorbike', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                bicycle_list = list(filter(lambda x: x[1] == 'bicycle', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                person_list = list(filter(lambda x: x[1] == 'person', vars(self)[f"tracking_count_on_lane_{lane_name}"]))
                vars(self)[f"{lane_name}_CAR"] = len(car_list)
                vars(self)[f"{lane_name}_TRUCK"] = len(truck_list)
                vars(self)[f"{lane_name}_BUS"] = len(bus_list)
                vars(self)[f"{lane_name}_MOTORBIKE"] = len(motorbike_list)
                vars(self)[f"{lane_name}_BICYCLE"] = len(bicycle_list)
                vars(self)[f"{lane_name}_PERSON"] = len(person_list)

    def counter(self, frame):
        
        objects_info = []
        objects = predict.yolo_prediction(self.model,self.device,frame,self.class_names) # detect all in frame
        objects = filter(lambda x: x[0] in self.permission, objects) # compare object name and permission name : filter
        objects = filter(lambda x: x[1] > 0.5,objects) # compare accracy: filter
        objects = list(objects)
        #filter out repeat bbox
        objects = filter_out_repeat(objects)

        detections = []
        for item in objects:
            detections.append([int(item[2][0] - item[2][2] / 2),
                               int(item[2][1] - item[2][3] / 2),
                               int(item[2][0] + item[2][2] / 2),
                               int(item[2][1] + item[2][3] / 2),
                               item[1]])
        track_bbs_ids = self.mot_tracker.update(np.array(detections))

        # painting lain_area
        for item in Lane_name:
            if len(vars(self)[f"countLane['{item}']"]) >1:
                for i in range(len(vars(self)[f"countLane['{item}']"])):
                    cv2.line(frame, tuple(vars(self)[f"countLane['{item}']"][i]), tuple(vars(self)[f"countLane['{item}']"][(i + 1) % (len(vars(self)[f"countLane['{item}']"]))]), (255, 0, 0), 1)
                cv2.putText(frame, item, tuple(vars(self)[f"countLane['{item}']"][0]), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0, 0, 255), thickness=1)

        if len(track_bbs_ids) > 0:
            for bb in track_bbs_ids:    #add all bbox to history
                id = int(bb[-1])
                objectName = get_objName(bb, objects)
                if id not in self.history.keys():  #add new id
                    self.history[id] = {}
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"] = []
                    self.history[id]["his"].append(objectName)
                else:
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"].append(objectName)

        for i, item in enumerate(track_bbs_ids):
            bb = list(map(lambda x: int(x), item))
            id = bb[-1]
            x1, y1, x2, y2 = bb[:4]

            his = self.history[id]["his"]
            result = {}
            for i in set(his):
                result[i] = his.count(i)
            res = sorted(result.items(), key=lambda d: d[1], reverse=True)
            objectName = res[0][0]

            boxColor = self.colorDict[objectName]
            cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, thickness=1)
            cv2.putText(frame, str(id) + "_" + objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_DUPLEX , 0.5,
                        boxColor,
                        thickness=1)
            objects_info.append([str(id), objectName, [(x1 + x2) / 2, (y1 + y2) / 2]])

        removed_id_list = []
        for id in self.history.keys():    #extract id after tracking
            self.history[id]["no_update_count"] += 1
            if  self.history[id]["no_update_count"] > 5:  # if object no tracking over 5 times
                his = self.history[id]["his"]
                result = {}
                for i in set(his):
                    result[i] = his.count(i)  # if object out of frame  
                res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                objectName = res[0][0]
                #del id
                removed_id_list.append(id)

        for id in removed_id_list:
            _ = self.history.pop(id)

        return frame, objects_info

    def write_to_excel(self):
        columns = ['CAR', 'BUS', 'TRUCK', 'MOTORBIKE', 'BICYCLE', 'PERSON']
        index = Lane_name
        content = []
        for item in Lane_name:
            content.append([vars(self)[f"{item}_CAR"], vars(self)[f"{item}_BUS"], vars(self)[f"{item}_TRUCK"], vars(self)[f"{item}_MOTORBIKE"], vars(self)[f"{item}_BICYCLE"], vars(self)[f"{item}_PERSON"]])
        df = pd.DataFrame(content, index=index, columns=columns)
        df.to_excel(os.path.join(self.save_dir, 'results.xlsx'), sheet_name='result')

def filter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if no repeat
            if not flag:
                new_objects.append(objects[i])
        #add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))


def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou

def get_objName(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0]

def pointInCountArea(painting, AreaBound, point):
    h,w = painting.shape[:2]
    point = np.array(point)
    point = point - AreaBound[:2]
    if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
        return 0
    else:
        return painting[point[1],point[0]]

