import cv2
import numpy as np
import time

np.random.seed(10)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        ##############################################################################

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath,'rt') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0,'__Background__')

        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

        #print(self.classesList)
    
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening file...")
            return
        (success, image) = cap.read()
        
        startTime = 0
        object_dict = {}
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.8)

            if len(bboxIdx) != 0:
                for i in range(0,len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for  c in self.colorList[classLabelID]]
                    
                    if classLabel not in [ 'person', 'animal']:
                        continue

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence*100)
                    #print(displayText)

                    x,y,w,h = bbox
                    
                    object_id = f"{classLabel}_{i}"
                    if object_id not in object_dict:
                        object_dict[object_id] = {"label": classLabel, "start_time": currentTime, "end_time": None}

                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=2)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    
                    ###################################################
                    lineWidth = min(int(w*0.3), int(h*0.3))
                    
                    cv2.line(image,(x,y),(x + lineWidth, y), classColor, thickness=5)
                    cv2.line(image,(x,y),(x, y + lineWidth), classColor, thickness=5)
                    
                    cv2.line(image,(x + w,y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image,(x + w,y), (x + w, y + lineWidth), classColor, thickness=5)
                    
                    ####################################################
                    cv2.line(image, (x,y+h), (x + lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x,y+h), (x, y + h - lineWidth), classColor, thickness=5)
                    
                    cv2.line(image, (x+w,y+h), (x + w - lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x+w,y+h), (x + w ,y + h - lineWidth), classColor, thickness=5)

            
            current_time = time.time()
            keys_to_remove = []
            for object_id, info in object_dict.items():
                if current_time - info["start_time"] > 8.0:  # Assume an object expires after 5 seconds
                    info["end_time"] = current_time
                    keys_to_remove.append(object_id)

            for key in keys_to_remove:
                print(f"{object_dict[key]['label']} (ID: {key}) detected from {object_dict[key]['start_time']} to {object_dict[key]['end_time']}")
                del object_dict[key]
                
            cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)       
            cv2.imshow("Object Detection App", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
            
        cv2.destroyAllWindows()