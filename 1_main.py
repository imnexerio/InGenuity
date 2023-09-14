import cv2
import sys

config_file=r'SR-DTU\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model=r'SR-DTU\frozen_inference_graph.pb'
weights_path=r"InGenuity\yolov4.weights"
cfg_path=r"InGenuity\yolov4.cfg"
#model = cv2.dnn_DetectionModel(weights_path,cfg_path)
model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels=['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
        'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
        'pottedplant','bed','diningtable','toilet','tvmonitorlaptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

model.setInputSize(320,320)
model.setInputScale(1/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

cap=cv2.VideoCapture(r'InGenuity\Surveillance Video.mp4')
#cap=cv2.VideoCapture(0)
def main():
    while True:
        ret,frame = cap.read()
        width=1000
        height=700
        dim=(width,height)
        frame=cv2.resize(frame,dim,interpolation=cv2.INTER_LINEAR)
        classIndex, confidece, bbox = model.detect(frame,nmsThreshold=0.1,confThreshold=0.422)

        if (len(classIndex)!=0):
            for ClassInd, conf, boxes in zip(classIndex.flatten(),confidece.flatten(), bbox):
                if (ClassInd<=90):
                    print((classIndex[0]))
                    #print(classLabels(list[ClassInd[0]]))
                    cv2.rectangle(frame,boxes,(255,0,0),2)
                    #cv2.putText(frame,classLabels[ClassInd-1]+str(confidece*100),(boxes[0]+10,boxes[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow('Detection in progress',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #cv2.destroyAllWindows()


main()