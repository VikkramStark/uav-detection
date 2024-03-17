from ultralytics import YOLO 
import cv2 
import cvzone 
import math 


# cap = cv2.VideoCapture("videos/traffic_cam.mp4")  
cap = cv2.VideoCapture("./videos/drone_sample_3.mp4")   

# cap = cv2.VideoCapture(0) 
cap.set(3,1280) 
cap.set(4,720) 

model = YOLO("./models/best.pt")      
# model = YOLO("./trained_weights/PPE/train_4/best.pt") 
# model = YOLO("./trained_weights/PPE/train_5/best.pt")     


# classNames = ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat"] 

classNames = model.names #{0:"Drone",1:"Other UAV"}   

while True:
    success, image = cap.read() 


    results = model(image, stream = True)  

    for r in results:
        boxes = r.boxes 
        for box in boxes:
            #OpenCV 
            bb = box.xyxy[0]  
            x1,y1,x2,y2 = map(int,bb)  
            # print(x1,y1,x2,y2) 
            # cv2.rectangle(image, (x1,y1), (x2,y2), (200,0,200), 3)  

            # CVZone 
            w,h = x2-x1, y2-y1 
            # print(bbox) 
            

            conf = math.ceil(box.conf[0]*100)/100 
            cls = int(box.cls[0]) 
            print("Class ID: ", cls, classNames.get(cls))   
            if conf>0.4:  # or classNames.get(cls) in ["car"] 
                cvzone.cornerRect(image, (x1,y1,w,h)) 
                cvzone.putTextRect(image, f"{classNames.get(cls)} {conf}", (max(0,x1),max(30,y1)), scale = 0.8, thickness = 2)       
    
    cv2.imshow("Captured Cam",image) 


    if cv2.waitKey(1) == ord("q"): 
        cap.release() 
        cv2.destroyAllWindows() 
        break 

cap.release() 
cv2.destroyAllWindows()  