from ultralytics import YOLO
import cv2 as cv

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
CHAIR_WIDTH = 24.0 #INCHES
BOTTLE_WIDTH = 2.0 #INCHES #44.8 pixels
TABLE_WIDTH = 42.0 #INCHES #942.2 pixels

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

model = YOLO("yolov8n.pt")

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

def object_detector(image):
    results = model.predict(source = image , verbose=False)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    # creating empty list to add objects data
    data_list =[]
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # getting the data 
        if int(detected_class) ==0 or int(detected_class) ==67 or int(detected_class) == 56 or int(detected_class) == 39 or int(detected_class) == 60: 
            data_list.append(class_names[int(detected_class)])
            data_list.append(x2-x1)             
            data_list.append(pt1)
            data_list.append(pt2)
            data_list.append(int(detected_class))
        # returning list containing the object data. 
    return data_list

# focal length finder function 
def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

mobile_data = object_detector('ReferenceImagesV8/cellphone.png')
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_data[1])

person_data = object_detector('ReferenceImagesV8/person.png')
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_data[1])

chair_data = object_detector('ReferenceImagesV8/chair.png')
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_data[1])

#bottle_data = object_detector('ReferenceImagesV8/chair.png')
#focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_data[1])
bottle_data = 44.8
focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_data)

table_data = 942.2
focal_table = focal_length_finder(KNOWN_DISTANCE, TABLE_WIDTH, table_data)

print(f"type is: {mobile_data[0]} and width is {mobile_data[1]}")
print(f"type is: {person_data[0]} and width is {person_data[1]}")
print(f"type is: {chair_data[0]} and width is {chair_data[1]}")
#print(f"type is: {bottle_data[0]} and width is {bottle_data[1]}")
print(f"type is: bottle and width is {bottle_data}")
print(f"type is: diningtable and width is {table_data}")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    data = object_detector(frame) 
    for i in range(0 , len(data) , 5):
        if data[i + 0] =='cell phone' or data[i + 0] =='person' or data[i + 0] == 'chair' or data[i + 0] == 'bottle' or data[i + 0] == 'diningtable' :
            if data[i + 0] =='cell phone' :
                distance = distance_finder (focal_mobile, MOBILE_WIDTH, data[i + 1])
            elif data[i + 0] =='person' :   
                distance = distance_finder (focal_person, PERSON_WIDTH, data[i + 1]) 
            elif data[i + 0] =='chair' :   
                distance = distance_finder (focal_chair, CHAIR_WIDTH, data[i + 1]) 
            elif data[i + 0] =='bottle' :   
                distance = distance_finder (focal_bottle, BOTTLE_WIDTH, data[i + 1]) 
            elif data[i + 0] =='diningtable' :   
                distance = distance_finder (focal_bottle, BOTTLE_WIDTH, data[i + 1]) 
            pt1 = data[i + 2]
            pt2 = data[i + 3]
            color= COLORS[data[i + 4] % len(COLORS)]
            cv.rectangle(frame, pt1, pt2,color,2)
            cv.rectangle(frame, (pt1[0], pt1[1]-3), (pt2[0], pt1[1]+30),BLACK,-1 )
            cv.putText(frame, f'{data[i + 0]}', (pt1[0]+5,pt1[1]+13), FONTS, 0.48, GREEN , 2 )   
            cv.putText(frame, f'Dis: {round(distance,2)} inch', (pt1[0]+5,pt1[1]+30), FONTS, 0.48, GREEN , 2 )     
    cv.imshow('frame',frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()
