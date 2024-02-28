import cv2 as cv
from ultralytics import YOLO

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_COMPLEX
# reading class name from text file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


model = YOLO("yolov8n.pt")

def object_detector(image):
    results = model.predict(source = image)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        color = COLORS[int(cls) % len(COLORS)]
        label = "%s : %f" % (class_names[int(detected_class)], confidence)
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv.rectangle(image, pt1, pt2, color, 2)
        cv.putText(frame, label, (pt1[0], pt1[1] - 10), fonts, 0.5, color, 2)


camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0
while True:
    ret, frame = camera.read()

    orignal = frame.copy()
    object_detector(frame)
    cv.imshow('oringal', orignal)

    print(capture == True and counter < 10)
    if capture == True and counter < 10:
        counter += 1
        cv.putText(
            frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
    else:
        counter = 0

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        capture = True
        number += 1
        cv.imwrite(f'ReferenceImagesV8/image{number}.png', orignal)
    if key == ord('q'):
        break
cv.destroyAllWindows()
