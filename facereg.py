import cv2
import matplotlib as plt
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread("PATHOF IMAGE") 

#age
age_result=DeepFace.analyze(img, actions=['age']) 
print(age_result)

#image emotion
predict = DeepFace.analyze(img, actions=['emotion'])
print(predict)
dominant_emotion = predict[0]['dominant_emotion']   
cv2.putText(img, dominant_emotion, (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2, cv2.LINE_4)
cv2.imshow('Image with Emotion', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#webcam emotion
cap = cv2.VideoCapture(1)   
if not cap.isOpened():
    raise IOError("Cant open webcam")
while(True):
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=['emotion'],enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, result[0]['dominant_emotion'] , (20, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 2, cv2.LINE_4)
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(result)
cap.release()
cv2.destroyAllWindows()
