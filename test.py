from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime
import csv
import cv2
import pickle
import numpy as np
import os

def speak(text):
    os.system(f"say '{text}'")

v = cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/name.pkl','rb') as f:
    LABLE=pickle.load(f)
with open('data/face_data.pkl','rb') as f:
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLE)
col_name=['NAME','TIME']
while True:
    ret, frame = v.read()
    if not ret:
        print(" Failed to grab frame")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=facedetect.detectMultiScale(gray,1.3, 5)
    for(x,y,w,h) in face:
        crop_img=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        op=knn.predict(resized_img)
        t=time.time()
        d=datetime.fromtimestamp(t).strftime("%d - %m - %Y")
        ts=datetime.fromtimestamp(t).strftime("%H: - %M - %S")
        exist=os.path.isfile("attendance/attandance_"+d+".csv")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(op[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
        att=[str(op[0]),str(ts)]
    cv2.imshow("Live Feed", frame) 
    
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        print(f"Key pressed: {k}")
    if k==ord('o'):
        speak("attandance taken")
        time.sleep(5)
        if exist:
            with open("attendance/attandance_"+d+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(att)
            csvfile.close()
        else:
            with open("attendance/attandance_"+d+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(col_name)
                writer.writerow(att)
            csvfile.close()
    if k == ord('q') :
        print(" 'q' was pressed. Exiting loop...")
        break

v.release()
cv2.destroyAllWindows()


