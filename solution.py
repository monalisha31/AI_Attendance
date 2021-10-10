import tkinter as tk
import csv
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk 
from ttkthemes import ThemedStyle

HEIGHT = 500
WIDTH = 900
root = tk.Tk()
root.title("AI Attendance: ")
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()
def resultwindow():

    window = tk.Toplevel(root)
    canvas = tk.Canvas(window, height=HEIGHT, width=WIDTH)
    canvas.pack()
    window.title("AI Attendance")
    window.geometry('900x500')
    style = ThemedStyle(window)
    style.set_theme("radiance")

    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)


    def clear():
        std_name.delete(0, 'end')
        res = ""
        label4.configure(text=res)


    def clear2():
        std_number.delete(0, 'end')
        res = ""
        label4.configure(text=res)


    def takeImage():
        name = (std_name.get())
        Id = (std_number.get())
        if name.isalpha():
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sampleNum = 0

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum = sampleNum + 1
                    cv2.imwrite("TrainingImages\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + h])
                    cv2.imshow('FACE RECOGNIZER', img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                if sampleNum > 20:
                    break

            cam.release()
            cv2.destroyAllWindows()
            
            res = 'Student details saved with: \n Roll number : ' + Id + ' and  Full Name: ' + name

            row = [Id, name]

            with open('student.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            label4.configure(text=res)
        else:

            if name.isalpha():
                res = "Enter correct Matric Number"
                label4.configure(text=res)


    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids


    def train():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("Training")
        recognizer.train(faces, np.array(Id))
        recognizer.save("Trainner.yml")
        res = "Image Trained"
        label4.configure(text=res)


    def identify():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        df = pd.read_csv("student.csv")
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cam = cv2.VideoCapture(0)
        col_names = {'Id', 'Name', 'Date', 'Time'}
        attendance = pd.DataFrame(columns=col_names)
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 60:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M')
                    aa = df.loc[df['ID'] == Id]['NAME'].values
                    tt = str(Id) + "-" + aa
                    attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                    row2 = [Id, aa, date, timeStamp]
                    with open('Attendance.csv', 'a+') as csvFile2:
                        writer2 = csv.writer(csvFile2)
                        writer2.writerow(row2)
                    csvFile2.close()
                    res = 'ATTENDANCE UPDATED WITH DETAILS'
                    label4.configure(text=res)

                else:
                    Id = 'Unknown'
                    tt = str(Id)
                    if conf > 65:
                        noOfFile = len(os.listdir("bad_images")) + 1
                        cv2.imwrite("UnknownImages\Image" + str(noOfFile) + ".jpg", img[y:y + h, x:x + w])
                        res = 'ID UNKNOWN, ATTENDANCE NOT UPDATED'
                        label4.configure(text=res)
                            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
                cv2.putText(img, str(tt), (x, y + h - 10), font, 0.8, (255, 255, 255), 1)
                cv2.imshow('FACE RECOGNIZER', img)
            if cv2.waitKey(1000) == ord('q'):
                break

            cam.release()
            cv2.destroyAllWindows()

   
    label1 = ttk.Label(window, text="Name :",
                      font=('Helvetica', 16))
    label1.pack()
    label1.place(x=83, y=40)
    std_name = ttk.Entry(window, width=25,font=('Helvetica', 14))
    std_name.pack()
    std_name.place(x=280, y=41)
    label2 = ttk.Label(window, text="Roll Number:",
                      font=('Helvetica', 16))
    label2.pack()
    label2.place(x=83, y=90)
    std_number = ttk.Entry(window, width=25,font=('Helvetica', 14))
    std_number.pack()
    std_number.place(x=280, y=91)
    label3 = ttk.Label(window, text="Class Code:",
                      font=('Helvetica', 16))
    label3.pack()
    label3.place(x=83, y=120)
    std_code = ttk.Entry(window, width=25,font=('Helvetica', 14))
    std_code.pack()
    std_code.place(x=280, y=120)

    clearBtn1 = ttk.Button(window, command=clear, text="CLEAR")
    clearBtn1.pack()
    clearBtn1.place(x=580, y=42)
    clearBtn2 = ttk.Button(window, command=clear, text="CLEAR")
    clearBtn2.pack()
    clearBtn2.place(x=580, y=92)

    label3 = ttk.Label(window, text="Output", width=10,
                      font=('Helvetica', 20))
    label3.pack()
    label3.place(x=320, y=205)
    label4 = ttk.Label(window, background="white", width=55)
    label4.pack()
    label4.place(x=95, y=260)

    takeImageBtn = ttk.Button(window, command=takeImage, text="Capture Image",
                                                      width=15)
    takeImageBtn.pack()

    takeImageBtn.place(x=130, y=155)
    trainImageBtn = ttk.Button(window, command=train, text="Train Model",
                              width=15)
    trainImageBtn.pack()
    trainImageBtn.place(x=340, y=155)
    trackImageBtn = ttk.Button(window, command=identify, text="Identify Student", width=18)
    trackImageBtn.pack()
    trackImageBtn.place(x=550, y=155)


label1_root = ttk.Label(root, text="AI Attendance", width=20,
                      font=('Helvetica', 20))
label1_root.pack()
label1_root.place(x=280, y=100)
button = ttk.Button(root, text="Proceed to Teacher's Portal",
                              command=lambda: resultwindow())

button.place(x=290, y=240)
root.mainloop()


