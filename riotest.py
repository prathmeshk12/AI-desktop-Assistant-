from logging import captureWarnings
from random import random
from time import time
from tracemalloc import start
from matplotlib import cm
import matplotlib 
import pyttsx3
import speech_recognition as sr
import datetime
import os
import wikipedia
import webbrowser
import sys
import cv2
import operator         #using mathamatical calculations.
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from Riogui import Ui_MainWindow
import openai
from config import apikey
import instaloader
import PyPDF2
import pywhatkit as kit
import requests
from requests import get
import pyjokes

# ChatGPT code
def ai(prompt):
    openai.api_key = apikey
    text = f"OpenAI response for Prompt: {prompt} \n *************************\n\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Wrap this inside of a try catch block
    # print(response["choices"][0]["text"])
    text += response["choices"][0]["text"]
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    # with open(f"Openai/prompt- {random.randint(1, 2343434356)}", "w") as f:
    with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt", "w") as f:
        f.write(text)

def say(text):
    os.system(f'say "{text}"')   

# ChatGPT code ends        


engine = pyttsx3.init('sapi5')           #use pyttsx3  : text to speech karnyasathi use kartat.
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

rate = engine.getProperty('rate')                         
engine.setProperty('rate', 125)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

class MainThread(QThread):
    def __init__(self):
        super(MainThread, self).__init__()

    def run(self):
        self.TaskExecution()  

    def takecommand(self):
        r = sr.Recognizer()              #use speechrecognition library : convert spoken word to text.
        with sr.Microphone() as source:
            print("listening....")
            r.pause_threshold = 1
            audio = r.listen(source,timeout=2,phrase_time_limit=5)

        try:
            print("Recognizing....")
            self.query = r.recognize_google(audio, language="en-in")
            print(f'user said: {self.query}\n')

        except Exception as e:
            speak("say that again please....")
            return "None"
        return self.query    


    def wish(self):
        hour = int(datetime.datetime.now().hour)    #use datetime library

        if hour>=0 and hour<12:
            speak("good morning")

        elif hour>=12 and hour<=18:
            speak("good afternoon")

        elif hour>=18 and hour<0:
            speak("good evening")
        speak("i am rio sir. please tell me how can i help you")                


    def TaskExecution(self):
        self.wish()


        while True :

        
        

            self.query = self.takecommand().lower()


            #use os library



            if "open word" in self.query:
                ntpath = "C:\\Program Files\\Microsoft Office\\root\\Office15\\WINWORD.EXE"
                os.startfile(ntpath)

            
            elif "open notepad" in self.query:
                ntpath = "C:\\Windows\\system32\\notepad.exe"
                os.startfile(ntpath)

            elif "open command prompt" in self.query:
                ntpath = "C:\\Windows\\system32\\cmd.exe"
                os.startfile(ntpath) 

            elif "open music" in self.query:
                apath = "C:\\Users\\user\\Downloads\\music.mp3"
                os.startfile(apath)        

            elif "open chrome" in self.query:
                ntpath = "C:\\Program Files\\BraveSoftware\Brave-Browser\\Application\\brave.exe"
                os.startfile(ntpath)  

            elif "open whatsapp" in self.query:
                ntpath = "C:\\Users\\Raj\\AppData\\Local\\WhatsApp\\WhatsApp.exe"
                os.startfile(ntpath)

            elif "ip address" in self.query:
                ip = get('https://api.ipify.org').text
                speak(f"your IP address is {ip}")   

            # to find a joke
            elif "tell me a joke" in self.query:
                joke = pyjokes.get_joke()
                speak(joke)

            elif "shut down the system" in self.query:
                os.system("shutdown /s /t 5")

            elif "restart the system" in self.query:
                os.system("shutdown /r /t 5")

            elif "sleep the system" in self.query:
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")                    

            #camera module

            elif "open camera" in self.query:
                file_name_path="C:\\Users\\Raj\\Desktop\\wetransfer_bnw-png_2022-02-19_0521\\wetransfer_face-model_2022-02-19_1054\\face model\\Photos\\"

                face_classifier = cv2.CascadeClassifier('D:\\python webinar 6th sem BCA\\Files\\wetransfer_bnw-png_2022-02-19_0521\\haarcascade_frontalface_default.xml')

                def face_extractor(img):

                #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        faces = face_classifier.detectMultiScale(img,1.1,5)

                        if faces ==():
                            return None

                        for(x,y,w,h) in faces:
                            cropped_face = img[y:y+h, x:x+w]

                        return cropped_face


                cap = cv2.VideoCapture(0)
                count = 0

                while True:
                    ret, frame = cap.read()
                    if face_extractor(frame) is not None:
                        count+=1
                        face = cv2.resize(face_extractor(frame),(700,700))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                        file_name_path = 'D:\\Rio\\Photos//user'+str(count)+'.jpg'
                        cv2.imwrite(file_name_path,face)

                        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,500,0),2)
                        cv2.imshow('Face Cropper',face)
                    else:
                        print("Face not Found")
                        pass

                    if cv2.waitKey(1)==13 or count==20:
                            break
                cap.release()
                cv2.destroyAllWindows()
                print('Save Image')


        #use wikipedia library

            elif "wikipedia" in self.query:
                speak("searching wikipedia....")
                self.query = self.query.replace("wikipedia","")
                results = wikipedia.summary(self.query, sentences=2)
                speak("according to wikipedia")
                print(results)   
                speak(results)


            #use webbrowser library

            # to close application
            elif "close notepad" in self.query:
                speak("okay sir, closing notepad")
                os.system("taskkill /f /im notepad.exe")
            # to close application
            elif "close command prompt" in self.query:
                speak("okay sir, closing command prompt")
                os.system("taskkill /f /im cmd.exe")


            elif "open youtube" in self.query:
                webbrowser.open("www.youtube.com")

            elif "open facebook" in self.query:
                webbrowser.open("www.facebook.com")  

            elif "open map" in self.query:
                webbrowser.open("www.google.com/maps")

            elif "play song on youtube" in self.query:
                kit.playonyt("see you again")         

            #elif "open google" in self.query:
                #webbrowser.open("www.google.com/")

            # again new code
            elif "Using artificial intelligence".lower() in self.query.lower():
                ai(prompt=self.query)

            elif "open google" in self.query:
                speak("sir, what should i search on google")
                cm = self.takecommand().lower()
                webbrowser.open(f"{cm}")          

            elif "you can sleep" in self.query:
                speak("thanks for using me sir, have a good day.")
                sys.exit()

            elif "open calculator" in self.query:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    speak("sir plese give, problem:")
                    print("listening....")
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                my_string = r.recognize_google(audio)
                print(my_string)

                def get_operator_fn(op):
                    return{
                        '+' : operator.add, #plus karnyasathi
                        '-' : operator.sub, #minus sathi
                        'x' : operator.mul,
                        '/' : operator.__truediv__,
                    }[op]

                def eval_binary_expr(op1, oper, op2):
                    op1,op2 = int(op1), int(op2)
                    return get_operator_fn(oper)(op1, op2)
                speak("answer is:")
                print(eval_binary_expr(*(my_string.split()))) 
                speak(eval_binary_expr(*(my_string.split())))
                
            elif "open brightness" in self.query :
                cap = cv2.VideoCapture(0)

                mpHands = mp.solutions.hands
                hands = mpHands.Hands()
                mpDraw = mp.solutions.drawing_utils

                while True:
                    success, img = cap.read()
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(imgRGB)

                    lmList = []
                    if results.multi_hand_landmarks:
                        for handlandmark in results.multi_hand_landmarks:
                            for id, lm in enumerate(handlandmark.landmark):
                                h, w, _ = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id, cx, cy])
                            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

                    if lmList != []:
                        x1, y1 = lmList[4][1], lmList[4][2]
                        x2, y2 = lmList[8][1], lmList[8][2]

                        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
                        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                        length = hypot(x2 - x1, y2 - y1)

                        bright = np.interp(length, [15, 220], [0, 100])
                        print(bright, length)
                        sbc.set_brightness(int(bright))

                    # Hand range 15 - 220
                    # Brightness range 0 - 100

                    cv2.imshow('Image', img)
                    if cv2.waitKey (1)==27:
                       break
                cap.release()   
                cv2.destroyAllWindows()
            

startExecution = MainThread()
        
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startTask)
        self.ui.pushButton_2.clicked.connect(self.close)

    def startTask(self):
        self.ui.movie = QtGui.QMovie("C:\Rio\img\mic.gif")
        self.ui.label.setMovie(self.ui.movie)
        self.ui.movie.start()
        startExecution.start()


app = QApplication(sys.argv)
Riogui = Main()
Riogui.show()
exit(app.exec_())        


         
         
            
            
        

