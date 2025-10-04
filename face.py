import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pygame
import threading
import time

# -------------------- Config --------------------
authorized_img_path = "Piyush_Lamba.jpeg"  # single authorized image
attendance_file = "attendance.csv"


# Sound files
sound_attendance = "done_done.mp3"
sound_already = "already_done.mp3"
sound_unauth = "alert.mp3"

# ORB threshold
MATCH_THRESHOLD = 25  # stricter matching
BEEP_INTERVAL = 10  # seconds between beeps per person

# -------------------- Setup --------------------
pygame.mixer.init()
last_beep_time = {}  # dictionary to track last beep per person

def play_sound(path, person):
    def sound_thread():
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        last_beep_time[person] = time.time()
    t = threading.Thread(target=sound_thread)
    t.start()

# Cascade and ORB
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
orb = cv2.ORB_create()

# Load single authorized image
if not os.path.exists(authorized_img_path):
    print(f"⚠️ {authorized_img_path} not found")
    exit()

img = cv2.imread(authorized_img_path, 0)
faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(150,150))
if len(faces) == 0:
    print("⚠️ No face detected in the image.")
    exit()

(x, y, w, h) = faces[0]
face_roi = img[y:y+h, x:x+w]
kp, des = orb.detectAndCompute(face_roi, None)
name = os.path.splitext(os.path.basename(authorized_img_path))[0]
authorized_data = [{"name": name, "kp": kp, "des": des}]

# CSV
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
else:
    df = pd.DataFrame(columns=["Name", "Time", "Date"])
    df.to_csv(attendance_file, index=False)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
marked_today = {}
today = datetime.now().strftime("%Y-%m-%d")
marked_today_list = df[df["Date"] == today]["Name"].tolist()
for n in marked_today_list:
    marked_today[n] = True

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(150,150))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        kp2, des2 = orb.detectAndCompute(face_roi, None)
        identified = False
        current_time = time.time()

        if des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(authorized_data[0]["des"], des2)
            match_score = len(matches)

            if match_score > MATCH_THRESHOLD:
                identified = True
                person_name = authorized_data[0]["name"]

                if person_name not in marked_today:
                    # Mark attendance
                    now = datetime.now()
                    time_now = now.strftime("%H:%M:%S")
                    date_now = now.strftime("%Y-%m-%d")
                    new_entry = {"Name": person_name, "Time": time_now, "Date": date_now}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv(attendance_file, index=False)
                    print(f"✅ Attendance marked for {person_name} at {time_now} on {date_now}")
                    marked_today[person_name] = True

                    if person_name not in last_beep_time or current_time - last_beep_time[person_name] > BEEP_INTERVAL:
                        play_sound(sound_attendance, person_name)

                    cv2.putText(frame, f"{person_name} Done", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{person_name} Already Done", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if person_name not in last_beep_time or current_time - last_beep_time[person_name] > BEEP_INTERVAL:
                        play_sound(sound_already, person_name)

        if not identified:
            cv2.putText(frame, "Not Authorized", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if "unauth" not in last_beep_time or current_time - last_beep_time["unauth"] > BEEP_INTERVAL:
                play_sound(sound_unauth, "unauth")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
