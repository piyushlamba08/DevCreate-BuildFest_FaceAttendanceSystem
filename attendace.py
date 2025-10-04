import os, time, csv
import cv2
import numpy as np
import face_recognition
from datetime import datetime, date

# ---------------- CONFIG ----------------

DATASET_DIR = "images"
CSV_FILE = "attendance_optimiiizenhjk.csv"
PROTO_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

DNN_CONF_THRESH = 0.35
UPSCALE_ATTEMPTS = [1.0, 1.5, 2.0, 3.0]
CONFIRM_THRESHOLD = 3
RECOG_FREQ = 3
MAX_TRACKER_AGE = 4.0
IOU_MATCH_THRESH = 0.3
TINY_FACE_THRESHOLD = 80


# ---------------- HELPERS ----------------


def get_name_from_path(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent and parent != os.path.basename(DATASET_DIR.strip("/")):
        return parent
    fname = os.path.basename(path)
    name = fname.split('.')[0]
    for sep in ['_', ' ', '-']:
        if sep in name:
            name = name.split(sep)[0]; break
    return name

def load_known_faces(dataset_dir=DATASET_DIR):
    known_encodings, known_names = [], []
    supported_ext = ('.jpg','.jpeg','.png')
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(supported_ext):
                path = os.path.join(root,f)
                img = face_recognition.load_image_file(path)
                img = np.array(img, dtype=np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                encs = face_recognition.face_encodings(img)
                if len(encs)>0:
                    known_encodings.append(encs[0])
                    known_names.append(get_name_from_path(path))
                    print(f"[+] Loaded {path} as {get_name_from_path(path)}")
                else:
                    print(f"[!] No face in {path}; skipping.")
    return known_encodings, known_names

def load_today_attendance(csv_file=CSV_FILE):
    marked = set()
    today = date.today().isoformat()
    if not os.path.exists(csv_file):
        return marked
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('Date') == today:
                    marked.add(r.get('Name'))
    except: pass
    return marked

def mark_attendance(name, csv_file=CSV_FILE):
    now = datetime.now()
    dstr = now.date().isoformat()
    tstr = now.strftime("%H:%M:%S")
    exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists: w.writerow(['Name','Date','Time'])
        w.writerow([name,dstr,tstr])
    print(f"[ATTENDANCE] {name} at {dstr} {tstr}")

# ---------------- FACE DETECTION ----------------
def dnn_detect_faces(net, rgb_image, conf_thresh=DNN_CONF_THRESH):
    h,w = rgb_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), 1.0,
                                 (300,300),(104.0,177.0,123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    boxes=[]
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf>conf_thresh:
            box = (detections[0,0,i,3:7] * np.array([w,h,w,h])).astype("int")
            x1,y1,x2,y2 = box
            x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
            boxes.append((x1,y1,x2,y2,conf))
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1); interH = max(0, yB - yA + 1)
    inter = interW * interH
    boxAA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    union = boxAA + boxBB - inter
    return inter/union if union>0 else 0

def create_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        else:
            raise RuntimeError("No suitable OpenCV tracker available.")


# ---------------- MAIN ----------------

def main():
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No known faces. Exiting."); return
    already_marked = load_today_attendance()
    print("Already marked today:", already_marked)

    if not (os.path.exists(PROTO_PATH) and os.path.exists(MODEL_PATH)):
        print("Please put model files in place."); return

    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(0.4)
    if not cap.isOpened(): print("Could not open camera."); return

    trackers = {}
    next_id = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: print("Failed to grab frame."); break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            frame_idx += 1
            detections = []

            # Multi-scale detection
            for scale in UPSCALE_ATTEMPTS:
                img_for_detect = rgb if scale==1.0 else cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                dets = dnn_detect_faces(net, img_for_detect)
                if dets:
                    if scale!=1.0:
                        sx = img_for_detect.shape[1]/w; sy = img_for_detect.shape[0]/h
                        dets = [(int(x1/sx), int(y1/sy), int(x2/sx), int(y2/sy), conf) for (x1,y1,x2,y2,conf) in dets]
                    detections = dets
                    break

            # Update trackers
            removed_ids=[]
            for tid,info in list(trackers.items()):
                ok, box = info['tracker'].update(frame)
                if not ok:
                    if time.time()-info['last_seen']>MAX_TRACKER_AGE:
                        removed_ids.append(tid)
                    continue
                x,y,w_box,h_box = [int(v) for v in box]
                info['bbox'] = (x,y,x+w_box,y+h_box)
                info['last_seen'] = time.time()

            # Match detections to trackers
            for (dx1,dy1,dx2,dy2,conf) in detections:
                best_tid = None; best_iou=0
                for tid,info in trackers.items():
                    if info.get('marked'): continue  # ignore marked trackers completely
                    i = iou((dx1,dy1,dx2,dy2),info['bbox'])
                    if i>best_iou: best_iou=i; best_tid=tid
                if best_iou>=IOU_MATCH_THRESH:
                    trackers[best_tid]['last_seen']=time.time()
                    try:
                        tr=create_tracker()
                        tr.init(frame,(dx1,dy1,dx2-dx1,dy2-dy1))
                        trackers[best_tid]['tracker']=tr
                        trackers[best_tid]['bbox']=(dx1,dy1,dx2,dy2)
                    except: pass
                else:
                    tr=create_tracker()
                    tr.init(frame,(dx1,dy1,dx2-dx1,dy2-dy1))
                    trackers[next_id]={'tracker':tr,'bbox':(dx1,dy1,dx2,dy2),'last_seen':time.time(),
                                       'frames':0,'name':None,'confirm':0,'marked':False,'last_recog_frame':0}
                    next_id+=1

            # remove old trackers
            for rid in removed_ids:
                trackers.pop(rid,None)

            # Recognition per tracker
            for tid,info in list(trackers.items()):
                if info.get('marked'): 
                    trackers.pop(tid)  # completely remove after attendance
                    continue
                info['frames']+=1
                if frame_idx-info['last_recog_frame']<RECOG_FREQ: continue
                info['last_recog_frame']=frame_idx
                x1,y1,x2,y2=info['bbox']
                x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
                if x2-x1<20 or y2-y1<20: continue
                face_crop=rgb[y1:y2,x1:x2]

                if (y2-y1)<TINY_FACE_THRESHOLD:
                    factor=int(TINY_FACE_THRESHOLD/(y2-y1))+1
                    cx=(x1+x2)//2; cy=(y1+y2)//2
                    nx1=max(0,cx-(x2-x1)*factor//2)
                    ny1=max(0,cy-(y2-y1)*factor//2)
                    nx2=min(frame.shape[1]-1,nx1+(x2-x1)*factor)
                    ny2=min(frame.shape[0]-1,ny1+(y2-y1)*factor)
                    face_crop=rgb[ny1:ny2,nx1:nx2]

                if face_crop.shape[0]<140:
                    scale_up=int(200/max(1,face_crop.shape[0]))+1
                    face_crop=cv2.resize(face_crop,(face_crop.shape[1]*scale_up,face_crop.shape[0]*scale_up),interpolation=cv2.INTER_CUBIC)

                face_crop=np.ascontiguousarray(face_crop,dtype=np.uint8)
                encs=face_recognition.face_encodings(face_crop)
                name="Unknown"
                if len(encs)>0:
                    face_encoding=encs[0]
                    matches=face_recognition.compare_faces(known_encodings,face_encoding,tolerance=0.60)
                    dists=face_recognition.face_distance(known_encodings,face_encoding)
                    if len(dists)>0:
                        best_idx=np.argmin(dists)
                        if matches[best_idx]:
                            name=known_names[best_idx]

                if info['name']==name:
                    info['confirm']+=1
                else:
                    info['name']=name; info['confirm']=1

                if name!="Unknown" and info['confirm']>=CONFIRM_THRESHOLD and not info['marked'] and name not in already_marked:
                    mark_attendance(name); info['marked']=True; already_marked.add(name)
                    trackers.pop(tid)  # immediately remove tracker after marking

            # Draw
            for tid,info in trackers.items():
                x1,y1,x2,y2=info['bbox']
                color=(255,200,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=info.get('name') or "Unknown"
                cv2.putText(frame,f"{label} [{info.get('confirm',0)}]",(x1,max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

            cv2.imshow("Attendance (q to quit)",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
