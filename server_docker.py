import dlib
import numpy as np
import cv2
import os
import time
import requests
import datetime
import shutil
import pickle
from flask import Flask, render_template, request
import zipfile
image_file_ext=(".jpg",".png",".jpeg",".jfif")
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

pose_predictor_68_point = dlib.shape_predictor(os.path.sep.join(["face_detector", "shape_predictor_68_face_landmarks.dat"]))
face_detector = dlib.get_frontal_face_detector()
Detecter_MMOD = dlib.cnn_face_detection_model_v1((os.path.sep.join(["face_detector", "mmod_human_face_detector.dat"])))
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
Net_Detecter_CV = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
try:
    Net_Detecter_CV.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    Net_Detecter_CV.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
except Exception as ee:
    print(ee)

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_locations(rgb_frame,number_of_times_to_upsample=1):
    return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in cnn_face_detector(img, number_of_times_to_upsample)]

def load_data(dicface={"known_face_names":[],"known_face_encodings":[],"known_face_id":[]}):
    if os.path.isfile('data_face.dat'):
        print("[Info] Loading Face Data...")
        with open("data_face.dat", "rb") as fp:   # Unpickling
            dicface = pickle.load(fp)
    return dicface

def compare_faces(known_face_encodings, face_encoding_to_check,known_face_names,known_face_id, tolerance=0.4):  
    name=""
    com_time=time.time()
    r=face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    npa_min=np.amin(r)
    index = np.where(r == npa_min)
    name2=known_face_names[index[0][0]]
    t2=npa_min
    ld=known_face_id[index[0][0]]
    if npa_min<tolerance:
        index = np.where(r == npa_min)
        name=known_face_names[index[0][0]]
    else :
        name="Unknown"
    return name,(npa_min),name2,ld

def detection_mode(mode="MMOD"):
    if not dlib.DLIB_USE_CUDA:
        return "CV","CUDA Unable"
    return mode,"CUDA Enable"

@app.route("/face_recognition",methods=['POST'])
def face_recognition():    
    if request.method == 'POST':
        if request.form['mode']=="":
            Detect_mode="MMOD"
        else:
            Detect_mode=request.form['mode']
        mode,detall=detection_mode(Detect_mode)
        print(detall)
        try:
            dicface=load_data()
            #person=0
            dict_name={'list_name':[],'s_name':[],'t':[]}
            #max_area=0
            f = request.files['file']
            file_dir=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
            f.save(file_dir)
            filename, file_extension=os.path.splitext(file_dir)
            if file_extension in image_file_ext:
                time_start=time.time()
                rgb_frame=cv2.imread(file_dir)
                (h, w) = rgb_frame.shape[:2]
                if Detect_mode=="MMOD":
                    face_locations = face_locations(rgb_frame,number_of_times_to_upsample=5)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,num_jitters=5)
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        area=(right-left)*(bottom-top)
                        if area<(h*w/100*10):
                            continue
                        """if area>max_area:
                            max_area=area
                            location=(left, top, right, bottom)"""
                        name="Unknown"
                        #person+=1
                        name,t,name2,face_id = compare_faces(dicface['known_face_encodings'], face_encoding,dicface['known_face_names'],dicface['known_face_id'])
                        dict_name['list_name'].append(name)
                        dict_name['s_name'].append(name2)
                        dict_name['t'].append(t)
                else:
                    blob = cv2.dnn.blobFromImage(cv2.resize(rgb_frame, (700, 700)), 1.0,(150,150), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()
                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.6:
                            
                            #person+=1
                            name="Unknown"
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)
                            area=(endX-startX)*(endY-startY)
                            if area<(h*w/100*10):
                                continue
                            face_encodings = face_recognition.face_encodings(rgb_frame, [(startY-20,endX+10,endY+20,startX-10)],num_jitters=5)
                            name,t,name2,face_id = compare_faces(dicface['known_face_encodings'], face_encodings[0],dicface['known_face_names'],dicface['known_face_id'])
                            dict_name['list_name'].append(name)
                            dict_name['s_name'].append(name2)
                            dict_name['t'].append(t)
                            """if area>max_area:
                                max_area=area
                                location=(startX, startY, endX, endY)"""
                    return_name=""
                    for x in list_name:
                        if return_name=="":
                            return_name=x
                        else:
                            return_name+=","+x
            else :
                return "File not support"
            tt=datetime.datetime.now()
            file_dir="//stroe//"+tt.strftime("%Y%m%d %H_%M_%S_%f")+".jpg"
            file_log = open("//stroe//log-"+tt.strftime("%Y%m%d")+".txt", "a")
            file_log.write(file_dir+"\t"+str(dict_name))
            cv2.imwrite(file_dir,rgb_frame)
            return return_name
        except Exception as ff:
            return ff


@app.route("/face_recognition_cv",methods=['POST'])
def face_recognition_cv():    
    return "...."

@app.route("/")
def test():    
    return "test"

@app.route("/upload",methods=['POST']) 
def upload():
    """try:
        if request.method == 'POST' and request.form['key']=="jt123":
            f = request.files['file']
            file_dir=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
            f.save(file_dir)
            upload_state['state']=1            
            with zipfile.ZipFile(file_dir, 'r') as zip_ref:
                zip_ref.extractall()
            for root, dirs, files in os.walk(UPLOAD_FOLDER):
                if 'known_face_names.dat' in files:
                    #shutil.move("/home/jt2/Downloads/nstda/uploads/known_face_names.dat", "/home/jt2/Downloads/nstda/known_face_names.dat")
                    os.replace(dir+"/uploads/known_face_names.dat", dir+"/known_face_names.dat")
                if 'known_face_encodings.dat' in files:
                    #shutil.move("path/to/current/known_face_encodings.dat", "path/to/new/destination/for/known_face_encodings.dat")
                    os.replace(dir+"/uploads/known_face_encodings.dat", dir+"/known_face_encodings.dat")
            shutil.rmtree(UPLOAD_FOLDER)
            os.mkdir(UPLOAD_FOLDER)
            time.sleep(3)
            upload_state['state']=2
            return "UPLOAD COMPLETE"
            else :
                return "Error"
        except Exception as inst:
            return "UPLOAD Error"+inst"""
    return "upload"

if __name__ == '__main__':
    app.run(host='0.0.0.0')