from flask import Flask, Response, render_template, redirect, request, session
import cv2, sys, os
import argparse
import os

##########James############
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from predect import Predictor, image_demo
##########James############

##########smallJames###########
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
##########smallJames###########
from flask_socketio import SocketIO


#Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
camera = cv2.VideoCapture(0)
current = None
socketio = SocketIO(app)

def gen_frames(res = '1080p', cls = [], conf = 0.6):

    ##########James############
    # exp = get_exp(None, 'yolox-s')
    # model = exp.get_model()
    # model.eval()

    # ckpt_file = 'YOLOX/yolox_s.pth'
    # ckpt = torch.load(ckpt_file, map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    # predictor = Predictor( model, exp, COCO_CLASSES, None, None, 'cpu', False, False)
    ##########James############

    ##########smallJames###########
    parser = argparse.ArgumentParser(description='Code for FaceTrackCam.')
    parser.add_argument('--camera', '-cam', help='Camera divide number.', type=int, default=0)

    # from bytetrack demo.py
    # ==========================================
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # ==========================================
    args = parser.parse_args()

    net = cv2.dnn.readNetFromDarknet('yolov4-tiny-obj.cfg', 'yolov4-tiny-obj_last.weights')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    font = cv2.FONT_HERSHEY_SIMPLEX # FPS font
    timer = Timer()

    tracker = BYTETracker(args, frame_rate=args.fps)
    frame_id = 0
    ##########smallJames###########

    while True:
        success, frame = camera.read()  # read the camera frame
       
        if not success:
            break
        else:
            # print(session['resolution'], file=sys.stderr)
            if   res == '720p':
                frame = cv2.resize(frame, (1280, 720))
            elif res == '480p':
                frame = cv2.resize(frame, (854, 480))
            elif res == '360p':
                frame = cv2.resize(frame, (640, 360))
            elif res == '240p':
                frame = cv2.resize(frame, (426, 240))

            ##########smallJames###########
            img_info = {"id": 0}
            frame=cv2.flip(frame,1)  #mirror the image
            height, width = frame.shape[:2]
            img_info['raw_img'] = frame
            img_info["height"] = height
            img_info["width"] = width
            
            timer.tic()
            classIds, scores, faces = model.detect(frame, confThreshold=0, nmsThreshold=0.4)
            
            if len(faces)!=0:
                faces[:, 2] = faces[:, 0] + faces[:, 2]
                faces[:, 3] = faces[:, 1] + faces[:, 3]

                scores = scores.reshape((len(faces), 1))
                faces = np.concatenate((faces, scores), axis=1)
                online_targets = tracker.update(faces, [img_info['height'], img_info['width']], (416, 416))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                timer.toc()
                frame = plot_tracking(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                        )
                
            else:
                timer.toc()
                frame = frame

            frame_id += 1
            ##########smallJames###########

            ##########James############
            # frame = image_demo(predictor, frame, conf = conf, cls = cls)
            ##########James############
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            global current
            current =  (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            yield current        

# Get video
@app.route('/video_feed')
def video_feed():
    try:
        session['resolution']
    except:
        session['resolution'] = '1080p'

    try:
        session['cls']
    except:
        session['cls'] = []
    
    try:
        session['conf']
    except:
        session['conf'] = 0.6

    current = gen_frames(res = session['resolution'], cls = session['cls'], conf = session['conf'])

    return Response(current, mimetype='multipart/x-mixed-replace; boundary=frame')


#start/stop
@app.route('/')
def index():
    previous_data = {"selected_item":"1080p"}
    return render_template('index.html',previous_data=previous_data)

@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Start':
            return redirect('/')
        elif request.form['submit_button'] == 'Pause':
            try:
                previous_data = {"selected_item":session['resolution'] }  
            except:
                previous_data = {"selected_item":"1080p" }  
            return render_template('freeze.html', previous_data=previous_data)


@app.route('/freeze')
def freeze():
    return Response(current, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/resolution', methods=['GET', 'POST'])
def resolution():
    if request.method == 'POST':
        if request.form['operator'] == '1080p':
            session['resolution'] = '1080p'
        elif request.form['operator'] == '720p':
            session['resolution'] = '720p'
        elif request.form['operator'] == '480p':
            session['resolution'] = '480p'
        elif request.form['operator'] == '360p':
            session['resolution'] = '360p'
        elif request.form['operator'] == '240p':
            session['resolution'] = '240p'
        
        previous_data = {"selected_item":session['resolution'] }      
        return render_template('index.html',previous_data=previous_data)


@app.route('/detect', methods=['POST'])
def my_form_post():
    text = request.form['text']
    text = str(text.lower())

    try:
        session['cls']
    except:
        session['cls'] = []

    c = session['cls']
    if request.form['submit_button'] == 'Add':
        print('Add:', text, file=sys.stderr)
        if text not in session['cls']:
            c.append(text)
            session['cls'] = c
        
            
    elif request.form['submit_button'] == 'Remove':
        print('Reomve:', text, file=sys.stderr)
        if text in session['cls']:
            c.remove(text)
            session['cls'] = c

    elif request.form['submit_button'] == 'submit':
        session['conf'] = float(text)
        print("session['conf']:", session['conf'], file=sys.stderr)

    print("session['cls']:", session['cls'], file=sys.stderr)
    
    
    try:
        previous_data = {"selected_item":session['resolution'] }  
    except:
        previous_data = {"selected_item":"1080p" }  
    return render_template('index.html', previous_data=previous_data)

@socketio.on('send_offset_to_server')
def handle_offset(offset_data):
    print(f'OffsetX: {offset_data["offsetX"]}, OffsetY: {offset_data["offsetY"]}, videoWidth: {offset_data["videoWidth"]}, videoHeight: {offset_data["videoHeight"]}')
    # Perform actions based on the received offset values...

if __name__ == "__main__":
    app.run(host='0.0.0.0',  port='8000', debug=True, threaded=True)
    socketio.run(app, debug=True)
