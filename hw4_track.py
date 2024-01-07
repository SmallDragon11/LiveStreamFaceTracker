import cv2
import argparse
import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

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

cap=cv2.VideoCapture(args.camera)

cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

print(cam_width)
print(cam_height)

net = cv2.dnn.readNetFromDarknet('yolov4-tiny-obj.cfg', 'yolov4-tiny-obj_last.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

font = cv2.FONT_HERSHEY_SIMPLEX # FPS font
timer = Timer()

tracker = BYTETracker(args, frame_rate=args.fps)
frame_id = 0
while cap.isOpened():
    img_info = {"id": 0}
    ret, frame= cap.read()
    frame=cv2.flip(frame,1)  #mirror the image
    height, width = frame.shape[:2]
    img_info['raw_img'] = frame
    img_info["height"] = height
    img_info["width"] = width
    
    timer.tic()
    classIds, scores, faces = model.detect(frame, confThreshold=0.2, nmsThreshold=0.4)
    
    if len(faces)!=0:
        faces[:, 2] = faces[:, 0] + faces[:, 2]
        faces[:, 3] = faces[:, 1] + faces[:, 3]
        # print(faces)

        scores = scores.reshape((len(faces), 1))
        faces = np.concatenate((faces, scores), axis=1)
        online_targets = tracker.update(faces, [img_info['height'], img_info['width']], (416, 416))
        # online_targets = tracker.update(faces, [img_info['height'], img_info['width']], (width, height))
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
                # results.append(
                #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                # )
        timer.toc()
        online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
        
    else:
        timer.toc()
        online_im = frame
    
    cv2.imshow('img', online_im)

    frame_id += 1
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()