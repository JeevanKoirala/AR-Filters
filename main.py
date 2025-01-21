import cv2
import mediapipe as mp

import numpy as np

import os

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

filters = {}
selected_filters = []




filter_keys = {"1": "hat", "2": "sunglasses", "3": "mustache"}


def load_filters():
    global filters
    filter_dir = "filters"
    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)
    for file in os.listdir(filter_dir):
        if file.endswith((".png", ".jpeg", ".jpg")):
            filter_name = os.path.splitext(file)[0]
            img = cv2.imread(os.path.join(filter_dir, file), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img = remove_white_background(img)
                filters[filter_name] = img

def remove_white_background(img):
    lower_white = np.array([200, 200, 200, 255])
    upper_white = np.array([255, 255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    img[mask > 0] = (255, 255, 255, 0)
    return img





def overlay_image(background, overlay, x, y, scale=1):
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]




def detect_faces(frame, face_detection, face_mesh, w, h):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_detection = face_detection.process(rgb_frame)
    results_mesh = face_mesh.process(rgb_frame)
    face_boxes = []
    face_landmarks = []
    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_boxes.append((x, y, width, height))
    if results_mesh.multi_face_landmarks:
        for face_landmark in results_mesh.multi_face_landmarks:
            face_landmarks.append(face_landmark)
    return face_boxes, face_landmarks





def apply_filters(frame, face_boxes, face_landmarks):
    for (x, y, width, height), landmarks in zip(face_boxes, face_landmarks):
        for filter_name in selected_filters:
            if filter_name in filters:
                if filter_name == "hat":
                    top_of_head = landmarks.landmark[10]
                    hat_x = int((x + width / 2) - (filters[filter_name].shape[1] * (width / filters[filter_name].shape[1]) / 2))
                    hat_y = int(top_of_head.y * frame.shape[0] - filters[filter_name].shape[0] * (width / filters[filter_name].shape[1]) * 1.4)
                    overlay_image(frame, filters[filter_name], hat_x, hat_y, scale=width / filters[filter_name].shape[1])
                elif filter_name == "sunglasses":
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[263]
                    glasses_x = int((left_eye.x + right_eye.x) / 2 * frame.shape[1] - width / 2)
                    glasses_y = int((left_eye.y + right_eye.y) / 2 * frame.shape[0] - height * 0.25)
                    overlay_image(frame, filters[filter_name], glasses_x, glasses_y, scale=width / filters[filter_name].shape[1])
                elif filter_name == "mustache":
                    nose_bottom = landmarks.landmark[2]
                    upper_lip_top = landmarks.landmark[13]
                    mustache_x = int(nose_bottom.x * frame.shape[1] - filters[filter_name].shape[1] / 2 * (width / filters[filter_name].shape[1]))
                    mustache_y = int((upper_lip_top.y + nose_bottom.y) / 2 * frame.shape[0] - filters[filter_name].shape[0] / 2 * (width / filters[filter_name].shape[1]))
                    overlay_image(frame, filters[filter_name], mustache_x, mustache_y, scale=width / filters[filter_name].shape[1])

def main():
    global selected_filters
    cap = cv2.VideoCapture(0)
    load_filters()
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            face_boxes, face_landmarks = detect_faces(frame, face_detection, face_mesh, w, h)
            apply_filters(frame, face_boxes, face_landmarks)
            
            cv2.putText(frame, "Press 1: Hat, 2: Sunglasses, 3: Mustache", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Active Filters: {', '.join(selected_filters)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("AR Filter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif chr(key) in filter_keys:
                filter_name = filter_keys[chr(key)]
                if filter_name in selected_filters:
                    selected_filters.remove(filter_name)
                else:
                    selected_filters.append(filter_name)
    
    cap.release()
    cv2.destroyAllWindows()


    

if __name__ == "__main__":
    main()
