"""
Sistema de login por gestos (MediaPipe Tasks API)
Requer: opencv-python, mediapipe
"""

import cv2
import mediapipe as mp
import time
import os
import urllib.request

PASSWORD_SEQUENCE = [2, 5, 0]
CONFIRMATION_TIME = 1.5

""" Baixa o modelo do MediaPipe se não existir localmente """
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

""" Conexões do esqueleto da mão para desenho """
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks_manual(img, landmarks):
    """ Desenha os pontos e linhas da mão usando OpenCV """
    h, w, _ = img.shape
    points = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)

def main():
    """ Inicializa os componentes do MediaPipe Tasks API """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    landmarker = HandLandmarker.create_from_options(options)

    """ IDs das pontas dos dedos (Indicador=8, Médio=12, Anelar=16, Mínimo=20) """
    finger_tips = [8, 12, 16, 20]

    cap = cv2.VideoCapture(0)
    current_step = 0
    gesture_start_time = None
    last_detected_fingers = -1
    unlocked = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        """ Converte a imagem e processa a detecção """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        
        fingers_up_count = 0

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_landmarks_manual(img, hand_landmarks)
                fingers_status = []
                
                """ Calcula o estado do polegar usando coordenadas X """
                is_right_hand = handedness[0].category_name == 'Right'
                thumb_tip_x = hand_landmarks[4].x
                thumb_ip_x = hand_landmarks[3].x
                
                if is_right_hand:
                    fingers_status.append(1 if thumb_tip_x < thumb_ip_x else 0)
                else:
                    fingers_status.append(1 if thumb_tip_x > thumb_ip_x else 0)

                """ Calcula os outros 4 dedos usando coordenadas Y """
                for tip_id in finger_tips:
                    fingers_status.append(1 if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y else 0)

                fingers_up_count = sum(fingers_status)

        if not unlocked:
            if fingers_up_count == PASSWORD_SEQUENCE[current_step]:
                """ Avança apenas se o gesto for mantido pelo tempo de confirmação """
                if last_detected_fingers != fingers_up_count:
                    gesture_start_time = time.time()
                    last_detected_fingers = fingers_up_count
                elif time.time() - gesture_start_time > CONFIRMATION_TIME:
                    current_step += 1
                    gesture_start_time = time.time()
                    last_detected_fingers = -1
                    
                    if current_step == len(PASSWORD_SEQUENCE):
                        unlocked = True
            else:
                last_detected_fingers = fingers_up_count
                gesture_start_time = time.time()

        if unlocked:
            cv2.rectangle(img, (0, 0), (640, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "ACESSO LIBERADO", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        else:
            cv2.putText(img, f"Dedos: {fingers_up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Etapa: {current_step}/{len(PASSWORD_SEQUENCE)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if last_detected_fingers == PASSWORD_SEQUENCE[current_step] and gesture_start_time:
                bar_length = min(int(((time.time() - gesture_start_time) / CONFIRMATION_TIME) * 200), 200)
                cv2.rectangle(img, (10, 120), (10 + bar_length, 140), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (10, 120), (210, 140), (255, 255, 255), 2)

        cv2.imshow("Login por Gestos", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            unlocked = False
            current_step = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()