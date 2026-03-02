import cv2
import mediapipe as mp
import time
import os
import urllib.request

class GestureAuthenticator:
    def __init__(self):
        # --- Configurações ---
        self.PASSWORD_SEQUENCE = [2, 5, 0]
        self.CONFIRMATION_TIME = 1.5
        self.MODEL_PATH = 'hand_landmarker.task'

        # --- Variáveis de Estado ---
        self.current_step = 0
        self.gesture_start_time = None
        self.last_detected_fingers = -1
        self.unlocked = False

        self._setup_mediapipe()

    def _setup_mediapipe(self):
        """Baixa o modelo e inicializa o MediaPipe"""
        if not os.path.exists(self.MODEL_PATH):
            print("Baixando modelo do MediaPipe...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", self.MODEL_PATH)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def count_fingers(self, hand_landmarks, handedness):
        """Retorna a quantidade de dedos levantados"""
        finger_tips = [8, 12, 16, 20]
        fingers_status = []
        is_right_hand = handedness[0].category_name == 'Right'
        thumb_tip_x = hand_landmarks[4].x
        thumb_ip_x = hand_landmarks[3].x
        
        if is_right_hand:
            fingers_status.append(1 if thumb_tip_x < thumb_ip_x else 0)
        else:
            fingers_status.append(1 if thumb_tip_x > thumb_ip_x else 0)

        for tip_id in finger_tips:
            fingers_status.append(1 if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y else 0)

        return sum(fingers_status)

    def reset_login(self):
        """Zera a tentativa de login"""
        self.current_step = 0
        self.gesture_start_time = None
        self.last_detected_fingers = -1
        self.unlocked = False

    def process_frame(self, img):
        """Recebe uma imagem da câmera, aplica as regras do login e devolve a imagem desenhada"""
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.landmarker.detect(mp_image)
        
        fingers_up_count = 0

        # Detecção e desenho
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                fingers_up_count = self.count_fingers(hand_landmarks, handedness)

        # Lógica de Login
        if not self.unlocked:
            if fingers_up_count == self.PASSWORD_SEQUENCE[self.current_step]:
                if self.last_detected_fingers != fingers_up_count:
                    self.gesture_start_time = time.time()
                    self.last_detected_fingers = fingers_up_count
                elif time.time() - self.gesture_start_time > self.CONFIRMATION_TIME:
                    self.current_step += 1
                    self.gesture_start_time = None
                    self.last_detected_fingers = -1
                    if self.current_step == len(self.PASSWORD_SEQUENCE):
                        self.unlocked = True
                else:
                    # Barra de progresso visual
                    bar_length = min(int(((time.time() - self.gesture_start_time) / self.CONFIRMATION_TIME) * 200), 200)
                    cv2.rectangle(img, (10, 120), (10 + bar_length, 140), (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(img, (10, 120), (210, 140), (255, 255, 255), 2)
            else:
                self.last_detected_fingers = fingers_up_count
                self.gesture_start_time = time.time()

            # Textos informativos
            cv2.putText(img, f"Dedos: {fingers_up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Etapa: {self.current_step+1}/{len(self.PASSWORD_SEQUENCE)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Tela de Sucesso
        if self.unlocked:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.putText(img, "ACESSO LIBERADO", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        return img