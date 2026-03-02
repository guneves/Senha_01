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
        self.entered_sequence = []  # Agora guardamos a tentativa do usuário
        self.gesture_start_time = None
        self.last_detected_fingers = -1
        self.access_status = "PENDING"  # Pode ser: PENDING, GRANTED ou DENIED
        self.result_time = None         # Controla o tempo da tela de Sucesso/Erro

        self._setup_mediapipe()

    def _setup_mediapipe(self):
        """Baixa o modelo e inicializa o MediaPipe no modo VÍDEO"""
        if not os.path.exists(self.MODEL_PATH):
            print("Baixando modelo do MediaPipe...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", self.MODEL_PATH)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        
        # MUDANÇA AQUI: RunningMode.VIDEO e redução leve da confiança para priorizar velocidade
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, 
            num_hands=1,
            min_hand_detection_confidence=0.5, 
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
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
        """Zera completamente a tentativa de login"""
        self.entered_sequence = []
        self.gesture_start_time = None
        self.last_detected_fingers = -1
        self.access_status = "PENDING"
        self.result_time = None

    def process_frame(self, img):
        """Recebe o frame, aplica as regras e devolve a imagem desenhada"""
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # MUDANÇA AQUI: No modo vídeo, precisamos passar o timestamp atual em milissegundos
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hand_detected = bool(result.hand_landmarks)
        fingers_up_count = 0

        # 1. Desenha os pontos se a mão estiver na tela
        if hand_detected:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                fingers_up_count = self.count_fingers(hand_landmarks, handedness)

        # 2. Lógica de Login (Enquanto estiver pendente)
        if self.access_status == "PENDING":
            if hand_detected:
                if self.last_detected_fingers != fingers_up_count:
                    # Mudou o gesto, reinicia o cronômetro
                    self.gesture_start_time = time.time()
                    self.last_detected_fingers = fingers_up_count
                elif time.time() - self.gesture_start_time > self.CONFIRMATION_TIME:
                    # Gesto confirmado! Adiciona na sequência
                    self.entered_sequence.append(fingers_up_count)
                    
                    # Força o usuário a fazer um novo gesto (ou abaixar a mão)
                    self.gesture_start_time = None
                    self.last_detected_fingers = -1 
                    
                    # Verifica se o usuário já digitou todos os números necessários
                    if len(self.entered_sequence) == len(self.PASSWORD_SEQUENCE):
                        if self.entered_sequence == self.PASSWORD_SEQUENCE:
                            self.access_status = "GRANTED"
                        else:
                            self.access_status = "DENIED"
                        self.result_time = time.time()
                else:
                    # Desenha a barra de progresso do gesto atual
                    bar_length = min(int(((time.time() - self.gesture_start_time) / self.CONFIRMATION_TIME) * 200), 200)
                    cv2.rectangle(img, (10, 120), (10 + bar_length, 140), (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(img, (10, 120), (210, 140), (255, 255, 255), 2)
            else:
                # Se não tem mão na tela, zera o cronômetro para evitar cliques acidentais
                self.gesture_start_time = None
                self.last_detected_fingers = -1
                cv2.putText(img, "Posicione sua mao na tela", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Textos Informativos
            if hand_detected:
                cv2.putText(img, f"Dedos: {fingers_up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Mostra a sequência já digitada (ex: [2, 5])
            cv2.putText(img, f"Tentativa: {self.entered_sequence}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. Tela de Sucesso
        elif self.access_status == "GRANTED":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.putText(img, "ACESSO LIBERADO", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        # 4. Tela de Erro (Acesso Negado)
        elif self.access_status == "DENIED":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.putText(img, "ACESSO NEGADO", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            cv2.putText(img, "Reiniciando...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # AUTO-RESET: Após 3 segundos de tela vermelha, o sistema reinicia sozinho
            if time.time() - self.result_time > 3.0:
                self.reset_login()

        return img