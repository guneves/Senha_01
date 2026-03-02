from flask import Flask, render_template, Response
import cv2
from gesture_auth import GestureAuthenticator  # Importa a nossa lógica separada

app = Flask(__name__)

# Instancia o motor do sistema de login
auth_system = GestureAuthenticator()

def generate_frames():
    """Captura a câmera e processa usando nossa classe externa"""
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # O frame cru entra no nosso motor, que devolve ele já desenhado com as regras de login
        processed_frame = auth_system.process_frame(frame)
        
        # Codifica e envia para o navegador web
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Renderiza a página HTML principal"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Rota que a tag <img> do HTML vai consumir"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)