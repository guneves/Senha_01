from flask import Flask, render_template, Response, jsonify
import cv2
from gesture_auth import GestureAuthenticator 

app = Flask(__name__)

auth_system = GestureAuthenticator()

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    import time
    prev_time = 0
    FPS_LIMIT = 20

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        current_time = time.time()
        if (current_time - prev_time) < (1.0 / FPS_LIMIT):
            continue
        prev_time = current_time
        
        processed_frame = auth_system.process_frame(frame)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset():
    """Rota chamada pelo navegador para forçar o reinício do sistema"""
    auth_system.reset_login()
    return jsonify({"status": "success", "message": "Verificação reiniciada com sucesso."})

@app.route('/status')
def status():
    """Rota que informa para o HTML se o acesso está pendente, negado ou liberado"""
    return jsonify({"status": auth_system.access_status})

if __name__ == '__main__':
    app.run(debug=True, port=5000)