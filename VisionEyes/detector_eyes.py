import cv2
import dlib
from scipy.spatial import distance
import os
import sys

def eye_aspect_ratio(eye):
    """
    Calcula a relação de aspecto do olho (EAR).
    """
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def inicializar_detector_de_faces(caminho_modelo=None):
    """
    Inicializa o detector de faces com o modelo pré-treinado do dlib.
    """
    if caminho_modelo is None:
        caminho_modelo = 'shape_predictor_68_face_landmarks.dat'  # Caminho padrão
    if not os.path.exists(caminho_modelo):
        raise ValueError("Modelo não encontrado. Forneça o caminho correto.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(caminho_modelo)
    return detector, predictor

def detectar_olhos_abertos(quadro, detector, predictor):
    """
    Detecta faces e verifica se os olhos estão abertos ou fechados.
    """
    # Assegura que o quadro está em 8 bits por canal, RGB ou Grayscale
    if quadro.dtype != 'uint8':
        quadro = (quadro * 255).astype('uint8')

    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY) if len(quadro.shape) == 3 else quadro
    faces = detector(cinza)
    olhos_abertos = False

    for face in faces:
        shape = predictor(cinza, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        EAR_THRESHOLD = 0.25

        if ear < EAR_THRESHOLD:
            color = (0, 0, 255)  # Vermelho se os olhos estiverem fechados
            status = "Olhos Fechados"
        else:
            color = (0, 255, 0)  # Verde se os olhos estiverem abertos
            status = "Olhos Abertos"
            olhos_abertos = True

        for (x, y) in left_eye:
            cv2.circle(quadro, (x, y), 2, color, -1)
        for (x, y) in right_eye:
            cv2.circle(quadro, (x, y), 2, color, -1)

        cv2.rectangle(quadro, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(quadro, status, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return quadro, olhos_abertos


def main(caminho_modelo=None):
    """
    Função principal que realiza o reconhecimento de olhos abertos/fechados em um vídeo.
    """
    try:
        detector, predictor = inicializar_detector_de_faces(caminho_modelo)
    except Exception as e:
        print(f"Erro ao inicializar o detector de faces: {e}")
        return

    caminho_video = 'olhos.mp4'
    captura_de_video = cv2.VideoCapture(caminho_video)

    if not captura_de_video.isOpened():
        print(f"Não foi possível abrir o vídeo: {caminho_video}")
        return

    print('\033[1;31;43m' + 'Processando vídeo...' + '\033[0;39;49m')

    try:
        while True:
            ret, quadro = captura_de_video.read()
            if not ret:
                break

            quadro, olhos_abertos = detectar_olhos_abertos(quadro, detector, predictor)

            cv2.imshow('Reconhecimento de Olhos', quadro)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Erro durante o processamento do vídeo: {e}")
    finally:
        captura_de_video.release()
        cv2.destroyAllWindows()
        print("Recursos liberados com sucesso.")

if __name__ == "__main__":
    caminho_modelo = sys.argv[1] if len(sys.argv) > 1 else None
    main(caminho_modelo)
