import cv2
import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye_points):
    """
    Calcula a relação de aspecto do olho (EAR).
    """
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def inicializar_dlib():
    """
    Inicializa o detector e o preditor do dlib.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

def detectar_olhos_abertos(quadro, detector, predictor):
    """
    Detecta faces e verifica se os olhos estão abertos ou fechados.
    """
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    faces = detector(cinza)
    olhos_abertos = False

    for face in faces:
        landmarks = predictor(cinza, face)
        left_eye = [landmarks.part(n).to_tuple() for n in range(42, 48)]
        right_eye = [landmarks.part(n).to_tuple() for n in range(36, 42)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0
        EAR_THRESHOLD = 0.25  # Valor de limiar para determinar se os olhos estão abertos

        if ear > EAR_THRESHOLD:
            olhos_abertos = True
            cv2.rectangle(quadro, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)  # Verde para olhos abertos
        else:
            cv2.rectangle(quadro, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)  # Vermelho para olhos fechados

    return quadro, olhos_abertos

def main():
    """
    Função principal para execução do detector de olhos abertos.
    """
    detector, predictor = inicializar_dlib()
    captura_de_video = cv2.VideoCapture(0)  # Use 0 para webcam padrão

    if not captura_de_video.isOpened():
        raise Exception("Não foi possível abrir a webcam.")

    print("Detectando olhos abertos ou fechados...")

    try:
        while True:
            ret, quadro = captura_de_video.read()
            if not ret:
                break

            quadro, olhos_abertos = detectar_olhos_abertos(quadro, detector, predictor)

            cv2.imshow('Detector de Olhos', quadro)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        captura_de_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
