import cv2
import time
import math as m
import mediapipe as mp
import os

# Calcular distancia offset
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# Calcular inclinacion de la postura corporal
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree

# Inicializar los contadores de frames
good_frames = 0
bad_frames = 0

# Definir la fuente
font = cv2.FONT_HERSHEY_SIMPLEX

# Definicion de colores
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Inicializar la clase de pose de mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializar videocapture
cap = cv2.VideoCapture(1)

# Datos para el videowriter
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')

# Video writer.
video_output = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)

while(True):
# Capturar frames.
    success, image = cap.read()
    if not success:
        print("Null.Frames")
    # Get fps.
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Convertir la imagen BGR a RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    keypoints = pose.process(image)

    # Convertir la imagen de nuevo a BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Coordenadas del Landmark de postura corporal
    lmPose = mp_pose.PoseLandmark

    # Obtener los landmarks de los keypoints
    lm = keypoints.pose_landmarks
    if lm is not None:
        lmPose = mp_pose.PoseLandmark
        # Hombro izquierdo
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)

        # Hombro derecho
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)

        # Oreja izquierda
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * width)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * height)

        # Cadera izquierda
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * width)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * height)

        # Calcular la distancia entre los puntos del hombro izquierdo y el hombro derecho
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Alinear la camara al punto de la vista lateral de la persona
        # El parametro Offset threshold se ha fijado a 30 tras realizar diferentes pruebas
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Alineado', (width - 175, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' No alineado', (width - 255, 30), font, 0.9, red, 2)

        # Calcular la inclinacion de la postura corporal y pintar los puntos de referencia
        # Calcular angulos
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Pintar puntos de referencia
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

        # Pintamos los puntos de cadera y hombros en la imagen
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Imprimimos el texto por pantalla, inclinacion de postura y angulos
        angle_text_string = 'Cuello : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))


        # Condiciones de deteccion de postura corporal
        # Determinar si la postura es buena o no
        # Los angulos del threshold han sido fijados tras varias pruebas
        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            # Union de puntos de referencia
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

            # Union de puntos de referencia
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calcular el tiempo que el sujeto esta en la postura actual
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        # Tiempo de pose
        if good_time > 0:
            time_string_good = 'Tiempo de postura correcta : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, height - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Tiempo de postura incorrecta : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, height - 20), font, 0.9, red, 2)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video_output.write(image)

cap.release()
cv2.destroyAllWindows()
