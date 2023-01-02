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

# Inicializar contador
counter = 0
stage = None

# Inicializar videocapture
cap = cv2.VideoCapture(0)

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

    if lm.landmark[lmPose.LEFT_WRIST].visibility < 0.1:
        not_found_string = 'Puntos no reconocidos'
        cv2.putText(image, not_found_string, (10, 30), font, 0.9, red, 2)

    if lm.landmark[lmPose.LEFT_WRIST].visibility >= 0.1:
        lmPose = mp_pose.PoseLandmark
        #print(lm.landmark[lmPose.LEFT_WRIST])

            

        # Muñeca izquierda
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * width)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * height)

        # Muñeca derecha
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * width)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * height)
        
        # Codo izquierdo
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * width)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * height)

        # Codo derecho
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * width)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * height)

        # Hombro izquierdo
        l_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
        l_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)

        # Hombro derecho
        r_shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)
        r_shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)

        # Calcular la distancia entre los puntos del hombro izquierdo y el hombro derecho
        offset = findDistance(l_wrist_x, l_wrist_y, r_wrist_x, r_wrist_y)

        # Calcular la inclinacion de la postura corporal y pintar los puntos de referencia
        # Calcular angulos
        wrist_inclination = findAngle(l_wrist_x, l_wrist_y, r_wrist_x, r_wrist_y)
        l_arm_inclination = findAngle(l_elbow_x, l_elbow_y, l_shoulder_x, l_shoulder_y)
        r_arm_inclination = findAngle(r_elbow_x, r_elbow_y, r_shoulder_x, r_shoulder_y)

        # Pintar puntos de referencia
        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, yellow, -1)
        cv2.circle(image, (r_wrist_x, r_wrist_y), 7, pink, -1)
        cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)
        cv2.circle(image, (r_elbow_x, r_elbow_y), 7, pink, -1)
        cv2.circle(image, (l_shoulder_x, l_shoulder_y), 7, yellow, -1)
        cv2.circle(image, (r_shoulder_x, r_shoulder_y), 7, pink, -1)

        # Imprimimos el texto por pantalla, inclinacion de postura y angulos
        angle_text_string = 'Angulo de inclinacion con la vertical : ' + str(int(wrist_inclination))
        l_arm_angle_text_string = 'Angulo de inclinacion brazos (L): ' + str(int(l_arm_inclination))
        r_arm_angle_text_string = 'Angulo de inclinacion brazos (R): ' + str(int(r_arm_inclination))
        # Condiciones de deteccion de postura corporal
        # Determinar si la postura es buena o no
        # Los angulos del threshold han sido fijados tras varias pruebas
        
        if wrist_inclination > 83 and wrist_inclination < 97 :
            bad_frames = 0
            good_frames += 1

            # cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, l_arm_angle_text_string, (10, 30), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(wrist_inclination)), (l_wrist_x + 10, l_wrist_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(l_arm_inclination)), (l_wrist_x + 10, l_wrist_y), font, 0.9, light_green, 2)
            # Union de puntos de referencia
            cv2.line(image, (l_wrist_x, l_wrist_y), (r_wrist_x, r_wrist_y), green, 4)
            cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), green, 4)
            cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), green, 4)
            cv2.line(image, (l_elbow_x, l_elbow_y), (l_shoulder_x, l_shoulder_y), green, 4)
            cv2.line(image, (r_elbow_x, r_elbow_y), (r_shoulder_x, r_shoulder_y), green, 4)
            if l_arm_inclination > 100 and r_arm_inclination > 100:
                stage = "down"
            if l_arm_inclination < 90 and r_arm_inclination < 90 and stage == 'down':
                stage = "up"
                counter +=1
        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(wrist_inclination)), (l_wrist_x + 10, l_wrist_y), font, 0.9, red, 2)

            # Union de puntos de referencia
            cv2.line(image, (l_wrist_x, l_wrist_y), (r_wrist_x, r_wrist_y), red, 4)
            cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), red, 4)
            cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), red, 4)
            cv2.line(image, (l_elbow_x, l_elbow_y), (l_shoulder_x, l_shoulder_y), red, 4)
            cv2.line(image, (r_elbow_x, r_elbow_y), (r_shoulder_x, r_shoulder_y), red, 4)
            

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

        cv2.putText(image, str(counter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video_output.write(image)

cap.release()
cv2.destroyAllWindows()
