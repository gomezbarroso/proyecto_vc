import cv2
import math as m
import mediapipe as mp


# Calcular distancia separacion
def calc_dist(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# Calcular inclinacion de la postura corporal
def calc_angulo(x1, y1, x2, y2):
    alpha = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    grados = int(180/m.pi)*alpha
    return grados

# Inicializar los contadores de frames
buenos_frames = 0
malos_frames = 0

# Definir la fuente
font = cv2.FONT_HERSHEY_SIMPLEX

# Definicion de colores
rojo = (50, 50, 255)
verde = (125, 255, 0)
amarillo = (0, 255, 255)
rosa = (255, 0, 255)

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
    # Obtener fps
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
        if lm.landmark[lmPose.LEFT_WRIST].visibility < 0.1:
            puntos_no_reconocidos = 'Puntos no reconocidos'
            cv2.putText(image, puntos_no_reconocidos, (10, 30), font, 0.9, rojo, 2)

        if lm.landmark[lmPose.LEFT_WRIST].visibility >= 0.1:
            lmPose = mp_pose.PoseLandmark

            # Muñeca izquierda
            muneca_izq_x = int(lm.landmark[lmPose.LEFT_WRIST].x * width)
            muneca_izq_y = int(lm.landmark[lmPose.LEFT_WRIST].y * height)

            # Muñeca derecha
            muneca_dcha_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * width)
            muneca_dcha_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * height)

            # Calcular la distancia entre los puntos de la muñeca izquierda y la muñeca derecha
            separacion = calc_dist(muneca_izq_x, muneca_izq_y, muneca_dcha_x, muneca_dcha_y)

            # Alinear la camara al punto de la vista frontal de la persona
            # El parametro separacion threshold se ha fijado a 30 tras realizar diferentes pruebas
            if separacion > 200:
                cv2.putText(image, str(int(separacion)) + ' Alineado', (10, 60), font, 0.9, verde, 2)
            else:
                cv2.putText(image, str(int(separacion)) + ' No alineado', (10, 60), font, 0.9, rojo, 2)

            # Calcular la inclinacion de la postura corporal y pintar los puntos de referencia
            # Calcular angulos
            inclinacion_muneca = calc_angulo(muneca_izq_x, muneca_izq_y, muneca_dcha_x, muneca_dcha_y)

            # Pintamos los puntos de las muñecas en la imagen
            cv2.circle(image, (muneca_izq_x, muneca_izq_y), 7, amarillo, -1)
            cv2.circle(image, (muneca_dcha_x, muneca_dcha_y), 7, rosa, -1)

            # Imprimimos el texto por pantalla, inclinacion de postura y angulos
            angulo_string = 'Angulo de inclinacion con la vertical : ' + str(int(inclinacion_muneca))

            # Condiciones de deteccion de postura corporal
            # Determinar si la postura es buena o no
            # Los angulos del threshold han sido fijados tras varias pruebas
            if inclinacion_muneca > 83 and inclinacion_muneca < 97 :
                malos_frames = 0
                buenos_frames += 1

                cv2.putText(image, angulo_string, (10, 30), font, 0.9, verde, 2)
                cv2.putText(image, str(int(inclinacion_muneca)), (muneca_izq_x + 10, muneca_izq_y), font, 0.9, verde, 2)
                # Union de puntos de referencia
                cv2.line(image, (muneca_izq_x, muneca_izq_y), (muneca_dcha_x, muneca_dcha_y), verde, 4)

            else:
                buenos_frames = 0
                malos_frames += 1

                cv2.putText(image, angulo_string, (10, 30), font, 0.9, rojo, 2)
                cv2.putText(image, str(int(inclinacion_muneca)), (muneca_izq_x + 10, muneca_izq_y), font, 0.9, rojo, 2)

                # Union de puntos de referencia
                cv2.line(image, (muneca_izq_x, muneca_izq_y), (muneca_dcha_x, muneca_dcha_y), rojo, 4)

            # Calcular el tiempo que el sujeto esta en la postura actual
            tiempo_correcto = (1 / fps) * buenos_frames
            tiempo_incorrecto = (1 / fps) * malos_frames

            # Tiempo de pose
            if tiempo_correcto > 0:
                tiempo_string_correcto = 'Tiempo de postura correcta : ' + str(round(tiempo_correcto, 1)) + 's'
                cv2.putText(image, tiempo_string_correcto, (10, height - 20), font, 0.9, verde, 2)
            else:
                tiempo_string_incorrecto = 'Tiempo de postura incorrecta : ' + str(round(tiempo_incorrecto, 1)) + 's'
                cv2.putText(image, tiempo_string_incorrecto, (10, height - 20), font, 0.9, rojo, 2)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video_output.write(image)

cap.release()
cv2.destroyAllWindows()
