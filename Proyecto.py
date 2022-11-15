import cv2
import time
import math as m
import mediapipe as mp
import os

# Definimos si deseamos partir de una grabación ya realizada (recorded = True) o si deseamos realizar una grabación (recorded = False) para evaluar la postura
print('Introduzca si desea operar con un video grabado [y] o si desea grabar uno nuevo [N]')
answ = input()
if answ == 'y':
    recorded = True
    print('Introduzca la ruta en la que se encuentra la grabación')
    record = input()
    while os.path.exists(record) == False:
        print('Introduzca una ruta valida')
        record = input()

    if os.path.exists(record) == True:
        record = input()
if answ == 'N':
    recorded = False
if answ != "y" and answ != "N":
    print('Introduzca si desea operar con un video grabado [y] o si desea grabar uno nuevo [N]')


# Iniciamos la grabación para evaluar la postura
def grabar():
    parar = False
    if recorded == False:
        vid = cv2.VideoCapture(1)
        fps = vid.get(cv2.CAP_PROP_FPS)  # Definir fps en funcion de la source
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Definir size en funcion de la source
        videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        while (True):
            if not parar:
                ret, frame = vid.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('s'):  # Si pulsamos s parar=true
                if parar:
                    parar = False
                else:
                    parar = True

            cv2.imshow('frame', frame)  # Display the resulting frame
            videoWriter.write(frame)  # Escribimos en el video generado con el writer los frames

# Definir el nombre (y ruta) del archivo en caso de que recorded == True
if recorded == False:
    grabar()

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
buenos_frames = 0
malos_frames = 0

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

















