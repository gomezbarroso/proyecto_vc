import cv2
import time
import math as m
import mediapipe as mp

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





