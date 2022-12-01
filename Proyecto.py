import cv2
import time
import math as m
import mediapipe as mp
import os

# Definimos si deseamos partir de una grabación ya realizada (recorded = True) o si deseamos realizar una grabación (recorded = False) para evaluar la postura
# print('Introduzca si desea operar con un video grabado [y] o si desea grabar uno nuevo [N]')
# answ = input()
# if answ == 'y':
#     recorded = True
#     print('Introduzca la ruta en la que se encuentra la grabación')
#     record = input()
#     while os.path.exists(record) == False:
#         print('Introduzca una ruta valida')
#         record = input()

#     if os.path.exists(record) == True:
#         record = input()
# if answ == 'N':
#     recorded = False
# if answ != "y" and answ != "N":
#     print('Introduzca si desea operar con un video grabado [y] o si desea grabar uno nuevo [N]')


# Iniciamos la grabación para evaluar la postura
# def grabar():
#     parar = False
#     if recorded == False:
#         vid = cv2.VideoCapture(1)
#         fps = vid.get(cv2.CAP_PROP_FPS)  # Definir fps en funcion de la source
#         size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Definir size en funcion de la source
#         videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#         while (True):
#             if not parar:
#                 ret, frame = vid.read()

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             if cv2.waitKey(1) & 0xFF == ord('s'):  # Si pulsamos s parar=true
#                 if parar:
#                     parar = False
#                 else:
#                     parar = True

#             cv2.imshow('frame', frame)  # Display the resulting frame
#             videoWriter.write(frame)  # Escribimos en el video generado con el writer los frames

# # Definir el nombre (y ruta) del archivo en caso de que recorded == True
# if recorded == False:
#     grabar()

# vid = cv2.VideoCapture(0)
# fps = vid.get(cv2.CAP_PROP_FPS)
# size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# stop = False
# frame = None

# while(True):
#     if not stop:
#         ret, frame = vid.read()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         if stop:
#             stop = False
#         else:
#             stop = True

#     cv2.imshow('frame', frame)
#     videoWriter.write(frame)

# vid.release()

# cv2.destroyAllWindows()


# Calcular distancia offset
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# Calcular inclinacion de la postura corporal
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree


# # Function to Send Poor Body Posture Alerts	
# def sendWarning(x):
# pass

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

# For webcam input replace file name with 0.
# file_name = 'output.avi'
cap = cv2.VideoCapture(0)
# Meta.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer.
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
while(True):
# Capture frames.
    success, image = cap.read()
    if not success:
        print("Null.Frames")
    # Get fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get height and width of the frame.
    h, w = image.shape[:2]

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image.
    keypoints = pose.process(image)

    # Convert the image back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Body Posture Landmark Coordinates
    mp_pose.Pose().process(image).pose_landmarks
    # norm_coordinate  = pose.process(image).pose_landmark.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].coordinate

    # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    if (lm is not None):
        lmPose  = mp_pose.PoseLandmark
        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)

        # Right shoulder.
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)

        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * width)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * height)

        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * width)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * height)


    # Alinear Cámara

    # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (width - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (width - 150, 30), font, 0.9, red, 2)


        # Calculate Body Posture Inclination and Draw Landmarks

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))


        # Body Posture Detectino Conditionals

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            # Join landmarks.
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

            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, height - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, height - 20), font, 0.9, red, 2)
   
   
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
    # if bad_time > 180:
    #     sendWarning()
cap.release()
cv2.destroyAllWindows()
