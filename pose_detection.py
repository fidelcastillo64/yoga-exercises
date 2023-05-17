import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
#importamos el modulo para dibujar puntos y lienas en las los videos
mp_drawing = mp.solutions.drawing_utils 
#holistic se encarga de dibujar los puntos de referencia de la postura del video
mp_holistic = mp.solutions.holistic 
#postura del triangulo "triangle pose"
#postura del warrior pose "warrior pose"
#postura del arbol "warrior pose"
#definimos un clase name adecuadamanet a la posicion que queremos detectar
class_name= "warrior pose"
#indicamos la ruta del video a estudiar
cap = cv2.VideoCapture("videos\posicion del guerrero II 1.mp4")
# Initiate holistic model

#definimos la confianza mínima requerida para detectar un objeto en la imagen
#indica la confianza mínima requerida para realizar el seguimiento de un punto de referencia de la postura a lo largo del tiempo.
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        #le damos a la imagen un formato RGB debido que el modelo de Mediapipe espera imágenes en formato RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #desactivamos la capacidad de estritura de la imagen para ahorra procesamiento
        image.flags.writeable = False        

        #usamos holistic para porcesar el Frame de y obtenemos los resultados
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # reactivamos la capacidad de lectura
        image.flags.writeable = True   
        #Aquí se realiza la transformación de color de vuelta a BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #realizamos el dibujo sobre la imagen importando los resultados, una constante de mediapipe
        #por ultimo definimos caracteristicas de dibujo como el radio de los puntos, color y grosor de las lineas
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        try:
            #obtenemos los puntos detectados en el Frame
            pose = results.pose_landmarks.landmark
            #guardamos las cordenadas en una lista plana para el .CSV
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
        
            # Concate rows
            row = pose_row
            
            # añadimos el nombre de la clase al indice 0
            row.insert(0, class_name)
            
            #exportamos las cordenadas al .CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
        #mostramos el frame de la imagen             
        cv2.imshow('Raw Webcam Feed', image)

        #indicamos que se pueda romper el bucle si se presiona Q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()