#importamos las librerias necesarias
import pickle 
import mediapipe as mp 
import cv2 
import numpy as np
import pandas as pd


#cargamos el modelo previamente utilizado usando pickle
with open('exercises.pkl', 'rb') as f:
    model = pickle.load(f)

#importamos los modulos necesarios para el procesamiento y dectecion de landmarks corparales
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 

#creamos un objeto capture para ver la camara web
cap = cv2.VideoCapture("videos\postura del triangulo.mp4")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #Se inicia un bucle infinito para procesar los fotogramas de video:
    while cap.isOpened():
        #Se lee un fotograma del video:
        ret, frame = cap.read()
        #Se convierte el fotograma de BGR a RGB y se realiza el procesamiento con holistic.process:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        results = holistic.process(image)
     
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #Se dibujan los landmarks corporales y las conexiones en la imagen:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        try:

            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            row = pose_row
            

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            # print(body_language_class, body_language_prob)
            
            # Se extraen las coordenadas del oído izquierdo y se dibuja un rectángulo y texto en la imagen para mostrar la clase de lenguaje corporal detectada
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
     
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            #Se muestran en la imagen el nombre de la clase y la probabilidad de detección:
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Exercise', image)
        #El bucle continúa hasta que se presione la tecla 'q' para salir:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
#Se liberan los recursos y se cierran las ventanas:
cap.release()
cv2.destroyAllWindows()
#Esta línea multiplica las coordenadas normalizadas del oído izquierdo por las dimensiones de la imagen (640x480) y las convierte en una tupla de enteros.
tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640,480]).astype(int))