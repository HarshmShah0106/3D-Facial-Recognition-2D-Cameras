import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import mediapipe as mp
import tensorflow

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

webcam=cv2.VideoCapture(0)

sfr = SimpleFacerec()
sfr.load_encoding_images("D:\Hackathon4\Test IMG")

while True:
    
    success,img = webcam.read()
    
    ret , frame = webcam.read()
    
     #Detect Faces
    face_location , face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_location, face_names):
        y1 , x2 , y2 , x1 = face_loc[0] , face_loc[1], face_loc[2],face_loc[3]

        cv2.putText(frame, name , (x1 , y1 - 10), cv2.FONT_HERSHEY_DUPLEX,1,(0,0,200),2)
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,200),4) #4 is used for rectangle width determination
        cv2.imshow("Frame",frame)
    # applying face mesh model using MediaPipe

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    result = mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True).process(img)


    # draw annotations on the image (Contours for Face Outline, Tesseleation for Face Mesh, Iris for Eyes)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if result.multi_face_landmarks:
        
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        )


            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()

            )

    cv2.imshow("Hackathon",img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


webcam.release()
cv2.destroyAllWindows