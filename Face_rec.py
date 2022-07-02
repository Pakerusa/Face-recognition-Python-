import face_recognition
import cv2
import numpy as np

video_cap = cv2.VideoCapture(0)

Pakerusa_image = face_recognition.load_image_file("Pakerusa.jpg")
Pakerusa_face_encoding = face_recognition.face_encodings(Pakerusa_image)[0]

# you can add More People.like that:
# 'NAME' _image = face_recognition.load_image_file(" 'IMAGE' ")
# 'NAME' _face_encoding = face_recognition.face_encodings( 'NAME' _image)[0]

known_face_encodings = [
    Pakerusa_face_encoding
    # Add more Names
    # dont forget the ,
]
known_face_names = [
    "Pakerusa"  
    # Add more Names
    # dont forget the ,
]

#face data is stored here:
face_pos = []
face_encodings = []
face_names = []
process_this_frame = True

# loop gets started

while True:
    
    ret, frame = video_cap.read()

    
    if process_this_frame:
    
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        #get locations
        face_pos = face_recognition.face_locations(rgb_small_frame)
        face_pos = face_recognition.face_encodings(rgb_small_frame, face_pos)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # Unkown faces get named Unknown 
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_pos, face_names):
 
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw rectangle anround face 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show the prossesed video 
    cv2.imshow('Face Rec', frame)

    # Press "E" to close window  
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


# clese Window
video_cap.release()
cv2.destroyAllWindows()