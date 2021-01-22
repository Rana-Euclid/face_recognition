import cv2
import face_recognition
import numpy as np
import os
import time
from PIL import Image

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Initialize some variables
        #self.face_locations = []
        #self.face_encodings = []
        #self.face_names = []
        self.process_this_frame = True
        # Create face encodings for all pics present in the face_db folder
        self.known_face_encodings = []
        self.known_face_names =[]
        self.face_names_dict = dict()
        for f in os.listdir('face_db/'):
            self.known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file('face_db/'+f))[0])
            self.known_face_names.append(f.split('.')[0])
    
    def __del__(self):
        self.video.release()

    # function for video streaming
    def get_frame(self):
        _, frame = self.video.read()
        if self.process_this_frame:
            frame,  face_names = self.process_frames(frame)
        self.process_this_frame = not self.process_this_frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        # Persistence and delay logic along with sort
        there = 30
        not_there = 150
        show_names=dict()
        for name in face_names:
            if name not in self.face_names_dict:
                self.face_names_dict[name] = (1,0)
            else:
                self.face_names_dict[name] = (self.face_names_dict[name][0]+1, self.face_names_dict[name][1])
        for name in self.face_names_dict:
            if name not in face_names:
                self.face_names_dict[name] = (self.face_names_dict[name][0], self.face_names_dict[name][1]+1)
            if self.face_names_dict[name][1]>not_there:
                self.face_names_dict[name] = (0, 0)
            if self.face_names_dict[name][0]>there:
                show_names[name]=self.face_names_dict[name][0]
        show_names = sorted(show_names, key=show_names.get)
        show_names_list = list(show_names)
        if 'Unknown' in show_names_list:
            show_names_list.remove('Unknown')
        print(show_names_list)
        return jpeg.tobytes(), show_names_list

    #function for processing frames
    def process_frames(self,frame):
        face_locations = []
        face_encodings = []
        face_names=[]
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)
        
        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, top -35), (right, top), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top-2), font, 1.0, (255, 255, 255), 1)
        
        return frame , face_names




