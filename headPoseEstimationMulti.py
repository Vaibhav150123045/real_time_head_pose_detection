import cv2
import mediapipe as mp
import numpy as np
import time
from manualThreading import WebcamStream

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = WebcamStream(0)
cap.start()

while cap.stopped is not True:
    image = cap.read()
    start = time.time()

    # flip the image horizontally for a mirrored view
    # convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # to improve performance, mark the image as not writeable
    image.flags.writeable = False

    # Get the results from the face mesh model
    results = face_mesh.process(image)

    # To improve performance, mark the image as writeable
    image.flags.writeable = True

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                if id == 33 or id == 263 or id == 61 or id == 291 or id == 1 or id == 199:
                    #  chin landmark (id 1)
                    #  nose tip landmark(id 199)
                    #  left and right eye landmarks(id 61 and 291)
                    #  left and right eyebrow landmarks(id 33 and 263)

                    if id == 1:
                        nose_2d = [lm.x * img_w, lm.y * img_h]
                        # Scale z for better visualization
                        nose_3d = [lm.x * img_w, lm.y * img_h, lm.z*3000]

                # Get the 3D coordinates
                x, y, z = lm.x * img_w, lm.y * img_h, lm.z
                face_3d.append([x, y, z])

                # Get the 2D coordinates
                face_2d.append([x, y])

            face_3d = np.array(face_3d)
            face_2d = np.array(face_2d)

            face_2d = np.array(face_2d, dtype=np.float32)
            face_3d = np.array(face_3d, dtype=np.float32)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float32)

            # Define the distortion coefficients
            dist_matrix = np.zeros((4, 1), dtype=np.float32)

            # Solve the PnP problem to get the rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get the rotation angles
            # Decompose the rotation matrix to get the Euler angles
            # and the rotation matrix
            # Qx, Qy, Qz are the Euler angles in radians
            # mtxR is the rotation matrix
            # mtxQ is the quaternion representation of the rotation
            # angles is a tuple of the Euler angles in radians
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print(f"Rotation Angles (radians): {angles}")
            x = angles[0] * 360  # Convert radians to degrees
            y = angles[1] * 360  # Convert radians to degrees
            z = angles[2] * 360  # Convert radians to degrees
            print(f"Rotation Angles: X: {x}, Y: {y}, Z: {z}")
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x > 10:
                text = "Looking Up"
            elif x < -10:
                text = "Looking Down"
            else:
                text = "Forward"

            # Display the nose position and orientation
            nose_3d_projection, jacobian = cv2.projectPoints(
                np.array(nose_3d, dtype=np.float32), rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*10))

            cv2.line(image, p1, p2, (0, 255, 0), 3)

            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"X: {int(x)} Y: {int(y)} Z: {int(z)}",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        end = time.time()
        total_time = end - start
        fps = 1 / total_time

        cv2.putText(image, f"FPS: {fps:.2f}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the face mesh
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.stop()
