import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from dtw import dtw
import csv
import os

base_path = '/Users/ankit/Documents/Projects/Form_Fixer'
coords_path = os.path.join(base_path, 'coords.csv')
model_path = os.path.join(base_path, 'deadlift.pkl')
video_path = os.path.join(base_path, 'Deadlift.mp4')
landmarks_path = os.path.join(base_path, 'deadlift_landmarks.npy')

'''# Capture Landmark and export to CSV
landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

with open(coords_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

def export_landmark(results, action):
    try:
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0, action)

        with open(coords_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_pose = mp.solutions.pose # Mediapipe Solutions
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open reference video file
cap_ref = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap_ref.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    print(f"Successfully opened video file {video_path}")

# Initialize list to store landmarks
landmarks_list = []

# Process video frames to capture reference poses
while cap_ref.isOpened():
    ret, frame = cap_ref.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract landmarks and append to list
        landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
        landmarks_list.append(landmarks)

# Convert list of landmarks to NumPy array and save to file
reference_landmarks = np.array(landmarks_list)
np.save('deadlift_landmarks.npy', reference_landmarks)

# Release reference video resources
cap_ref.release()

# Open video file for live processing
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    print(f"Successfully opened video file {video_path}")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, image = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        k = cv2.waitKey(1)                
        if k == 117:
            export_landmark(results, 'up')
        if k == 100:
            export_landmark(results, 'down')      
        cv2.imshow('Raw Webcam Feed', image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

'''


# Load reference pose landmarks from Deadlift.mp4 (assuming you have them stored)
reference_landmarks = np.load(landmarks_path)
df = pd.read_csv(coords_path)
X = df.drop('class', axis=1) #features

# Extract the landmarks from the DataFrame
webcam_landmarks = df.drop('class', axis=1).to_numpy()

# Load reference pose landmarks from Deadlift.mp4 (assuming you have them stored)
reference_landmarks = np.load(landmarks_path)

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Function to compute DTW distance between two sequences of landmarks
def compute_dtw_distance(landmarks1, landmarks2):
    dist, _, _, _ = dtw(landmarks1, landmarks2, dist=lambda x, y: np.linalg.norm(x - y, ord=2))
    return dist

y = df['class'] #target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

from sklearn.metrics import accuracy_score, precision_score, recall_score #Accuracy metrics
import pickle

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    #print(algo, accuracy_score(y_test.values, yhat), precision_score(y_test.values, yhat, average="binary", pos_label="up"), recall_score(y_test.values, yhat, average="binary", pos_label="up"))

with open(model_path, 'wb') as f:
    pickle.dump(fit_models['rf'], f)

counter = 0
current_stage = ''

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to extract pose landmarks from a frame
def extract_landmarks(frame):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
            return landmarks
        else:
            return None

cap = cv2.VideoCapture(0)

# Open the reference video file
cap_ref = cv2.VideoCapture('Deadlift.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        ret_ref, frame_ref = cap_ref.read()

        if not ret_ref:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        # Extract landmarks from the current frame if pose estimation is successful
        if results.pose_landmarks:
            landmarks_webcam = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()

            # Draw landmarks from the current video onto the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )

            # Perform pose estimation on the reference video frame
            results_ref = pose.process(frame_ref)

            # Extract landmarks from the reference video frame if pose estimation is successful
            if results_ref.pose_landmarks:
                landmarks_ref = np.array([[res.x, res.y] for res in results_ref.pose_landmarks.landmark]).flatten()

                # Draw landmarks and connections from the reference video onto the current video frame
                mp_drawing.draw_landmarks(image, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

        landmarks = ['class']
        for val in range(1, 33+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'down'
            elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage="up"
                counter += 1
                print(current_stage)

            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)] * 100,2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Reps', (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            pass

        # Display the frame with both sets of landmarks
        cv2.imshow('Current Video with Reference Landmarks', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
# Release the webcams and close all OpenCV windows
cap.release()
cap_ref.release()
cv2.destroyAllWindows()


df = pd.read_csv('coords.csv')
X = df.drop('class', axis=1) #features

y = df['class'] #target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

from sklearn.metrics import accuracy_score, precision_score, recall_score #Accuracy metrics
import pickle

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    #print(algo, accuracy_score(y_test.values, yhat), precision_score(y_test.values, yhat, average="binary", pos_label="up"), recall_score(y_test.values, yhat, average="binary", pos_label="up"))

with open('deadlift.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

cap = cv2.VideoCapture(0)
counter = 0
current_stage = ''

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_pose = mp.solutions.pose # Mediapipe Solutions

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
        
        landmarks = ['class']
        for val in range(1, 33+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'down'
            elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage="up"
                counter += 1
                print(current_stage)

            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)] * 100,2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'Reps'
                        , (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter)
                        , (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        except Exception as e:
            pass
        
        cv2.imshow('Raw Webcame Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()

