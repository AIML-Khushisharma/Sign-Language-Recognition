#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow opencv-python mediapipe scikit-learn matplotlib')


# In[2]:


import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# In[3]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# In[4]:


def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results


# In[5]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))


# In[6]:


cap =  cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
      ret,frame = cap.read()
      image, results = mediapipe_detection(frame, holistic)
      print(results)
    
      draw_landmarks(image, results)
    
      cv.imshow('Video', image)

      if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# In[7]:


len(results.face_landmarks.landmark)


# In[8]:


draw_landmarks(frame, results)


# In[9]:


plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))


# In[10]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# In[11]:


len(extract_keypoints(results))


# In[14]:


Data_path = os.path.join('Language Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30


# In[15]:


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(Data_path, action, str(sequence)))
        except:
            pass


# In[16]:


cap =  cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret,frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                draw_landmarks(image, results)
                
                if frame_num==0:
                    cv.putText(image, 'Starting Collection', (120,200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv.LINE_AA)
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
                    cv.imshow('Video', image)
                    cv.waitKey(2000)
                else:
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
                    cv.imshow('Video', image)
                
                keypoints = extract_keypoints(results)
                numpy_path = os.path.join(Data_path, action, str(sequence), str(frame_num))
                np.save(numpy_path, keypoints)

                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv.destroyAllWindows()


# In[17]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[18]:


label_map = {label:num for num,label in enumerate(actions)}


# In[19]:


label_map


# In[20]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(Data_path, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[21]:


np.array(sequences).shape


# In[22]:


np.array(labels).shape


# In[23]:


X = np.array(sequences)


# In[24]:


X.shape


# In[25]:


y = to_categorical(labels).astype(int)


# In[26]:


y


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[42]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# In[30]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)


# In[63]:


model = Sequential()

# Add LSTM layers with different configurations
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(Dropout(0.2))  # Add dropout regularization
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))  # Add dropout regularization

# Add Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add output layer
model.add(Dense(actions.shape[0], activation='softmax'))


# In[64]:


optimizer = Adam(learning_rate=0.001)  # Try different learning rates
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[46]:


model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])


# In[47]:


model.summary()


# In[48]:


res = model.predict(X_test)


# In[58]:


actions[np.argmax(res[3])]


# In[59]:


actions[np.argmax(y_test[3])]


# In[60]:


model.save('signlanguage.h5')


# In[61]:


model.save('actions.keras')


# In[62]:


del model


# In[65]:


model.load_weights('signlanguage.h5')


# In[66]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[67]:


yhat = model.predict(X_test)


# In[68]:


ytrue = np.argmax(y_test, axis = 1).tolist()
yhat = np.argmax(yhat, axis = 1).tolist()


# In[69]:


multilabel_confusion_matrix(ytrue, yhat)


# In[75]:


accuracy_score(ytrue, yhat)


# In[78]:


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv.putText(output_frame, actions[num], (0, 85+num*40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        
    return output_frame


# In[79]:


plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))


# In[80]:


sequence.reverse()
len(sequence)
sequence.append('def')
sequence.reverse()
sequence[-30:]


# In[82]:


sequence = []
sentence = []
thresold = 0.4

cap =  cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        print(results)
    
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence)==30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            if res[np.argmax(res)] > thresold:
                if(len(sentence)) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
            
            image = prob_viz(res, actions, image, colors)
            
        cv.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv.putText(image, ' '.join(sentence), (3,30), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)        
    
        cv.imshow('Video', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()

