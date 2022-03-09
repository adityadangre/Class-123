import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps

x, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 2500, train_size = 7500)
x_train_scalled = x_train / 255
x_test_scalled = x_test / 255

classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scalled, y_train)

y_predict = classifier.predict(x_test_scalled)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)


cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape 

        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 3)

        roi = gray[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

        im_pil = Image.fromarray(roi)
        im_bw = im_pil.convert('L')
        im_bw_resized = im_bw.resize((28, 28), Image.ANTIALIAS)
        im_bw_resized_inverted = PIL.ImageOps.invert(im_bw_resized)

        pixel_filter = 20
        min_pixel = np.percentile(im_bw_resized_inverted, pixel_filter)
        im_bw_resized_inverted_scaled = np.clip(im_bw_resized_inverted - min_pixel, 0, 255)

        max_pixel = np.max(im_bw_resized_inverted)
        im_bw_resized_inverted_scaled = np.asarray(im_bw_resized_inverted_scaled) / max_pixel
        
        test_sample = np.array(im_bw_resized_inverted_scaled).reshape(1, 784)
        test_predict = classifier.predict(test_sample)
        print('Predicted Class is', test_predict)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.DestroyAllWindows()