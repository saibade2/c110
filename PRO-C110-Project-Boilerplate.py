import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()

    # Modify the input data by:

    # 1. Resizing the image

    img = cv2.resize(frame,(224,224))

    # 2. Converting the image into Numpy array and increase dimension

    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    # 3. Normalizing the image
    normalised_image = test_image/255.0

    # Predict Result
    prediction = model.predict(normalised_image)

    rock = int(prediction[0][0]*100) 
    paper = int(prediction[0][1]*100) 
    scissor = int(prediction[0][2]*100) 
    # printing percentage confidence print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

    print("Prediction : ", prediction)
        
    cv2.imshow("Result",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Closing")
        break

video.release()
