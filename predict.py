#inference
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor.h5')

image = cv2.imread('./pred/pred31.jpg')
img = Image.fromarray(image)
img = img.resize((64 , 64))
img = np.array(img)

#print(img)


input_img = np.expand_dims(img , axis=0)
predict_x=model.predict(input_img) 
result=np.argmax(predict_x,axis=1)

if result==0:
    print("brain tumor not found")
else:
     print("brain tumor found")
#print(f'class {result} ')




