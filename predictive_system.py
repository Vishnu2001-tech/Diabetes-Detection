import numpy as np
import pickle 

#Load the model
loaded_model = pickle.load(open('D:/Diabetes Classifer/trained_model.sav','rb'))

input = [5,166,72,19,175,25.8,0.587,51]

convert =np.asarray(input)
convert_shaped = convert.reshape(1,-1)

prediction = loaded_model.predict(convert_shaped)

print(prediction)

if(prediction[0]==0):
    print("Person is Diabetic")
else:
    print("Person is Fine")
