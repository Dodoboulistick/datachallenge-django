#/usr/bin/python3.8
import pickle
import pandas as pd
import numpy as np

#PATH TO THIS FOLDER
PATH = '/'

#one hot encoding of data keywords is known key words 
def encode(keywords: list):
    data = pd.read_csv('./col.csv',index_col='Unnamed: 0')
    data.columns = ['keywords']
    new_data = []
    new_data = data['keywords'].isin(keywords).astype(int)

    return(new_data)

#get possible tag name 
def get_index(array : list):
  #np array named index empty
  index = np.array([])
  #if the element in the list is greater than 0 then we add the index of the element in the list to the np array
  for i in range(len(array)):
      if array[i] > 0.0:
          index = np.append(index, i)
  #return the index
  return(index)
  
def prediction(keywords : list):
  #load model
  loaded_model = pickle.load(open('./models/knn_model.sav','rb'))
  #encode keywords
  data = encode(keywords)
  #reshape the endoded keywords vector
  data = np.array(data).reshape(1, -1)
  #predict the tags with the keywords
  pred = loaded_model.predict_proba(data)

  pred = pred[0]
    
  #get the index of the prediction
  index = get_index(pred)
  #get the classes of the prediction 
  classes = loaded_model.classes_

  #Get the classes of the prediction when the prediction is greater than 0
  classes = classes[index.astype(int)]

  return(classes)
  