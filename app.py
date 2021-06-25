import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv('heart.csv')

st.text("author: Siranjeevi")
st.title("Heart Disease Checkup")
st.header('Patient Data')

x = df.drop(['target'], axis=1)
y = df['target']


sex_dictionary = {'Male':1, 'Female':0}
fasting_dictionary = {'true':1, 'false':0}
ex_dict = {'yes':1, 'no':0}

def user_report():
  age = st.slider('Age', 29,77, 30)
  sex = st.slider('Sex (Male=1, Female=0)', 0, 1, 0 )
  cp = st.slider('Chest Pain Type (4 values)', 0,3, 1 )
  trtbps = st.slider('Resting Blood Pressure (in mm Hg on admission to the hospital)', 94,200, 120 )
  chol = st.slider('Serum Cholestrol in mg/dl',126, 564, 200)
  fbs = st.slider('fasting blood sugar &gt; 120 mg/dl (True = 1, False=0)', 0,1 ,0)
  restecg = st.slider('Resting electrocardiographic results (values 0,1,2)', 0,2, 0 )
  thalachh = st.slider('Maximum heart rate achieved', 71,202, 110 )
  exng = st.slider('exercise induced angina (yes=1, no=0)', 0, 1, 0 )
  oldpeak = st.slider('oldpeak = ST depression induced by exercise relative to rest', 0,6, 2 )
  slp = st.slider('the slope of the peak exercise ST segment', 0, 2, 0)
  caa = st.slider('number of major vessels (0-3) colored by flourosopy', 0, 4, 0)
  thall = st.slider('thal: 3 = normal; 6 = fixed defect; 7 = reversable defect', 0, 3, 0)



  user_report_data = {
	  'age': age,
	  'sex': sex,
	  'cp': cp,
	  'trtbps': trtbps,
	  'chol': chol,
	  'fbs': fbs,
	  'restecg': restecg,
	  'thalachh': thalachh,
	  'exng': exng,
	  'oldpeak': oldpeak,
	  'slp': slp,
	  'caa': caa,
	  'thall': thall
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

kn = KNeighborsClassifier()
kn.fit(x, y)
user_result = kn.predict(user_data)





# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


 























# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are in Good Condition'
else:
  output = 'You Have Heart Disease'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y, kn.predict(x))*100)+'%')











