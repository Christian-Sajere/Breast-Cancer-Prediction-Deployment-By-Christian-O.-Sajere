import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df=pd.read_csv('data.csv')


#Encode the categorical data value
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)


#split the dataset into independent (X) and dependent column (Y) data sets.
#Y has the dagonostic whether the patient have cancer or not.While X have the features for cancer detection
X = df.iloc[:,2:31].values
Y=df.iloc[:,1].values


#Split the dataset into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25 ,random_state =0)


#Scale the data (Feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#create a function for the models
def models(X_train, Y_train):
    
    #logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    #Decision Tree classicifier
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion= 'entropy', random_state=0)
    tree.fit(X_train,Y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest= RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
    forest.fit(X_train,Y_train)
    

    return log, tree,forest

st.write("""# Breast Cancer Prediction Software""") 

st.write("    ")

images=open('cancer-2D_image.png','rb').read()
st.image(images, width=300)







radius_mean=st.text_input('radius mean')                  

texture_mean=st.text_input('texture mean')                   

perimeter_mean=st.text_input('perimeter mean')                

area_mean=st.text_input('area mean')

smoothness_mean=st.text_input('smoothness mean')

compactness_mean=st.text_input('compactness mean')

concavity_mean=st.text_input('concavity mean')

concave_points_mean=st.text_input('concave points mean')

symmetry_mean=st.text_input('symmetry mean')

fractal_dimension_mean=st.text_input('fractal dimension mean')




radius_se=st.text_input('radius se')                  

texture_se=st.text_input('texture se')                   

perimeter_se=st.text_input('perimeter se')                

area_se=st.text_input('area se')

smoothness_se=st.text_input('smoothness se')

compactness_se=st.text_input('compactness se')

concavity_se=st.text_input('concavity se')

concave_points_se=st.text_input('concave points se')

symmetry_se=st.text_input('symmetry se')

fractal_dimension_se=st.text_input('fractal dimension se')




radius_worst=st.text_input('radius_worst')                  


texture_worst=st.text_input('texture_worst')                   


perimeter_worst=st.text_input('perimeter_worst')                


area_worst=st.text_input('area worst')


smoothness_worst=st.text_input('smoothness worst')


compactness_worst=st.text_input('compactness worst')


concavity_worst=st.text_input('concavity worst')


concave_points_worst=st.text_input('concave points worst')


symmetry_worst=st.text_input('symmetry worst')


fractal_dimension_worst=st.text_input('fractal dimension worst')





arr={'radius_mean':[radius_mean],'texture_mean':[texture_mean],'perimeter_mean':[perimeter_mean],'area_mean':[area_mean],'smoothness_mean':[smoothness_mean],'compactness_mean':[compactness_mean],'concavity_mean':[concavity_mean],
     'concave points_mean':[concave_points_mean],'symmetry_mean':[symmetry_mean],'fractal_dimension_mean':[fractal_dimension_mean],
     'radius_se':[radius_se],'texture_se':[texture_se],'perimeter_se':[perimeter_se],'area_se':[area_se],'smoothness_se':[smoothness_se],'compactness_se':[compactness_se],'concavity_se':[concavity_se],'concave points_se':[concave_points_se],
     'symmetry_se':[symmetry_se],'fractal_dimension_se':[fractal_dimension_se],
     'radius_worst':[radius_worst],'texture_worst':[texture_worst],'perimeter_worst':[perimeter_worst],'area_worst':[area_worst],'smoothness_worst':[smoothness_worst],'compactness_worst':[compactness_worst],'concavity_worst':[concavity_worst],
     'concave points_worst':[concave_points_worst],'symmetry_worst':[symmetry_worst],'fractal_dimension_worst':[fractal_dimension_worst]}


dg=pd.DataFrame(arr)

model = models(X_train, Y_train)



X = dg.iloc[:,0:29].values

if st.button("Make Prediction"): 
    st.write("Decision Tree training Prediction")
    sam=(model[2].predict(X))
    tom=accuracy_score(Y_test, model[2].predict(X_test))*100
    
    if sam == 1:
        st.write("You have  "+str(tom)+"   percent chance that you have Breast Cancer")
    else:
        st.write("You have  "+str(tom-100 )+"   percent chance that you are DO NOT HAVE Breast Cancer")

st.write("       ")

st.write("       ")

st.write("       ")

st.write("       ")

st.write("""
#  Context
Breast Cancer Wisconsin (Diagnostic) Data Set features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
Attribute Information:


1) ID number

2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)
The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.
Missing attribute values: none
Class distribution: 357 benign, 212 malignant.

""")


   
   

  