
#use them on gooogle collbab  https://colab.research.google.com/drive/1E5Sqm1wKs61PvAEbuxk1KeRO6zBUbVQz


link->
#https://colab.research.google.com/drive/1E5Sqm1wKs61PvAEbuxk1KeRO6zBUbVQz?usp=sharing


import pandas as pd
from sklearn.impute import SimpleImputer

#data
data={
    'Feature1': [1.0,2.0,None,4.0,5.0],
     'Feature2' : [2.0,None,4.0,5.0,None],
      'Feature3': [None,4.0,5.0,3.0,3.5]
}


df=pd.DataFrame(data)

imputer = SimpleImputer(strategy='mean')
df_imputed=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
print(df)
print("after imputation")
print(df_imputed)



#encoding categorial variables

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data={
    'Color':['Red','Blue','Green','Blue','Red']
}
df=pd.DataFrame(data)
print(df)
print("After one hot coding")

encoder= OneHotEncoder(sparse=False)
encoded_categories=encoder.fit_transform(df[['Color']])
df_encoded=pd.DataFrame(encoded_categories,columns=encoder.get_feature_names_out(['Color']))
df=pd.concat([df,df_encoded],axis=1).drop('Color',axis=1)
print(df)



#FEATURE SCALING
#min-max

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Correcting the data dictionary
data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [2.0, 4.0, 6.0, 8.0, 10.0]
}

df = pd.DataFrame(data)
print(df)

# Feature scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("After Min-Max Scaling")
print(df_scaled)


#4. Feature Creation

#Feature creation is the process of creating new features from existing features.

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

# Feature creation
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df)
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Feature1', 'Feature2']))

print("After Creating Polynomial Features:\n", df_poly)



#FEATURE SELECTION
#Feature selection is the process of selecting a subset(BEST) of features from a larger set of features.

#1. VARIANCE THRESHOLDING
#Variance thresholding is a feature selection method that removes features with low variance.







#2.CORRELATION MATRIX
#Correlation matrix is a matrix that shows the correlation between features. Removing correlated data to reduce reduntancy





#3 Domain knowledge
#Domain knowledge is the knowledge of the domain of the problem. 