


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

# TYPES OF DATA :--- PHONE , LAPTOPS , CAMERA , HARD DRIVE , GRAPHIC CARD , INTERNATIONAL AIRLINES
data= pd.read_excel("laptop.xlsx")
data['Category']='laptop'
data_phone=pd.read_excel("phone.xlsx")
data_phone['Category']='phone'
data_phone['Product']=data_phone['Company'].map(str) +str(' ')+data_phone['Product'].map(str)
data_camera=pd.read_excel("camera.xlsx")
data_camera['Category']='camera'
data_camera['Company']=data_camera['Product'].apply(lambda x: str(x).split()[0])
data_hard_drive=pd.read_excel("hard_drive.xlsx")
data_hard_drive['Category']='Hard Drive'
data_graphic_card=pd.read_excel("graphic_card.xlsx")
data_graphic_card['Category']='Graphic Card'
data_airline=pd.read_excel("international_airlines.xlsx")
data_airline['Category']='Airline'

# ALL DATA IS CONCATENATED INTO 'data2'
data2=pd.concat([data,data_camera,data_hard_drive,data_graphic_card,data_airline,data_phone],sort=True)
data2['Product']=data2['Product'].apply(lambda x: str(x))
data2 = data2.sample(frac=1).reset_index(drop=True)

# COMPANY NAMES ARE ENCODED USING LABEL ENCODER AND STORED IN ANOTHER COLUMN NAMED 'Company2'
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data2['Company'])
le.classes_
data2['Company2']=le.transform(data2['Company'])
#print(data2.tail())



#PIPELINING IS USED TO TRAIN THE MODEL
# RANDOM FOREST CLASSIFIER IS USED
# PRODUCT MODELS ARE VECTORIZED USING CountVectorizer()
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer()),
                     ('clf', RandomForestClassifier(n_estimators=10)),
])
classifier.fit(data2['Product'],data2['Company2'])



# FUNCITON TO PRINT CATEGORY TREE
def get_category_tree(model):
    prediction=classifier.predict([model])
    df=data2.loc[data2['Company2']==prediction[0]]
    tree={
        "category":df['Category'].value_counts(),
        "brand":df.iloc[0,1]
    }
    print(tree)
    
   
    
# INFINITE LOOP FOR TAKING THE INPUT 

while(True):
    item_model= input("Enter the item_model -> ")
    get_category_tree(item_model)
    


