from flask import Flask, render_template
import pandas as pd
import pickle
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler 
import numpy as np
                           


app = Flask(__name__)

@app.route("/tables")
def show_tables():
   
    #reading the test file
    df_events_test = pd.read_pickle("df_test.pkl")
    print(df_events_test.head())
   
    # load the vectorizer

    #loading the brand encoder file to transform test data
    events_brand_encoder = pickle.load(open('events_brand_encoder.pkl','rb'))
    events_brand_test = events_brand_encoder.transform(df_events_test['brand'])

    #loading the model encoder file to transform test data
    events_model_encoder = pickle.load(open('events_model_encoder.pkl','rb'))
    events_model_test = events_model_encoder.transform(df_events_test['model'])

    #loading the active labels encoder file to transform test data
    events_active_labels_encoder = pickle.load(open('events_active_labels_encoder.pkl','rb'))
    events_active_labels_test = events_active_labels_encoder.transform(df_events_test['active_app_labels'])

    #loading the installed labels encoder file to transform test data
    events_installed_labels_encoder = pickle.load(open('events_installed_labels_encoder.pkl','rb'))
    events_installed_labels_test = events_installed_labels_encoder.transform(df_events_test['installed_app_labels'])

    scaler = StandardScaler()
  
    events_lat_test = scaler.fit_transform(df_events_test['mean_latitude'].values.reshape(-1,1))
   
    events_long_test = scaler.fit_transform(df_events_test['mean_longitude'].values.reshape(-1,1))
   
    events_travels_test = scaler.fit_transform(df_events_test['num_travels'].values.reshape(-1,1))

    

    events_x_test = hstack((events_brand_test, events_model_test, events_installed_labels_test, events_active_labels_test,
                         events_lat_test, events_long_test, events_travels_test, df_events_test['location_available'].values.reshape(-1,1), 
                         np.array(df_events_test['activity_hour'].to_list()), np.array(df_events_test['activity_day'].to_list()), 
                         df_events_test['app_usage'].values.reshape(-1,1), df_events_test['app_usage_session'].values.reshape(-1,1), 
                         np.array(df_events_test['installed_app_onehot'].to_list()), 
                         np.array(df_events_test['active_app_onehot'].to_list()))) 

    #events_model_xgb = "Events - RBF XGBRegressor"
    #age prediction
   
    model_age = pickle.load(open('xgboost_age-jupyter.pkl', 'rb'))
    age_pred_test = model_age.predict(events_x_test)

    print(age_pred_test)
    df_events_test['predict_age'] = age_pred_test

    #gender prediction
   
    model_gender = pickle.load(open('xgboost_gender_jupyter.pkl', 'rb'))
    predict_y_proba = model_gender.predict_proba(events_x_test)
    predict_gender = model_gender.predict(events_x_test)
    print(predict_gender)
    df_events_test['predict_gender'] = predict_gender

   #assigning predict probability to female and male prediction
    df_events_test['female_prediction'] = predict_y_proba[:,0]
    df_events_test['male_prediction'] = predict_y_proba[:,1]
    print(df_events_test.head())
    

    df_events_test.set_index(['device_id'], inplace=True)
    df_events_test.index.name=None

    #assigning class 0 as F and class 1 as M
    df_events_test['predict_gender'] = df_events_test['predict_gender'].apply(lambda x: 'M' if x == 1 else 'F') 


    dfgender = df_events_test[["brand","model","gender","age","group","predict_gender","female_prediction","male_prediction"]]
    dfgenderM = dfgender.loc[(dfgender.male_prediction >= 0.6)]
    dfgenderF = dfgender.loc[ (dfgender.female_prediction >=0.5)]

    print(dfgender.shape)
    #creating agebin for predicted age
    df_events_test['agebin'] = pd.cut(df_events_test['predict_age'], [0,24, 32, 100], labels=['0-24', '25-32', '32+'])
    dfage = df_events_test[["brand","model","gender","age","group","predict_age", "agebin"]]

    agebin1 =df_events_test.loc[df_events_test.agebin=='0-24']
    agebin1 = agebin1[["brand","model","gender","age","group","predict_age","agebin"]]

    agebin2 =df_events_test.loc[df_events_test.agebin=='25-32']
    agebin2 = agebin2[["brand","model","gender","age","group","predict_age","agebin"]]
    
    agebin3 =df_events_test.loc[ df_events_test.agebin=='32+']
    agebin3 = agebin3[["brand","model","gender","age","group","predict_age","agebin"]]

   
    return render_template('view.html',tables=[ dfgender.to_html(classes='genderpred'),  
    dfage.to_html(classes='agepred'),dfgenderM.to_html(classes='male'), dfgenderF.to_html(classes='female') ,agebin1.to_html(classes='agebin1'),  
    agebin2.to_html(classes='agebin2'), agebin3.to_html(classes='agebin3')], titles = ['na', "Gender Prediction","Age prediction",
    "Gender Prediction-Campaign 3: Personalised call and data packs targeting male customers", "Campaign 1: Specific personalised fashion-related campaigns targeting female customers & Campaign 2: Specific cashback offers on special days",
    "Campaign 4: Bundled smartphone offers for the age group [0-24]",
    "Campaign 5: Special offers for payment wallet offers [24-32]","Campaign 6: Special cashback offers for Privilege Membership [32+]"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")