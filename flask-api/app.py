import flask
import os
from flask import jsonify, request
from flask import flash, redirect, url_for, session
from joblib import load
from flask_cors import CORS, cross_origin
import requests, json
import pandas as pd
import requests



predictions = pd.read_csv("predictionApi.csv",index_col=0)
prof = pd.read_csv("profileApi.csv")
hotelData = pd.read_csv("hotels-dataset.csv")
hotelData.set_index('id', inplace=True)
prof.set_index('userID', inplace=True)



app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = 'super secret key'
cors = CORS(app, resources={r"/*": {"origins": "*"}})







@app.route('/test', methods=['GET','POST'])
def test():
    print(prof.head())
    data = [ 1 , 2 , "Buckle My Shoe" , 3 , 4 , "Shut the Door" ]
    return jsonify( data )







@app.route('/predict', methods=['GET'])
def predict():

    # print( json.dumps( request.json['data'] ) )

    try :
        print("hi")

        user = request.args.get('user')

        myProfile = {
            "U1001" : {
                "name" : "Riya Patil",
                "password" : "123"
            },
            "U1002" : {
                "name" : "Prachiti Patil",
                "password" : "123"
            },
            "U1003" : {
                "name" : "Amey Patil",
                "password" : "123"
            },
            "U1004" : {
                "name" : "Priyanka Patil",
                "password" : "123"
            }
        }
        userData = ( prof.loc[user , :].to_json() )
        userData = json.loads(userData.replace("\'", '"'))

        hotelList = predictions[user]
        # print(hotelList[:10])
        arr = ( hotelList.sort_values(ascending=False)[:10].index )
        # print(arr)
        hotels = hotelData.loc[ arr , : ].to_dict('records')

        # hotels = json.loads(hotels.replace("\'", '"'))
        print(hotels)
        # hot = {}

        # i = 0
        # for row in hotels:
        #     hot[i] = row
        #     i += 1
        
        # # print(hot)

        return jsonify( { "userData" : userData, "profile" : myProfile[user],"hotels" : hotels , "status"  : True } )

    except Exception as e:
        return jsonify( { "result" : "error" , "status"  : False  } )      





@app.route('/', methods=['GET'])
def home():
    print("loaded")
    return "Welcome to My API"




if __name__ == '__main__':
    app.run()
