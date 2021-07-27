import os
from flask import Flask,request,Response
from flask_cors import CORS,cross_origin
import json
from icecream import ic
from utils.utils import create_user_project_dir,extractFromUserData,trainingModel
from wsgiref import simple_server

app=Flask(__name__)
CORS(app)

trainingDataFolderpath='training_data/'

@app.route('/train',methods=['POST'])
@cross_origin()
def trainModel():
    try:
        if request.get_json() is not None:
            data=request.json['data']
        if request.json['userId'] is not None:
            user=str(request.json['userId'])
            #ic(request.json['userId'])
        if request.json['projectId'] is not None:
            project = str(request.json['projectId'])
            #ic(request.json['projectId'])

        ##userid and projectid dir create
        create_user_project_dir(user,project)

        path=trainingDataFolderpath+user+'/'+project
        #ic(path)

        #data conversion
        train_data_dict=extractFromUserData(data)
        #ic(train_data_dict)

        #saving as json-data-file in path
        with open(path+'/training_data.json','w',encoding='utf-8') as f:
            json.dump(train_data_dict,f,ensure_ascii=False,indent=8)

        json_path=path+'/training_data.json'
        model_path=path
        model_score=trainingModel(json_path,model_path)
        ic(model_score)

    except ValueError as val:

        return Response("Value not found inside  json trainingData", val)

    except KeyError as keyval:

        return Response("Key value error incorrect key passed", keyval)

    except Exception as e:

        return Response((str(e)))

    return Response("Success")


if __name__ == "__main__":
    #clntApp = clientApp()
    app.run(port=8080,debug=True)