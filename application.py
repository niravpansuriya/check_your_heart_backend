from flask import Flask, request, Response
import os
import json
from declaration import *
from cron_job import startCron
import threading

app = Flask(__name__)

@app.route("/", methods=["GET"])
def helloworld():
    return "helloworld"

@app.route("/uploadfile", methods=["PUT"])
def uploadfile():
    f = request.files["file"]
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    filePath = absolute_path + "/test.wav"
    f.save(filePath)
    return "success"


@app.route("/getresults", methods=["GET"])
def getresults():
    # results = r.classify()
    # print(results)
    # return results
    os.system("python3 run.py")
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    file1 = open(absolute_path + "/MyFile.txt", "r+")
    string = file1.readline(10)
    # print (string)
    return string


@app.route("/check", methods=["POST"])
def upload():
    dataJson = request.get_json()
    filename = dataJson["filename"]
    username = dataJson["username"]

    queue.append({"filename": filename, "userId": username})
    
    response_data = {"success": True, "message": "File uploaded successfully."}
    response_json = json.dumps(response_data)
    return Response(response_json, mimetype="application/json")

t = threading.Thread(target=startCron)
t.start()

app.run(host="0.0.0.0", debug=True, port=5001)

