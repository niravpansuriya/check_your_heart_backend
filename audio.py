import requests
import urllib.parse
import os

def getAudio(fileName):
    url = "https://firebasestorage.googleapis.com/v0/b/checkyourheart-119cc.appspot.com/o"
    file_path = "Audio/{}".format(fileName)
    encoded_file_path = urllib.parse.quote(file_path, safe='')
    response = requests.get(url + "/" + encoded_file_path + "?alt=media")

    with open("./audios/{}".format(fileName), "wb") as f:
        f.write(response.content)

def removeAudio(fileName):
    # Specify the file path
    file_path = "./audios/{}".format(fileName)

    # Delete the file
    os.remove(file_path)
