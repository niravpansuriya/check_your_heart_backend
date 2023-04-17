import time
from audio import *
from declaration import *
from run import *
from firebase import *

def cron():
    if(len(queue)):
        job = queue.pop()
        executeJob(job)
        
def startCron():
    while True:
        cron()
        time.sleep(5)

def executeJob(job):
    filename = job["filename"]
    userId = job["userId"]

    # get file
    getAudio(filename)

    res = classify("./audios/{}".format(filename))
    
    token = getCloudMessagingToken(userId)
    
    addRecord(userId, getEpoch(), res)
    
    if token:
        pushNotification(token, "", res)
    
    # remove file
    removeAudio(filename)