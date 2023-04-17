import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import messaging
from utils import *

cred = credentials.Certificate('./credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://checkyourheart-119cc-default-rtdb.firebaseio.com'
})

def getCloudMessagingToken(userId):
    ref = db.reference('/messaging-tokens/{}'.format(userId))
    data = ref.get()
    return data

def pushNotification(token, title, body):
    # print("token",token)
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        token=token
    )

    response = messaging.send(message)
    return response


def addRecord(userId, timestamp, result):
    # ref = db.reference('/history/{}'.format(userId))
    ref = db.reference("/history")
    ref.child(userId).push().set(
        {
        'timestamp': timestamp,
        'result': result
    })