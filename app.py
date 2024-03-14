import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from arduino_control import move_servo_to_90_and_back
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam .
face_detector = cv2.CascadeClassifier(
    'C:\\Users\\Sudo\\Downloads\\Face-Check-In-main\\Face-Check-3d\\static\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    fee_status = request.form['feestatus'] # This will be 'paid' or 'notpaid'

    # Processing the fee status
    fee_status_binary = 1 if fee_status.lower() == 'paid' else 0

    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{fee_status_binary}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def takeAttendance():
    # Check if the face recognition model is available
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)
    ret = True

    # Flag to check if attendance is recorded successfully
    attendance_recorded = False

    # Loop until 'q' is pressed or a face is detected
    while ret:
        ret, frame = cap.read()

        # Check if the frame is not empty
        if ret:
            # Extract faces from the frame
            faces = extract_faces(frame)

            # Check if faces are detected
            if faces is not None and len(faces) > 0:
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)

                    # Extract and resize the detected face
                    face_img = cv2.resize(frame[y:y + h, x:x + w], (50, 50))

                    # Identify the person in the detected face
                    identified_person = identify_face(face_img.reshape(1, -1))[0]

                    # Add attendance for the identified person
                    add_attendance(identified_person)

                    # Display the identified person's name
                    cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                                cv2.LINE_AA)

                    # Set attendance_recorded flag to True
                    attendance_recorded = True
                    # Move the motor to 90 degrees and then back
                    move_servo_to_90_and_back()

            # Display the frame
            cv2.imshow('Attendance', frame)

            # Break the loop if 'q' is pressed or attendance is recorded
            if cv2.waitKey(1) == ord('q') or attendance_recorded:
                break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # If attendance is successfully recorded, open the door
    if attendance_recorded:
    # Extract attendance information
        names, rolls, times, l = extract_attendance()

    # Render the home template with attendance information
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def addnewuser():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)