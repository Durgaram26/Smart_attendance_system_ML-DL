import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import traceback

# Defining Flask App
app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def send_email(recipient, subject, body, attachment_path):
    try:
        # Outlook SMTP server configuration
        smtp_server = 'smtp-mail.outlook.com'
        smtp_port = 587  # Outlook SMTP port

        # Sender and recipient details
        sender_email = "*********"
        sender_password = "*********"  # Use your actual email and password here

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject

        # Email body
        msg.attach(MIMEText(body, 'plain'))

        # Attachment
        attachment = open(attachment_path, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename=Attendance-{datetoday}.csv')
        msg.attach(part)

        # Connect to Outlook server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send email
        text = msg.as_string()
        server.sendmail(sender_email, recipient, text)
        server.quit()
    except Exception as e:
        print("Failed to send email:", e)
        traceback.print_exc()

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Initialize the attendance CSV with data from student.csv
student_csv_path = 'student.csv'
attendance_csv_path = f'Attendance/Attendance-{datetoday}.csv'

if not os.path.exists(attendance_csv_path):
    student_df = pd.read_csv(student_csv_path)
    student_df['Time'] = ""
    student_df['Status'] = "Absent"
    student_df.to_csv(attendance_csv_path, index=False)

# Get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train the model on all the faces available in faces folder
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

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(attendance_csv_path)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    statuses = df['Status']
    l = len(df)
    return names, rolls, times, statuses, l

# Add Attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_csv_path)
    mask = (df['Name'] == username) & (df['Roll'] == int(userid))
    if not df.loc[mask, 'Status'].eq("Present").any():
        df.loc[mask, 'Time'] = current_time
        df.loc[mask, 'Status'] = "Present"
        df.to_csv(attendance_csv_path, index=False)

# Get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

# Delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(os.path.join(duser, i))
    os.rmdir(duser)

def update_excel_with_attendance():
    attendance_df = pd.read_csv(attendance_csv_path)
    student_df = pd.read_csv(student_csv_path)

    student_df['Time'] = ""
    student_df['Status'] = "Absent"

    for index, row in attendance_df.iterrows():
        name = row['Name']
        roll = row['Roll']
        time = row['Time']
        status = row['Status']
        mask = (student_df['Name'] == name) & (student_df['Roll'] == int(roll))
        student_df.loc[mask, 'Time'] = time
        student_df.loc[mask, 'Status'] = status

    updated_excel_path = f'student-{datetoday}.csv'
    student_df.to_csv(updated_excel_path, index=False)

    return updated_excel_path

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder(os.path.join('static/faces', duser))

    ## if all the faces are deleted, delete the trained file...
    if not os.listdir('static/faces/'):
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except Exception as e:
        print("Error training model:", e)
        traceback.print_exc()

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, statuses, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # Send email after attendance is taken
    recipient = 'delshi@rocketmail.com'
    subject = f'Attendance for {datetoday2}'
    body = 'Please find attached the attendance Excel file.'
    attachment_path = update_excel_with_attendance()
    send_email(recipient, subject, body, attachment_path)

    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = os.path.join('static/faces', f'{newusername}_{newuserid}')
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
