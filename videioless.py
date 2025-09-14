import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
import pandas as pd
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import traceback
import face_recognition
import pickle
import threading
from flask import send_file



# Define a global stop flag
stop_flag = threading.Event()

# Defining Flask App
app = Flask(__name__)

nimgs = 100

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize necessary directories
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

# Paths to store the pickle file and the face images directory
encodings_path = 'static/face_encodings.pkl'
faces_dir = 'static/faces'

# Encode faces and save to pickle file
def encode_faces_to_pickle():
    known_face_encodings = []
    known_face_names = []

    for user in os.listdir(faces_dir):
        user_dir = os.path.join(faces_dir, user)
        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(user)

    # Save encodings to pickle file
    with open(encodings_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

# Load encodings from pickle file
def load_encodings_from_pickle():
    if os.path.exists(encodings_path):
        with open(encodings_path, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        known_face_encodings, known_face_names = encode_faces_to_pickle()

    return known_face_encodings, known_face_names

# Load or encode face encodings
known_face_encodings, known_face_names = load_encodings_from_pickle()

def reset_attendance():
    df = pd.read_csv(attendance_csv_path)
    df['Time'] = ""
    df['Status'] = "Absent"
    df.to_csv(attendance_csv_path, index=False)

# Get the total number of registered users
def totalreg():
    return len(os.listdir(faces_dir))

# Extract info from today's attendance file
def extract_attendance():
    df = pd.read_csv(attendance_csv_path)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    statuses = df['Status']
    l = len(df)
    return names, rolls, times, statuses, l

# Add attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')
    now = datetime.now()
    current_time_str = now.strftime("%H:%M:%S")

    late_threshold = datetime.strptime("08:45:00", "%H:%M:%S").time()

    df = pd.read_csv(attendance_csv_path)
    mask = (df['Name'] == username) & (df['Roll'] == int(userid))

    if not df.loc[mask, 'Status'].isin(["Present", "Late"]).any():
        df.loc[mask, 'Time'] = current_time_str
        if now.time() > late_threshold:
            df.loc[mask, 'Status'] = "Late"
        else:
            df.loc[mask, 'Status'] = "Present"

        df.to_csv(attendance_csv_path, index=False)

# Get names and roll numbers of all users
def getallusers():
    userlist = os.listdir(faces_dir)
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

# Update Excel with attendance
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

# Send email with attendance
def send_email(recipient, subject, body, attachment_path):
    try:
        # Outlook SMTP server configuration
        smtp_server = 'smtp-mail.outlook.com'
        smtp_port = 587

        # Sender and recipient details
        sender_email = "password"
        sender_password = "password"

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

################## ROUTING FUNCTIONS #########################

@app.route('/')
def home():
    names, rolls, times, statuses, l = extract_attendance()

    # Calculate total counts
    present_count = (statuses == 'Present').sum()
    absent_count = (statuses == 'Absent').sum()

    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2, present_count=present_count, absent_count=absent_count)

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder(os.path.join(faces_dir, duser))

    # If all the faces are deleted, delete the trained file...
    if not os.listdir(faces_dir):
        if os.path.exists('static/face_encodings.pkl'):
            os.remove('static/face_encodings.pkl')

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/download_attendance')
def download_attendance():
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    return send_file(attendance_file, as_attachment=True)

@app.route('/start', methods=['GET'])
def start():
    global stop_flag
    stop_flag.clear()  # Clear the stop flag before starting

    names, rolls, times, statuses, l = extract_attendance()

    if not known_face_encodings:
        return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No faces encoded.')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    detection_times = {}  # Dictionary to store detection start times for faces
    confidence_threshold = 1.0  # Set a threshold for face recognition confidence

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            name = "Unknown"  # Initialize name before using
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < confidence_threshold:
                name = known_face_names[best_match_index]

                # Record the current time for the detected face
                if name not in detection_times:
                    detection_times[name] = datetime.now()
                else:
                    detection_duration = (datetime.now() - detection_times[name]).total_seconds()
                    if detection_duration >= 5:
                        add_attendance(name)
                        detection_times.pop(name)  # Remove the entry after attendance is marked
            else:
                # Reset the detection time if no match or confidence is low
                if name in detection_times:
                    detection_times.pop(name)

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

@app.route('/stop_attendance')
def stop_attendance():
    global stop_flag
    stop_flag.set()  # Set the stop flag to stop the attendance process
    return "Attendance process stopped"  # Optional: Return a message indicating attendance has stopped



@app.route('/stop', methods=['GET'])
def stop():
    global stop_flag
    stop_flag.set()  # Set the stop flag to stop the attendance process
    return "Attendance process stopped"



@app.route('/adduser')
def adduser():
    return render_template('adduser.html', totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/reset_attendance', methods=['GET'])
def reset_attendance_route():
    reset_attendance()
    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Attendance reset successfully.')

@app.route('/adduser', methods=['POST'])
def adduser_post():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userfolder = os.path.join(faces_dir, f"{newusername}_{newuserid}")

    if not os.path.isdir(userfolder):
        os.makedirs(userfolder)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    instructions = ["Look straight", "Look left", "Look right", "Look up", "Look down"]
    capture_interval = 1  # seconds
    i, j = 0, 0
    instruction_index = 0
    start_time = datetime.now()

    while j < nimgs:
        _, frame = cap.read()
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Display instructions on the frame
        if instruction_index < len(instructions):
            instruction_text = instructions[instruction_index]
            cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if elapsed_time > capture_interval:
            faces = face_recognition.face_locations(frame)
            if faces:
                i += 1
                for face_location in faces:
                    top, right, bottom, left = face_location
                    face_img = frame[top:bottom, left:right]
                    face_img = cv2.resize(face_img, (224, 224))
                    cv2.imwrite(os.path.join(userfolder, f"{newusername}_{i}.jpg"), face_img)
                    j += 1
                    if j >= nimgs:
                        break
                instruction_index += 1  # Move to the next instruction
                start_time = datetime.now()  # Reset the timer

        cv2.imshow('Adding new user', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

 
    # Re-encode faces after adding the new user
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = encode_faces_to_pickle()
    return render_template('adduser.html', totalreg=totalreg(), datetoday2=datetoday2, mess='User added successfully!')


    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=message)

if __name__ == "__main__":
    app.run(debug=True)

"""if __name__ == "__main__":
    app.run(host='192.168.248.100', port=5000, debug=True)"""
