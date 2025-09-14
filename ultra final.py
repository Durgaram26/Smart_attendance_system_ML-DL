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
student_csv_path = r'D:\Carrer\my_project\face-recognition-based-attendance-system-master\student_final.csv'
attendance_csv_path = f'Attendance/Attendance-{datetoday}.csv'

if not os.path.exists(attendance_csv_path):
    if os.path.exists(student_csv_path):
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

    if os.path.exists(faces_dir):
        for user in os.listdir(faces_dir):
            user_dir = os.path.join(faces_dir, user)
            if os.path.isdir(user_dir):
                for img_name in os.listdir(user_dir):
                    img_path = os.path.join(user_dir, img_name)
                    try:
                        # Load and verify the image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Failed to load image: {img_path}")
                            os.remove(img_path)  # Remove invalid image
                            continue
                        # Convert to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img_rgb.dtype != np.uint8:
                            img_rgb = img_rgb.astype(np.uint8)
                        # Process with face_recognition
                        face_encodings = face_recognition.face_encodings(img_rgb)
                        if face_encodings:
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(user)
                        else:
                            print(f"No faces found in {img_path}, skipping...")
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        os.remove(img_path)  # Remove invalid image
                        continue

    # Save encodings to pickle file
    with open(encodings_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names
# Load encodings from pickle file
def load_encodings_from_pickle():
    if os.path.exists(encodings_path):
        try:
            with open(encodings_path, 'rb') as f:
                known_face_encodings, known_face_names = pickle.load(f)
        except Exception as e:
            print(f"Error loading encodings: {e}")
            known_face_encodings, known_face_names = encode_faces_to_pickle()
    else:
        known_face_encodings, known_face_names = encode_faces_to_pickle()

    return known_face_encodings, known_face_names

# Load or encode face encodings
known_face_encodings, known_face_names = load_encodings_from_pickle()

# Get the total number of registered users
def totalreg():
    if os.path.exists(faces_dir):
        return len([d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))])
    return 0

# Extract info from today's attendance file
def extract_attendance():
    if os.path.exists(attendance_csv_path):
        df = pd.read_csv(attendance_csv_path)
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        statuses = df['Status']
        l = len(df)
        return names, rolls, times, statuses, l
    return [], [], [], [], 0

# Add attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')
    now = datetime.now()
    current_time_str = now.strftime("%H:%M:%S")

    late_threshold = datetime.strptime("08:45:00", "%H:%M:%S").time()

    if os.path.exists(attendance_csv_path):
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
    if os.path.exists(faces_dir):
        userlist = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
        names = []
        rolls = []
        l = len(userlist)

        for i in userlist:
            name, roll = i.split('_')
            names.append(name)
            rolls.append(roll)

        return userlist, names, rolls, l
    return [], [], [], 0

# Delete a user folder
def deletefolder(duser):
    if os.path.exists(duser):
        pics = os.listdir(duser)
        for i in pics:
            os.remove(os.path.join(duser, i))
        os.rmdir(duser)

# Update Excel with attendance
def update_excel_with_attendance():
    if os.path.exists(attendance_csv_path) and os.path.exists(student_csv_path):
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
    return None

# Send email with attendance
def send_email(recipient, subject, body, attachment_path):
    try:
        # Outlook SMTP server configuration
        smtp_server = 'smtp-mail.outlook.com'
        smtp_port = 587

        # Sender and recipient details
        sender_email = "jananishri638@gmail.com"
        sender_password = "jeni$3103"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject

        # Email body
        msg.attach(MIMEText(body, 'plain'))

        # Attachment
        if attachment_path and os.path.exists(attachment_path):
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
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)
        traceback.print_exc()

################## ROUTING FUNCTIONS #########################

@app.route('/')
def home():
    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

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

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, statuses, l = extract_attendance()

    if not known_face_encodings:
        return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No faces encoded.')

    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Could not open camera.')
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detection_times = {}  # Dictionary to store detection start times for faces
    confidence_threshold = 0.6  # Adjusted threshold for better recognition

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to capture frame or frame is None")
                continue  # Try again, don't break
            print(f"frame dtype: {frame.dtype}, shape: {frame.shape}, type: {type(frame)}")
                
            # Convert BGR to RGB and ensure proper format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            print(f"rgb_frame dtype: {rgb_frame.dtype}, shape: {rgb_frame.shape}, type: {type(rgb_frame)}, contiguous: {rgb_frame.flags['C_CONTIGUOUS']}")
            if rgb_frame.dtype != np.uint8:
                print("rgb_frame is not uint8, skipping...")
                continue
            if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
                print(f"Invalid rgb_frame shape: {rgb_frame.shape}, skipping...")
                continue
            
            # Scale down frame for faster processing
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                name = "Unknown"  # Initialize name before using
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index] and face_distances[best_match_index] < confidence_threshold:
                        name = known_face_names[best_match_index]

                        # Record the current time for the detected face
                        if name not in detection_times:
                            detection_times[name] = datetime.now()
                        else:
                            detection_duration = (datetime.now() - detection_times[name]).total_seconds()
                            if detection_duration >= 3:  # Reduced to 3 seconds for faster response
                                add_attendance(name)
                                detection_times.pop(name)  # Remove the entry after attendance is marked
                    else:
                        # Reset the detection time if no match or confidence is low
                        if name in detection_times:
                            detection_times.pop(name)

                cv2.rectangle(frame, (left, top), (right, bottom), (86, 32, 251), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    
    except Exception as e:
        print(f"Error during face recognition: {e}")
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Send email after attendance is taken
    try:
        recipient = 'delshi@rocketmail.com'
        subject = f'Attendance for {datetoday2}'
        body = 'Please find attached the attendance Excel file.'
        attachment_path = update_excel_with_attendance()
        if attachment_path:
            send_email(recipient, subject, body, attachment_path)
    except Exception as e:
        print(f"Error sending email: {e}")

    names, rolls, times, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/adduser')
def adduser():
    return render_template('adduser.html', totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/adduser', methods=['POST'])
def adduser_post():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userfolder = os.path.join(faces_dir, f"{newusername}_{newuserid}")

    if not os.path.isdir(userfolder):
        os.makedirs(userfolder)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return render_template('adduser.html', totalreg=totalreg(), datetoday2=datetoday2, mess='Could not open camera.')
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    instructions = ["Look straight", "Look left", "Look right", "Look up", "Look down"]
    capture_interval = 2  # seconds
    i, j = 0, 0
    instruction_index = 0
    start_time = datetime.now()

    try:
        while j < nimgs:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to capture frame or frame is None")
                continue  # Try again, don't break
            print(f"frame dtype: {frame.dtype}, shape: {frame.shape}, type: {type(frame)}")

            # Ensure frame is uint8 and has 3 channels
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Invalid frame shape: {frame.shape}, skipping...")
                continue

            elapsed_time = (datetime.now() - start_time).total_seconds()

            # Display instructions on the frame
            if instruction_index < len(instructions):
                instruction_text = instructions[instruction_index]
                cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if elapsed_time > capture_interval:
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                print(f"rgb_frame dtype: {rgb_frame.dtype}, shape: {rgb_frame.shape}, type: {type(rgb_frame)}, contiguous: {rgb_frame.flags['C_CONTIGUOUS']}")
                if rgb_frame.dtype != np.uint8:
                    print("rgb_frame is not uint8, skipping...")
                    continue
                if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
                    print(f"Invalid rgb_frame shape: {rgb_frame.shape}, skipping...")
                    continue
                faces = face_recognition.face_locations(np.ascontiguousarray(rgb_frame))
                
                if faces:
                    i += 1
                    for face_location in faces:
                        top, right, bottom, left = face_location
                        face_img = frame[top:bottom, left:right]
                        if face_img.size > 0:  # Check if face image is valid
                            face_img = cv2.resize(face_img, (224, 224))
                            cv2.imwrite(os.path.join(userfolder, f"{newusername}_{i}.jpg"), face_img)
                            j += 1
                            if j >= nimgs:
                                break
                    instruction_index += 1  # Move to the next instruction
                    start_time = datetime.now()  # Reset the timer

            cv2.imshow('Adding new user', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    
    except Exception as e:
        print(f"Error during user addition: {e}")
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Re-encode faces after adding the new user
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = encode_faces_to_pickle()
    return render_template('adduser.html', totalreg=totalreg(), datetoday2=datetoday2, mess='User added successfully!')

if __name__ == "__main__":
    app.run(debug=True)