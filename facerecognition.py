import cv2
import pyodbc
import numpy as np
import serial
import datetime
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# Mapping dictionary for weekdays
weekday_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Specify the absolute path to the XML file
cascade_path = "C:/Users/USER/PycharmProjects/pythonProjectALLADIN/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer.create()

# Serial communication settings
ser = serial.Serial('COM4', 115200)  # Replace 'COM4' with the appropriate port and baud rate


class ConnectionDialog(tk.simpledialog.Dialog):
    def __init__(self, parent):
        self.server_entry = None
        self.database_entry = None
        self.username_entry = None
        self.password_entry = None
        super().__init__(parent)

    def body(self, master):
        tk.Label(master, text="Server Name:").grid(row=0, sticky=tk.W)
        tk.Label(master, text="Database Name:").grid(row=1, sticky=tk.W)
        tk.Label(master, text="Username:").grid(row=2, sticky=tk.W)
        tk.Label(master, text="Password:").grid(row=3, sticky=tk.W)

        self.server_entry = tk.Entry(master)
        self.database_entry = tk.Entry(master)
        self.username_entry = tk.Entry(master)
        self.password_entry = tk.Entry(master, show="*")

        self.server_entry.grid(row=0, column=1)
        self.database_entry.grid(row=1, column=1)
        self.username_entry.grid(row=2, column=1)
        self.password_entry.grid(row=3, column=1)

        return self.server_entry  # Focus on the server entry field

    def apply(self):
        self.result = (
            self.server_entry.get(),
            self.database_entry.get(),
            self.username_entry.get(),
            self.password_entry.get()
        )


# Attempt to connect to the SQL Server
connected = False
root = tk.Tk()  # Create a single instance of Tk

while not connected:
    # Get SQL Server connection details from the user using custom dialog
    connection_dialog = ConnectionDialog(root)
    connection_details = connection_dialog.result

    # Build the connection string
    connection_string = (
        f'DRIVER={{SQL Server}};'
        f'SERVER={connection_details[0]};'
        f'DATABASE={connection_details[1]};'
        f'UID={connection_details[2]};'
        f'PWD={connection_details[3]};'
    )

    # Attempt to connect to the SQL Server
    try:
        conn = pyodbc.connect(connection_string)
        print("Connected to the database.")
        connected = True
    except pyodbc.Error as e:
        print("Error connecting to the database:", e)
        retry = messagebox.askretrycancel("Connection Error", "Failed to connect to the SQL Server. Retry?")
        if not retry:
            root.destroy()  # Destroy the Tk instance before exiting
            exit()  # Exit the script if the user chooses not to retry

# Check if the AttendanceRecord table exists
check_table_query = """
SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'AttendanceRecord';
"""

try:
    cursor = conn.cursor()
    cursor.execute(check_table_query)
    table_exists = cursor.fetchone()[0]
except pyodbc.Error as e:
    print("Error checking if table exists:", e)
    table_exists = False

if not table_exists:
    # Execute SQL query to create the AttendanceRecord table
    create_table_query = """
    CREATE TABLE AttendanceRecord (
        AttendanceRecordID INT PRIMARY KEY IDENTITY(1,1),
        PersonID NVARCHAR(50),  
        Name VARCHAR(100),
        Subject NVARCHAR(100), 
        Type NVARCHAR(50),
        RoomEntered NVARCHAR(50),
        DateTimeEntered DATETIME,
        DateTimeExited DATETIME, 
        DetectionStarted_Enter TIME,
        RecognitionEnded_Enter TIME,
        RecognitionTime_Enter DECIMAL(10, 2),
        DetectionStarted_Exit TIME,
        RecognitionEnded_Exit TIME,
        RecognitionTime_Exit DECIMAL(10, 2)
    );
    """

    try:
        # Execute the query to create the table
        cursor.execute(create_table_query)
        conn.commit()
        print("AttendanceRecord table created.")
    except pyodbc.Error as e:
        print("Error creating AttendanceRecord table:", e)
else:
    print("AttendanceRecord table already exists.")


def get_face_data_from_db(current_weekday):
    cursor = conn.cursor()

    # Fetch data from CapturedFaces, PersonTable, and SubjectTable using JOINs
    query = """
    SELECT CF.CapturedFacesID, PT.Name, ST.Subject, PT.Type, CF.ImageData, CF.Weekdays
    FROM CapturedFaces CF
    INNER JOIN PersonTable PT ON CF.PersonID = PT.PersonID_ID
    INNER JOIN SubjectTable ST ON CF.SubjectID = ST.SubjectID
    """
    cursor.execute(query)

    rows = cursor.fetchall()
    cursor.close()

    # Filter the results based on the current weekday
    filtered_rows = [
        (label, labelname, subject, persontype, image_data, weekday)
        for label, labelname, subject, persontype, image_data, weekday in rows
        if weekday_mapping.get(weekday, -1) == current_weekday
    ]

    return filtered_rows


# Define a function to check if a rectangle is centered
def is_centered(rect, frame_width, frame_height, tolerance=0.7):
    x, y, w, h = rect
    # Calculate the center of the detected face
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    # Calculate the tolerance limits
    x_tolerance = frame_width * tolerance
    y_tolerance = frame_height * tolerance

    # Check if the face center is within the central region of the frame
    if frame_width / 2 - x_tolerance <= face_center_x <= frame_width / 2 + x_tolerance \
            and frame_height / 2 - y_tolerance <= face_center_y <= frame_height / 2 + y_tolerance:
        return True
    else:
        return False


def warning():
    while True:
        ser.write(b'0')
        break


def convert_varbinary_to_image(varbinary_data):
    nparr = np.frombuffer(varbinary_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


# Checking for Attendance Record Table
def check_existing_record(person_id, current_date):
    cursor = conn.cursor()
    query = "SELECT COUNT(*) FROM AttendanceRecord WHERE PersonID = ? AND CONVERT(DATE, DateTimeEntered) = ?"
    cursor.execute(query, (person_id, current_date))
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0


# Get the current weekday
current_datetime = datetime.datetime.now()
current_weekday = current_datetime.weekday()

# Get training images and labels for the current weekday
face_data = get_face_data_from_db(current_weekday)

images = []
labels = []
labelsname = []
subjects = []
types = []
weekdays = []

for label, labelname, subject, persontype, image_data, weekday in face_data:
    face_image = convert_varbinary_to_image(image_data)
    images.append(np.array(face_image))
    labels.append(label)
    labelsname.append(labelname)
    subjects.append(subject)
    types.append(persontype)
    weekdays.append(weekday)

# Train the recognizer
recognizer.train(images, np.array(labels))

# Initialize two VideoCapture objects for entering and exiting cameras
cap_enter = cv2.VideoCapture(0)  # Camera for entering the room
cap_exit = cv2.VideoCapture(1)  # Camera for going outside the room (adjust the index as needed)

root.destroy()  # Destroy the Tk instance when done with the GUI

enter_recognition_start_time = None
recognition_time = None
exit_recognition_start_time = None
recognition_time1 = None
detection_started_enter = None
recognition_ended_enter = None
detection_started_exit = None
recognition_ended_exit = None

# Initialize variables to keep track of failed attempts and the time of the last attempt
failed_attempts = 0
failed_attempts1 = 0
last_attempt_time = time.time()
last_attempt_time1 = time.time()

while True:

    # Capture frames from entering and exit cameras
    ret_enter, frame_enter = cap_enter.read()
    ret_exit, frame_exit = cap_exit.read()

    # Convert frames to grayscale for face detection
    gray_enter = cv2.cvtColor(frame_enter, cv2.COLOR_BGR2GRAY)
    gray_exit = cv2.cvtColor(frame_exit, cv2.COLOR_BGR2GRAY)

    # Detect faces for entering camera
    faces_enter = face_cascade.detectMultiScale(frame_enter, scaleFactor=1.1, minNeighbors=22, minSize=(30, 30))

    # Loop through the detected faces and draw rectangles around them
    for (x, y, w, h) in faces_enter:
        cv2.rectangle(frame_enter, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only process centered faces
    centered_faces_enter = [face for face in faces_enter if is_centered(face,
                                                                        frame_enter.shape[1], frame_enter.shape[0])]
    recognized_label = ""
    access_status = ""
    access_Status2 = ""
    # Add a flag to keep track of whether the signal has been sent or not
    signal_sent = False
    signal_sent1 = False
    allow_to_enter = False
    allow_to_enter2 = False

    if len(centered_faces_enter) >= 1:
        x, y, w, h = centered_faces_enter[0]

        cv2.rectangle(frame_enter, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_image = cv2.cvtColor(
            frame_enter[max(0, y - 1):min(frame_enter.shape[0], y + h + 1), max(0, x - 1):min(frame_enter.shape[1],
                                                                                              x + w + 1)],
            cv2.COLOR_BGR2GRAY)
        # Apply contrast adjustment to the neighborhood window
        alpha = 1.5  # Adjust this value to control the contrast
        beta = 0  # Adjust this value to control the brightness
        contrast_adjusted_face = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)

        # Apply bilateral filter
        diameter = 9
        sigma_color = 75
        sigma_space = 75
        filtered_face = cv2.bilateralFilter(contrast_adjusted_face, diameter, sigma_color, sigma_space)

        # Apply histogram equalization with 3x3 neighborhood window
        equalized_face = cv2.equalizeHist(filtered_face)
        # Apply histogram equalization with 3x3 neighborhood window
        equalized_face2 = cv2.equalizeHist(filtered_face)

        # Blend the original face image with the equalized face image
        blend_alpha = 0.5  # Adjust this value to control the blending
        blended_face = cv2.addWeighted(equalized_face, blend_alpha, equalized_face2, 1 - blend_alpha, 0)

        # Resize the blended face image to 181x181 pixels
        resized_face = cv2.resize(blended_face, (181, 181))

        if enter_recognition_start_time is None:
            enter_recognition_start_time = time.time()
            detection_started_enter = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        # Recognize the face for entering camera
        label, confidence = recognizer.predict(resized_face)
        if confidence <= 46:
            recognition_time = time.time() - enter_recognition_start_time
            enter_recognition_start_time = None
            person_id = label
            Name = labelsname[labels.index(label)]
            RoomEntered = "ITLAB3"
            sub = subjects[labels.index(label)]
            pertype = types[labels.index(label)]
            cursor = conn.cursor()
            cursor.execute("SELECT PT.PersonID, PT.Name, ST.Subject, PT.Type, TimeIn, TimeOut, Weekdays "
                           "FROM CapturedFaces CF INNER JOIN PersonTable PT ON CF.PersonID = PT.PersonID_ID "
                           "INNER JOIN SubjectTable ST ON CF.SubjectID = ST.SubjectID"
                           " WHERE CapturedFacesID = ?", (person_id,))
            person_row = cursor.fetchone()
            cursor.close()
            failed_attempts = 0  # Reset failed attempts upon successful recognition

            if person_row:
                person_id, Name, sub, pertype, time_in_str, time_out_str, weekday = person_row
                recognized_label = f"Person ID: {person_id}"

                # Convert time_in_str and time_out_str to datetime.time objects
                time_in = datetime.datetime.strptime(time_in_str, "%H:%M").time()
                time_out = datetime.datetime.strptime(time_out_str, "%H:%M").time()

                # Determine if the person can enter based on time_in, time_out, and weekdays
                current_datetime = datetime.datetime.now()
                current_time = current_datetime.time()

                if time_in <= current_time:
                    if not signal_sent:
                        # Send the signal
                        ser.write(b'1')
                        signal_sent = True  # Set the flag to indicate the signal has been sent
                    access_status = "You may Enter"
                    access_status2 = ""
                    allow_to_enter = True
                    # Set the RecognitionEnded timestamp for entry
                    recognition_ended_enter = datetime.datetime.now().strftime("%H:%M:%S.%f")[
                                              :-3]  # Include milliseconds

                else:
                    access_status = "Check Your Schedule"
                    access_Status2 = "You're Not Allowed to Enter"
                    allow_to_enter = False

            if allow_to_enter:
                # Check if a record already exists for the current user and date
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                if check_existing_record(str(person_id), current_date):
                    # Update the existing record instead of inserting a new one
                    update_query = "UPDATE AttendanceRecord SET DateTimeEntered = ?, " \
                                   "DetectionStarted_Enter = ?, RecognitionEnded_Enter = ?, " \
                                   "RecognitionTime_Enter = ? " \
                                   "WHERE PersonID = ? AND CONVERT(DATE, DateTimeEntered) = ?"
                    entered_datetime = datetime.datetime.now()
                    formatted_datetime = entered_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime
                    cursor = conn.cursor()
                    cursor.execute(update_query, (formatted_datetime, detection_started_enter,
                                                  recognition_ended_enter, recognition_time,
                                                  str(person_id), current_date))
                    conn.commit()

                else:
                    # Insert record into AttendanceRecord table
                    insert_query = "INSERT INTO AttendanceRecord (PersonID, Name, Subject, " \
                                   "Type, RoomEntered, DateTimeEntered, DetectionStarted_Enter, " \
                                   "RecognitionEnded_Enter,  RecognitionTime_Enter) " \
                                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    entered_datetime = datetime.datetime.now()
                    formatted_datetime = entered_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime
                    cursor = conn.cursor()
                    cursor.execute(insert_query, (str(person_id), str(Name), sub,
                                                  pertype, RoomEntered, formatted_datetime, detection_started_enter,
                                                  recognition_ended_enter,
                                                  recognition_time))
                    # Convert person_id to str
                    conn.commit()

                if signal_sent:
                    time.sleep(5)  # Sleep for 5 seconds before re-checking the time condition

        else:

            # If it's the start of a new attempt, update the last_attempt_time
            if time.time() - last_attempt_time >= 5:
                failed_attempts += 1  # Increment failed attempts
                last_attempt_time = time.time()  # Update the time of the last attempt

                # If three consecutive failed attempts, update AttendanceRecord
                if failed_attempts == 3:
                    ser.write(b'3')
                    person_id = "UnknownID"
                    Name = "UnkownUser"
                    RoomEntered = "ITLAB3"
                    sub = "RecognitionFailed"
                    pertype = "RecognitionFailed"

                    # Insert record into AttendanceRecord table
                    insert_query = "INSERT INTO AttendanceRecord (PersonID, Name, Subject, " \
                                   "Type, RoomEntered, DetectionStarted_Enter) " \
                                   "VALUES (?, ?, ?, ?, ?, ?)"

                    cursor = conn.cursor()
                    cursor.execute(insert_query, (person_id, Name, sub,
                                                  pertype, RoomEntered, detection_started_enter))
                    conn.commit()

                    # Reset failed attempts counter
                    failed_attempts = 0
                    last_attempt_time = time.time()  # Update the time of the last attempt

            recognized_label = "Unknown User"
            access_status = "SCANNNING "
            allow_to_enter = False

            # Display the number of attempts on the screen
            cv2.putText(frame_enter, f"Attempts: {failed_attempts}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame_enter, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate text size for recognized label and access status
        label_text_size = cv2.getTextSize(recognized_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        status_text_size = cv2.getTextSize(access_status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        status_text_size2 = cv2.getTextSize(access_status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]

        # Calculate the position to display recognized label and access status
        label_text_x = x + int((w - label_text_size[0]) / 2)
        label_text_y = y - 10  # Place just above the rectangle
        status_text_x = x + int((w - status_text_size[0]) / 2)
        status_text_y = y + h + status_text_size[1] + 10  # Place below the rectangle
        status_text_x1 = x + int((w - status_text_size[0]) / 1)
        status_text_y1 = y + h + status_text_size[1] + 35  # Place below the rectangle

        # Display recognized label above the rectangle
        cv2.putText(frame_enter, recognized_label, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)
        # Display access status below the rectangle
        cv2.putText(frame_enter, access_status, (status_text_x, status_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_enter, access_Status2, (status_text_x1, status_text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        print("confidence value:", confidence)
    else:
        enter_recognition_start_time = None
        # Reset failed attempts counter
        failed_attempts = 0
        last_attempt_time = time.time()  # Update the time of the last attempt

    # Ensure to modify your code to handle the entering camera's logic

    # Detect faces for exit camera
    faces_exit = face_cascade.detectMultiScale(gray_exit, scaleFactor=1.1, minNeighbors=22, minSize=(30, 30))
    # Loop through the detected faces and draw rectangles around them
    for (x, y, w, h) in faces_exit:
        cv2.rectangle(frame_exit, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Only process centered faces
    centered_faces_exit = [face for face in faces_exit if is_centered(face, frame_exit.shape[1], frame_exit.shape[0])]

    if len(centered_faces_exit) >= 1:
        x, y, w, h = centered_faces_exit[0]
        cv2.rectangle(frame_exit, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_image1 = cv2.cvtColor(
            frame_exit[max(0, y - 1):min(frame_exit.shape[0], y + h + 1), max(0, x - 1):min(frame_exit.shape[1],
                                                                                            x + w + 1)],
            cv2.COLOR_BGR2GRAY)

        # Apply contrast adjustment to the brightened face
        alpha = 1.5  # Adjust this value to control the contrast
        beta = 0  # No additional brightness after the initial adjustment
        contrast_adjusted_face1 = cv2.convertScaleAbs(face_image1, alpha=alpha, beta=beta)

        # Apply bilateral filter
        diameter = 9
        sigma_color = 75
        sigma_space = 75
        filtered_face1 = cv2.bilateralFilter(contrast_adjusted_face1, diameter, sigma_color, sigma_space)

        # Apply histogram equalization with 3x3 neighborhood window
        equalized_face1 = cv2.equalizeHist(filtered_face1)
        # Apply histogram equalization with 3x3 neighborhood window
        equalized_face3 = cv2.equalizeHist(filtered_face1)

        # Blend the original face image with the equalized face image
        blend_alpha = 0.5  # Adjust this value to control the blending
        blended_face1 = cv2.addWeighted(equalized_face1, blend_alpha, equalized_face3, 1 - blend_alpha, 0)

        # Resize the blended face image to 181x181 pixels
        resized_face1 = cv2.resize(blended_face1, (181, 181))

        if exit_recognition_start_time is None:
            exit_recognition_start_time = time.time()
            detection_started_exit = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        # Recognize the face for entering camera
        label, confidence1 = recognizer.predict(resized_face1)
        if confidence1 <= 40:
            recognition_time1 = time.time() - exit_recognition_start_time
            exit_recognition_start_time = None
            person_id = label
            Name = labelsname[labels.index(label)]
            RoomEntered = "ITLAB3"
            sub = subjects[labels.index(label)]
            pertype = types[labels.index(label)]
            cursor = conn.cursor()
            cursor.execute("SELECT PT.PersonID, PT.Name, ST.Subject, PT.Type, TimeIn, TimeOut, Weekdays "
                           "FROM CapturedFaces CF INNER JOIN PersonTable PT ON CF.PersonID = PT.PersonID_ID "
                           "INNER JOIN SubjectTable ST ON CF.SubjectID = ST.SubjectID"
                           " WHERE CapturedFacesID = ?", (person_id,))
            person_row = cursor.fetchone()
            cursor.close()

            if person_row:
                person_id, Name, sub, pertype, time_in_str, time_out_str, weekday = person_row
                recognized_label = f"Person ID: {person_id}"

                # Convert time_in_str and time_out_str to datetime.time objects
                time_in = datetime.datetime.strptime(time_in_str, "%H:%M").time()
                time_out = datetime.datetime.strptime(time_out_str, "%H:%M").time()

                # Determine if the person can enter based on time_in, time_out, and weekdays
                current_datetime = datetime.datetime.now()
                current_time = current_datetime.time()

                if time_in <= current_time <= time_out:
                    if not signal_sent1:
                        # Send the signal
                        ser.write(b'1')
                        signal_sent1 = True  # Set the flag to indicate the signal has been sent
                    access_status = "You may Enter"
                    access_Status2 = ""
                    allow_to_enter2 = True
                    recognition_ended_exit = datetime.datetime.now().strftime("%H:%M:%S.%f")[
                                              :-3]  # Include milliseconds

                else:
                    access_status = "Check Your Schedule"
                    access_Status2 = "You're Not Allowed to Exit"
                    allow_to_enter = False

            if allow_to_enter2:
                if allow_to_enter2:
                    # Check if a record already exists for the current user and date
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    if check_existing_record(str(person_id), current_date):
                        # Update the existing record with exit data
                        update_query = "UPDATE AttendanceRecord SET DateTimeExited = ?, " \
                                       "DetectionStarted_Exit = ?, RecognitionEnded_Exit = ?, " \
                                       "RecognitionTime_Exit = ? " \
                                       "WHERE PersonID = ? AND CONVERT(DATE, DateTimeEntered) = ?"
                        exited_datetime = datetime.datetime.now()
                        formatted_datetime1 = exited_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime
                        cursor = conn.cursor()
                        cursor.execute(update_query, (formatted_datetime1, detection_started_exit,
                                                      recognition_ended_exit, recognition_time1,
                                                      str(person_id), current_date))
                        conn.commit()
                    else:
                        # Insert record into AttendanceRecord table
                        insert_query = "INSERT INTO AttendanceRecord (PersonID, Name, Subject, " \
                                       "Type, RoomEntered, DateTimeEntered, DetectionStarted_Exit, " \
                                       "RecognitionEnded_Exit,  RecognitionTime_Exit) " \
                                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        exited_datetime = datetime.datetime.now()
                        formatted_datetime1 = exited_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime
                        cursor = conn.cursor()
                        cursor.execute(insert_query, (str(person_id), str(Name), sub,
                                                      pertype, RoomEntered, formatted_datetime1, detection_started_exit,
                                                      recognition_ended_exit, recognition_time1))
                        # Convert person_id to str
                        conn.commit()

                if signal_sent1:
                    time.sleep(5)  # Sleep for 5 seconds before re-checking the time condition

        else:
            # If it's the start of a new attempt, update the last_attempt_time
            if time.time() - last_attempt_time1 >= 4:
                failed_attempts1 += 1  # Increment failed attempts
                last_attempt_time1 = time.time()  # Update the time of the last attempt

                # If three consecutive failed attempts, update AttendanceRecord
                if failed_attempts1 == 3:
                    ser.write(b'3')
                    person_id = "UnknownID"
                    Name = "UnknownUser"
                    RoomEntered = "ITLAB3"
                    sub = "RecognitionFailed"
                    pertype = "RecognitionFailed"

                    # Insert record into AttendanceRecord table
                    insert_query = "INSERT INTO AttendanceRecord (PersonID, Name, Subject, " \
                                   "Type, RoomEntered, DetectionStarted_Exit) " \
                                   "VALUES (?, ?, ?, ?, ?, ?)"

                    cursor = conn.cursor()
                    cursor.execute(insert_query, (person_id, Name, sub,
                                                  pertype, RoomEntered, detection_started_exit))
                    conn.commit()

                    # Reset failed attempts counter
                    failed_attempts1 = 0
                    last_attempt_time1 = time.time()  # Update the time of the last attempt

            recognized_label = "Unknown User"
            access_status = "SCANNNING "
            allow_to_enter = False

            # Display the number of attempts on the screen
            cv2.putText(frame_exit, f"Attempts: {failed_attempts1}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame_exit, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate text size for recognized label and access status
        label_text_size = cv2.getTextSize(recognized_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        status_text_size = cv2.getTextSize(access_status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        status_text_size2 = cv2.getTextSize(access_status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]

        # Calculate the position to display recognized label and access status
        label_text_x = x + int((w - label_text_size[0]) / 2)
        label_text_y = y - 10  # Place just above the rectangle
        status_text_x = x + int((w - status_text_size[0]) / 2)
        status_text_y = y + h + status_text_size[1] + 10  # Place below the rectangle
        status_text_x1 = x + int((w - status_text_size[0]) / 1)
        status_text_y1 = y + h + status_text_size[1] + 35  # Place below the rectangle

        # Display recognized label above the rectangle
        cv2.putText(frame_exit, recognized_label, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)
        # Display access status below the rectangle
        cv2.putText(frame_exit, access_status, (status_text_x, status_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_exit, access_Status2, (status_text_x1, status_text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        print("confidence1 value:", confidence1)
    else:
        exit_recognition_start_time = None
        # Reset failed attempts counter
        failed_attempts1 = 0
        last_attempt_time1 = time.time()  # Update the time of the last attempt
    # Ensure to modify your code to handle the exit camera's logic
    # Display frames for both cameras
    cv2.imshow('Entering Camera', frame_enter)
    cv2.imshow('Exit Camera', frame_exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release both camera captures and close OpenCV windows
cap_enter.release()
cap_exit.release()
cv2.destroyAllWindows()
