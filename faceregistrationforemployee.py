import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import pyodbc

# Specify the absolute path to the XML file
cascade_path = "C:/Users/USER/PycharmProjects/pythonProjectALLADIN/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

shots_counter = 0
cap = cv2.VideoCapture(0)

# Global variables to store time in and time out values
time_in_value = None
time_out_value = None
server_name = 'DESKTOP-0Q9NNNR'
database_name = 'FaceRecognition'
username = 'SuperAdmin1'
password = 'admin123'

connection_string = f'Driver={{SQL Server}};' \
                    f'Server={server_name};' \
                    f'Database={database_name};' \
                    f'UID={username};' \
                    f'PWD={password};'

# Create a connection to the SQL Server
try:
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()
    print("Connected to SQL Server successfully.")
except pyodbc.Error as e:
    print(f"Error connecting to SQL Server: {str(e)}")
    # Handle the error or exit the script


# Function to create the PersonTable
def create_person_table():
    if not table_exists('PersonTable'):
        try:
            cursor.execute('''
                CREATE TABLE PersonTable (
                    PersonID_ID INT IDENTITY(1,1) PRIMARY KEY,
                    PersonID NVARCHAR(50),
                    Name  VARCHAR(MAX),
                    Course NVARCHAR(MAX),
                    Section NVARCHAR(MAX), 
                    Type NVARCHAR(50)
                )
            ''')
            connection.commit()
            print("Table 'PersonTable' created successfully.")
        except pyodbc.Error as e:
            print(f"Error creating 'PersonTable': {str(e)}")
    else:
        print("Table 'PersonTable' already exists.")


# Function to create the CapturedFaces table
def create_captured_faces_table():
    if not table_exists('CapturedFaces'):
        try:
            cursor.execute('''
                CREATE TABLE CapturedFaces (
                    CapturedFacesID INT IDENTITY(1,1) PRIMARY KEY,
                    TimeIn VARCHAR(MAX),
                    TimeOut VARCHAR(MAX),
                    Weekdays NVARCHAR(MAX),
                    ImageData VARBINARY(MAX),
                    PersonID INT,
                    SubjectID INT,
                    FOREIGN KEY (PersonID) REFERENCES PersonTable(PersonID_ID),
                    FOREIGN KEY (SubjectID) REFERENCES SubjectTable(SubjectID)
                )
            ''')
            connection.commit()
            print("Table 'CapturedFaces' created successfully.")
        except pyodbc.Error as e:
            print(f"Error creating 'CapturedFaces': {str(e)}")
    else:
        print("Table 'CapturedFaces' already exists.")


# Function to create the SubjectTable
def create_subject_table():
    if not table_exists('SubjectTable'):
        try:
            cursor.execute('''  
                CREATE TABLE SubjectTable (
                    SubjectID INT IDENTITY(1,1) PRIMARY KEY,
                    Subject NVARCHAR(MAX),
                    SchoolYear NVARCHAR(MAX)
                )
            ''')
            connection.commit()
            print("Table 'SubjectTable' created successfully.")
        except pyodbc.Error as e:
            print(f"Error creating 'SubjectTable': {str(e)}")
    else:
        print("Table 'SubjectTable' already exists.")


def change_button_color(event):
    submit_button.config(bg="green")  # Change the background color to green when clicked
    root.after(1000, reset_button_color)  # Schedule a function call to reset the button color after 1000 milliseconds


def reset_button_color():
    submit_button.config(bg="SystemButtonFace")  # Change the background color back to its original value


def validate_weekdays(weekdays):
    valid_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in weekdays:
        if day.strip() not in valid_weekdays:
            return False
    return True


def get_time_values():
    global time_in_value, time_out_value

    time_in_str = time_in_entry.get()
    time_out_str = time_out_entry.get()
    subject_str = subject_entry.get()
    weekday_str = weekday_entry.get()

    # Split the input strings by commas
    time_in_values = [t.strip() for t in time_in_str.split(',')]
    time_out_values = [t.strip() for t in time_out_str.split(',')]
    subjects = [s.strip() for s in subject_str.split(',')]
    weekdays = [w.strip() for w in weekday_str.split(',')]

    if not validate_weekdays(weekdays):
        messagebox.showerror("Invalid Weekday", "Weekdays should be spelled "
                                                "correctly (e.g., Monday, Tuesday, Wednesday).")
        return

    if not all(time.isdigit() for time in time_in_values) or not all(time.isdigit() for time in time_out_values):
        messagebox.showerror("Invalid Time Value", "Time In and Time Out values should be numeric.")
        return

    try:
        time_in_value = [datetime.strptime(time, "%H%M").time() for time in time_in_values]
        time_out_value = [datetime.strptime(time, "%H%M").time() for time in time_out_values]
    except ValueError:
        messagebox.showerror("Invalid Time Format", "Time In and Time Out should be in the format HHMM (e.g., 0900).")
        return

    # Check if weekday contains only alphabetical characters, commas, or dashes
    if not all(char.isalpha() or char in (',', '-') for char in ''.join(weekdays)):
        messagebox.showerror("Invalid Weekday", "Weekday should only contain alphabetical characters, "
                                                "commas, or dashes.")
        return

    subjects = [subject.strip() for subject in subjects]
    if not all(subjects):
        messagebox.showerror("No Subject Entered", "Please enter at least one subject.")
        return
    weekdays = [weekday.strip() for weekday in weekdays]
    if not all(weekdays):
        messagebox.showerror("No Weekday Entered", "Please enter at least one Weekday.")
        return

    if len(time_in_value) != len(time_out_value) or len(time_in_value) \
            != len(weekdays) or len(time_in_value) != len(subjects):
        messagebox.showerror("Input Mismatch", "The number of Time In, Time Out, Weekdays, and Subjects must match.")
        return

    Name = Name_entry.get()
    if not Name:
        messagebox.showerror("No Name has been Entered", "Please enter your Name.")
        return
    person_id = person_id_entry.get()
    if not person_id:
        messagebox.showerror("No PersonID has been Entered", "Please Enter a PersonID")
        return
    # Confirmation dialog
    confirmation_message = f"Notice! \nAll Inputs should correspond to each other \n" \
                           f" \nPersonID: {person_id_entry.get()}\n" \
                           f"Name: {Name_entry.get()}\nSubjects: {', '.join(subjects)}\n" \
                           f"Time In: {', '.join(time.strftime('%H:%M') for time in time_in_value)}\n" \
                           f"Time Out: {', '.join(time.strftime('%H:%M') for time in time_out_value)}\n" \
                           f"Weekdays: {', '.join(weekdays)}"
    response = messagebox.askyesno("Confirm Input Values", confirmation_message)
    if response:
        global shots_counter
        capture_face()
        root.quit()
    else:
        return


def table_exists(table_name):
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'")
    return cursor.fetchone()[0] > 0


def insert_face_data(person_type, person_id, Name, course, section, subject, schoolyear, time_in, time_out, weekday, image_data):
    try:
        # Check if the person already exists in PersonTable
        cursor.execute('SELECT COUNT(*) FROM PersonTable WHERE PersonID = ?', person_id)
        person_exists = cursor.fetchone()[0] > 0

        # If the person does not exist, insert data into PersonTable
        if not person_exists:
            cursor.execute('''
                INSERT INTO PersonTable (PersonID, Name, Course, Section, Type)
                VALUES (?, ?, ?, ?, ?)
            ''', person_id, Name, course, section, person_type)
            connection.commit()

        # Check if the subject already exists in SubjectTable
        cursor.execute('SELECT COUNT(*) FROM SubjectTable WHERE Subject = ? AND SchoolYear = ?', subject, schoolyear)
        subject_exists = cursor.fetchone()[0] > 0

        # If the subject does not exist, insert data into SubjectTable
        if not subject_exists:
            cursor.execute('''
                INSERT INTO SubjectTable (Subject, SchoolYear)
                VALUES (?, ?)
            ''', subject, schoolyear)
            connection.commit()

        # Retrieve the PersonID and SubjectID for the foreign keys in CapturedFaces
        cursor.execute('SELECT PersonID_ID FROM PersonTable WHERE PersonID = ?', person_id)
        person_id_fk = cursor.fetchone()[0]

        cursor.execute('SELECT SubjectID FROM SubjectTable WHERE Subject = ? AND SchoolYear = ?', subject, schoolyear)
        subject_id_fk = cursor.fetchone()[0]

        # Insert data into CapturedFaces
        cursor.execute('''
                    INSERT INTO CapturedFaces (TimeIn, TimeOut, Weekdays, ImageData, PersonID, SubjectID)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', time_in,  time_out, weekday, image_data, person_id_fk,
                       subject_id_fk)
        connection.commit()

    except pyodbc.Error as e:
        # Display a message box if the insertion fails
        messagebox.showerror("SQL Server Insertion Error", f"Error inserting data into CapturedFaces table:\n{str(e)}")


def is_face_centered(rect, frame_width, frame_height, tolerance=0.7):
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


def capture_face():
    global shots_counter
    if shots_counter >= 100:
        print("Maximum number of shots reached.")
        return
    # Hide the displayed image
    hide_image()

    time_in_str = time_in_entry.get()
    time_out_str = time_out_entry.get()
    subject_str = subject_entry.get()
    weekday_str = weekday_entry.get()

    # Split the input strings by commas
    time_in_values = [t.strip() for t in time_in_str.split(',')]
    time_out_values = [t.strip() for t in time_out_str.split(',')]
    subjects = [s.strip() for s in subject_str.split(',')]
    weekdays = [w.strip() for w in weekday_str.split(',')]

    person_type = "Employee"
    section = ""
    schoolyear = ""
    person_id = person_id_entry.get()
    if not person_id:
        print("PersonID is required.")
        return

    Name = Name_entry.get()
    if not Name:
        print("Name is required.")
        return

    # Check if course is non-numeric
    course = ""

    ret, frame = cap.read()

    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    centered_faces_enter = [face for face in face_rects if is_face_centered(face, frame.shape[1], frame.shape[0])]

    if len(centered_faces_enter) == 0:
        cv2.putText(frame, "No Face Detected!", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=pil_image)
        video_label.configure(image=img)
        video_label.image = img
        root.update()
        return

    x, y, w, h = centered_faces_enter[0]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face_image = cv2.cvtColor(
        frame[max(0, y - 1):min(frame.shape[0], y + h + 1), max(0, x - 1):min(frame.shape[1], x + w + 1)],
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

    image_data = cv2.imencode('.jpg', resized_face)[1].tobytes()

    # Insert data into the database
    for i in range(len(time_in_value)):
        insert_face_data(
            person_type,
            person_id,
            Name,
            course,
            section,
            subjects[i],  # Use the i-th subject
            schoolyear,
            time_in_value[i].strftime('%H:%M'),  # Format the i-th time_in_value
            time_out_value[i].strftime('%H:%M'),  # Format the i-th time_out_value
            weekdays[i],  # Use the i-th weekday
            image_data
        )

    shots_counter += 1

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    img = ImageTk.PhotoImage(image=pil_image)
    video_label.configure(image=img)
    video_label.image = img
    root.update()


def capture_face2():
    global shots_counter
    if shots_counter >= 100:
        print("Maximum number of shots reached.")
        return
    # Hide the displayed image
    hide_image()

    time_in_str = time_in_entry.get()
    time_out_str = time_out_entry.get()
    subject_str = subject_entry.get()
    weekday_str = weekday_entry.get()

    # Split the input strings by commas
    time_in_values = [t.strip() for t in time_in_str.split(',')]
    time_out_values = [t.strip() for t in time_out_str.split(',')]
    subjects = [s.strip() for s in subject_str.split(',')]
    weekdays = [w.strip() for w in weekday_str.split(',')]

    person_type = "Employee"
    section = ""
    schoolyear = ""
    person_id = person_id_entry.get()
    if not person_id:
        print("PersonID is required.")
        return

    Name = Name_entry.get()
    if not Name:
        print("Name is required.")
        return

    # Check if course is non-numeric
    course = ""

    ret, frame = cap2.read()

    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    centered_faces_enter = [face for face in face_rects if is_face_centered(face, frame.shape[1], frame.shape[0])]

    if len(centered_faces_enter) == 0:
        cv2.putText(frame, "No Face Detected!", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=pil_image)
        video_label.configure(image=img)
        video_label.image = img
        root.update()
        return

    x, y, w, h = centered_faces_enter[0]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face_image = cv2.cvtColor(
        frame[max(0, y - 1):min(frame.shape[0], y + h + 1), max(0, x - 1):min(frame.shape[1], x + w + 1)],
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

    image_data = cv2.imencode('.jpg', resized_face)[1].tobytes()

    # Insert data into the database
    for i in range(len(time_in_value)):
        insert_face_data(
            person_type,
            person_id,
            Name,
            course,
            section,
            subjects[i],  # Use the i-th subject
            schoolyear,
            time_in_value[i].strftime('%H:%M'),  # Format the i-th time_in_value
            time_out_value[i].strftime('%H:%M'),  # Format the i-th time_out_value
            weekdays[i],  # Use the i-th weekday
            image_data
        )

    shots_counter += 1

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    img = ImageTk.PhotoImage(image=pil_image)
    video_label.configure(image=img)
    video_label.image = img
    root.update()


# Define a variable to track camera feed status
camera_feed_active = False


# Function to hide the displayed image
def hide_image():
    if not camera_feed_active:
        image_label.pack_forget()


# Function to display an image in the GUI
def display_image(image_path):
    global image_label
    if not camera_feed_active:
        img = Image.open(image_path)

        # Convert the image to RGBA mode to preserve transparency
        img = img.convert("RGBA")

        # Create a PhotoImage object from the RGBA image
        img = ImageTk.PhotoImage(img)

        image_label = tk.Label(video_frame, image=img, bg="green")
        image_label.image = img  # Prevent image from being garbage collected
        image_label.pack()


# Create the GUI window
root = tk.Tk()
root.title("Employee Registration Window")

# Set the window size and position
window_width = 1000
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Configure window background
root.configure(bg="green")

# Frame for input fields on the left
input_frame = tk.Frame(root, bg="green")
input_frame.pack(side="left", padx=20, pady=20)

# Frame for video capture on the right
video_frame = tk.Frame(root, bg="green")
video_frame.pack(side="right", padx=20, pady=20)

# PersonID input field
person_id_label = tk.Label(input_frame, text="PersonID:", bg="green", fg="white", font=("Arial", 14))
person_id_label.pack()

person_id_entry = tk.Entry(input_frame, bg="white")
person_id_entry.pack()

# Name input field
Name_label = tk.Label(input_frame, text="Name:", bg="green", fg="white", font=("Arial", 14))
Name_label.pack()

Name_entry = tk.Entry(input_frame, bg="white")
Name_entry.pack()

# Subject input field
subject_label = tk.Label(input_frame, text="Subjects: \n Ex. Math,Science,History",
                         bg="green", fg="white", font=("Arial", 10))
subject_label.pack()

subject_entry = tk.Entry(input_frame, bg="white")
subject_entry.pack()

# Time in and time out input fields
time_in_label = tk.Label(input_frame, text="Time In: \n (HHMM - 24h format ex. 0800 = 8:00am)  \n"
                                           " Ex. 0800,1300,1700", bg="green", fg="white", font=("Arial", 10))
time_in_label.pack()

time_in_entry = tk.Entry(input_frame, bg="white")
time_in_entry.pack()

time_out_label = tk.Label(input_frame, text="Time Out: \n (HHMM - 24h format ex. 2000 = 8:00pm) \n "
                                            "Ex. 1000,1500,2000", bg="green", fg="white", font=("Arial", 10))
time_out_label.pack()

time_out_entry = tk.Entry(input_frame, bg="white")
time_out_entry.pack()

# Weekday input field
weekday_label = tk.Label(input_frame, text="Scheduled Day: \n Ex. (Monday,Tuesday,Wednesday,Thursday,Friday)",
                         bg="green", fg="white", font=("Arial", 10))
weekday_label.pack()

weekday_entry = tk.Entry(input_frame, bg="white")
weekday_entry.pack()


# Submit button
submit_button = tk.Button(input_frame, text="Submit", command=get_time_values,
                          bg="white", fg="black", font=("Arial", 16))
submit_button.pack()

# Bind the submit button to the change_button_color function when clicked
submit_button.bind("<Button-1>", change_button_color)

# Video label for displaying the webcam feed
video_label = tk.Label(video_frame, bg="white")
video_label.pack()

# Load and display the image
image_path = "C:/Users/USER/PycharmProjects/pythonProjectALLADIN/ALLA Interface/Resources/Logo1.png"
# Replace with the actual image path
display_image(image_path)


# Start the GUI event loop
root.mainloop()


# Check if time in and time out values are provided
if time_in_value is None or time_out_value is None:
    print("Time in and time out values are required.")


# Open the webcam
cap = cv2.VideoCapture(0)


# Create the PersonTable, SubjectTable, and CapturedFaces table
create_person_table()
create_subject_table()
create_captured_faces_table()

# Set the camera feed as active
camera_feed_active = True

# Initialize counter for capturing 100 shots
shots_counter = 0


while shots_counter < 100:
    capture_face()

# Release the video capture object and close the windows
cap.release()

# Show registration successful message box
messagebox.showinfo("Registration Successful, Face the Exit Camera",
                    "All Information have been registered successfully! Please Go To Exit Camera for another Shots")

# Initialize the second camera
cap2 = cv2.VideoCapture(1)  # Use the appropriate camera index (e.g., 1 for the second camera)

# Reset the shots counter for the second set of shots
shots_counter = 0

# Capture and process the second set of shots
while shots_counter < 100:
    capture_face2()

# Release the second camera
cap2.release()

cv2.destroyAllWindows()

# Show registration successful message box
messagebox.showinfo("Registration Successful", "All Information have been registered successfully!")
