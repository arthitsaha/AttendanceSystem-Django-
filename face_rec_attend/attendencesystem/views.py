from distutils import file_util
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import Employee, Attendance
import face_recognition
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout as django_logout
from django.shortcuts import redirect
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect ,FileResponse
import os
import dlib
import cv2
import imutils
import numpy as np
import face_recognition
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
from .models import Employee, Attendance
from django.contrib import messages
import matplotlib as plt
import matplotlib.pyplot as plt
from django.db.models import Count
from django.db.models.functions import ExtractWeekDay, ExtractDay, ExtractMonth, ExtractYear
from django.db.models import Count, F, ExpressionWrapper, fields ,Value ,Sum
import datetime
from io import BytesIO


@login_required
def home(request):
    employee_name = request.user.username
    try:
        employee = Employee.objects.get(name=employee_name)
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found.')
        return redirect('attendencesystem:logout')  # Redirect to logout or a proper error page.

    # Get all attendance records for the employee, not just today's.
    records = Attendance.objects.filter(employee=employee).order_by('-date')

    return render(request, 'home.html', {'records': records})


def signup_user(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Authenticate and login the user
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            # Redirect to the home page or any other desired page
            return render(request, 'train.html')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Check for dataset folder (assuming the folder name matches username)
            dataset_folder = os.path.join(settings.BASE_DIR, 'data', 'training_dataset', username)
            if not os.path.exists(dataset_folder):
                print("Dataset folder doesn't exist")   
                return render(request, 'train.html') 
            else:
                print("Dataset folder exist")
                return render(request,'home.html')
        else:
            # Display error message for invalid credentials
            error_message = 'Invalid username or password'
            return render(request, 'registration/login.html', {'error_message': error_message})
    else:
        return render(request, 'registration/login.html')


    
def logout_user(request):
    django_logout(request)
    return redirect(reverse('attendencesystem:login'))


@login_required


def create_dataset(request):
    username = request.user.username
    dataset_dir = os.path.join( 'data', 'training_dataset', username)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print("[INFO] Loading the Haar cascade classifier for face detection")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("[INFO] Initializing Video capture")
    cap = cv2.VideoCapture(0)  # Default camera

    if not cap.isOpened():
        return render(request, 'train.html', {'error': 'Unable to open the camera.'})

    sampleNum = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))  # Resize the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade classifier
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            continue  # No faces found, skip frame

        for (x, y, w, h) in faces:
            try:
                face_roi = frame[y:y+h, x:x+w]  # Region of interest containing the face in color
                cv2.imwrite(os.path.join(dataset_dir, f'{sampleNum}.jpg'), face_roi)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                sampleNum += 1
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        cv2.imshow("Add Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if sampleNum >= 30:  # Redirect after capturing 30 images
            break

    cap.release()
    cv2.destroyAllWindows()

    if sampleNum >= 30:
        print("[INFO] Training the recognizer with the dataset.")
        train_face_recognizer(dataset_dir)
        return render(request, 'home.html', {'success': 'Dataset creation completed.'})

    else:
        return render(request, 'train.html', {'error': 'Unable to capture any faces.'})
    


def train_face_recognizer(dataset_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                id = os.path.basename(root)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                face_samples.append(image)
                ids.append(int(id))

    recognizer.train(face_samples, np.array(ids))
    recognizer.save('trainer/trainer.yml')

def mark_attendance_with_face_recognition(request):
    username = request.user.username
    dataset_dir = os.path.join('data', 'training_dataset', username)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(roi_gray)
            if (confidence > 70 and str(id) == username):
                mark_attendance_in(request)
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('attendencesystem:home')





@login_required

def mark_attendance_in(request):
    employee_name = request.user.username
    try:
        employee = Employee.objects.get(name=employee_name)
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found.')
        return redirect('attendencesystem:home')

    if request.method == 'POST':
        today = timezone.localdate()
        attendance_record, created = Attendance.objects.get_or_create(
            employee=employee,
            date=today,
            defaults={'time_in': timezone.localtime()}
        )
        if not created and attendance_record.time_in:
            messages.info(request, 'Attendance already marked for today.')
        elif created:
            messages.success(request, 'Attendance marked successfully.')
        return redirect('attendencesystem:home')

    # This part of returning records is not needed here if you always redirect after POST
    # records = Attendance.objects.filter(employee=employee).order_by('-date')
    # return render(request, 'home.html', {'records': records})

@login_required
def mark_attendance_out(request):
    employee_name = request.user.username
    try:
        employee = Employee.objects.get(name=employee_name)
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found.')
        return redirect('attendencesystem:home')

    if request.method == 'POST':
        today = timezone.localdate()
        try:
            attendance_record = Attendance.objects.get(employee=employee, date=today)
            if attendance_record.time_out is None:
                attendance_record.time_out = timezone.localtime()
                attendance_record.save()
                messages.success(request, 'Time out marked successfully.')
            else:
                messages.info(request, 'Time out has already been marked.')
        except Attendance.DoesNotExist:
            messages.error(request, 'No attendance record found for today to mark time out.')
        return redirect('attendencesystem:home')

    # Redirect is always happening, so no need to handle GET specifically here
    

import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.http import HttpResponse
from .models import Attendance  # Import Attendance model
import datetime  # Import datetime module

def attendance_graph(request):
    # Fetch attendance data from the database
    attendance_data = Attendance.objects.values_list('date', 'time_in', 'time_out')
    
    # Calculate total working hours for each weekday
    weekday_hours = [0] * 7  # Initialize with 7 zeros (one for each weekday)
    
    for date, time_in, time_out in attendance_data:
        if time_in and time_out:
            # Combine date and time to create datetime objects
            datetime_in = datetime.datetime.combine(date, time_in)
            datetime_out = datetime.datetime.combine(date, time_out)
            
            # Calculate the duration between time_in and time_out
            duration = datetime_out - datetime_in
            total_hours = duration.total_seconds() / 3600  # Convert to hours
            
            # Add the total hours to the corresponding weekday
            weekday = datetime_in.weekday()
            weekday_hours[weekday] += total_hours
    
    # Prepare data for the graph
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    x = weekdays
    y = weekday_hours
    
    # Create the graph
    plt.figure(figsize=(8, 6))
    plt.bar(x, y)
    plt.xlabel('Weekday')
    plt.ylabel('Total Working Hours')
    plt.title('Total Working Hours Graph')
    
    # Save the graph to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode the graph image as base64
    graph_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return HttpResponse(graph_image)









