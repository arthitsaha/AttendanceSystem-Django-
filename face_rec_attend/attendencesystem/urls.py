# attendencesystem/urls.py
from django.urls import path
from . import views

app_name = 'attendencesystem'

urlpatterns = [
    path('', views.home, name='home'),  # Add this line
    path('signup/', views.signup_user, name='signup'),
    path('login/',views.login_user,name='login'),
    path('logout/',views.logout_user,name='logout'),
    path('create_dataset/',views.create_dataset,name="create_dataset"),
    path('mark_attendance_in/',views.mark_attendance_in,name='mark_attendance_in'),
    path('mark-attendance-face/', views.mark_attendance_with_face_recognition, name='mark_attendance_with_face_recognition'),
    path('mark_attendance_out/',views.mark_attendance_out,name='mark_attendance_out'),
    path('attendance_graph/',views.attendance_graph,name='attendance_graph'),
    
]
