from django.db import models
# models.py
from django.db import models

class Employee(models.Model):
    name = models.CharField(max_length=100)
    face_encoding = models.BinaryField(null=True, blank=True)

class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True, blank=True)
    date = models.DateField()
    time_in = models.TimeField(null=True, blank=True)
    time_out = models.TimeField(null=True, blank=True)
    
