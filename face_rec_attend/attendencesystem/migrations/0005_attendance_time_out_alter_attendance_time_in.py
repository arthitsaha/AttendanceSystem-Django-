# Generated by Django 5.0.4 on 2024-04-27 01:45

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("attendencesystem", "0004_rename_time_attendance_time_in"),
    ]

    operations = [
        migrations.AddField(
            model_name="attendance",
            name="time_out",
            field=models.TimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="attendance",
            name="time_in",
            field=models.TimeField(blank=True, null=True),
        ),
    ]
