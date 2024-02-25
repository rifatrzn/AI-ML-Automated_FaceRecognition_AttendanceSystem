from datetime import datetime

def log_attendance(person_name):
    timestamp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    with open("attendance_log.csv", "a") as file:
        file.write(f"{timestamp}, {person_name}\n")
