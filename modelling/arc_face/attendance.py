# attendance.py
import csv
import os
from datetime import date, datetime


class AttendanceLogger:
    def __init__(self, path="attendance_arcface.csv"):
        self.path = path
        self.today = date.today().strftime("%Y-%m-%d")
        self.marked = set()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "date", "time"])
        else:
            with open(self.path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and row[1] == self.today:
                        self.marked.add(row[0])

    def mark(self, name):
        if name in self.marked:
            return False

        now = datetime.now()
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S")
            ])

        self.marked.add(name)
        print(f"[ATTENDANCE] Marked {name}")
        return True
