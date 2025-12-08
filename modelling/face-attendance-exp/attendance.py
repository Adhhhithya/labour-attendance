"""
attendance.py
CSV-based attendance logging with once-per-day per-person enforcement.
"""
import csv
import os
from datetime import date, datetime


class AttendanceLogger:
    """
    Handles attendance.csv file.
    - Appends new rows with name, date, time.
    - Ensures each person is only logged once per day.
    """

    def __init__(self, path: str = "attendance.csv"):
        """
        Initialize attendance logger.
        
        Args:
            path: Path to attendance CSV file.
        """
        self.path = path
        self.today = date.today().strftime("%Y-%m-%d")
        self.marked_today = set()
        self._init_file()

    def _init_file(self):
        """Create file if missing; load today's entries into marked_today set."""
        if not os.path.exists(self.path):
            with open(self.path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "date", "time"])
            print(f"[INFO] Created new attendance file: {self.path}")
        else:
            with open(self.path, mode="r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        name, row_date = row[0], row[1]
                        if row_date == self.today:
                            self.marked_today.add(name)
            print(f"[INFO] Already marked today: {self.marked_today}")

    def mark_if_live_and_not_marked(self, name: str) -> bool:
        """
        Mark attendance for name if not already marked today.
        
        Args:
            name: Person name to mark.
            
        Returns:
            True if a new row was written, False if already marked today.
        """
        if name in self.marked_today:
            return False

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])

        self.marked_today.add(name)
        print(f"[ATTENDANCE] Marked {name} at {date_str} {time_str}")
        return True
