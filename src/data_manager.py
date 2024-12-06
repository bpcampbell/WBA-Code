import csv
from pathlib import Path
import logging
from datetime import datetime
from time import perf_counter_ns

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, output_path):
        """Initialize data manager.
        
        Args:
            output_path: Path to output CSV file (already includes timestamp)
        """
        self.output_path = Path(output_path)
        self.fieldnames = ['left_angle', 'right_angle', 'delta_angle', 
                          'time', 'frame']
        self.start_time = None
        self.first_timestamp = None
        self.csv_file = None
        self.writer = None
        self.setup_csv()
        
    def setup_csv(self):
        """Initialize CSV file with headers"""
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open CSV file and initialize writer
        self.csv_file = open(self.output_path, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        # Record start time in nanoseconds
        self.start_time = perf_counter_ns()
        logger.info(f"Initialized data recording to: {self.output_path}")
        
    def write_row(self, data):
        """Write a row of data to CSV.
        
        Args:
            data: Dictionary containing wing analysis results
        """
        if not self.writer:
            raise RuntimeError("CSV writer not initialized")
            
        # Calculate current time in milliseconds
        current_time = perf_counter_ns() / 1_000_000  # Current time in ms
        
        # Store first timestamp if not set
        if self.first_timestamp is None:
            self.first_timestamp = current_time
            
        # Calculate elapsed time relative to first timestamp
        elapsed_time = current_time - self.first_timestamp
        
        row_data = {
            'left_angle': data['left_angle'],
            'right_angle': data['right_angle'],
            'delta_angle': data['delta_angle'],
            'time': elapsed_time,
            'frame': data.get('frame', 0)
        }
        
        self.writer.writerow(row_data)
        self.csv_file.flush()
        
    def close(self):
        """Close the CSV file"""
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"Closed data recording file: {self.output_path}") 