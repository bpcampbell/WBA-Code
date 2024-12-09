from time import perf_counter_ns
from pathlib import Path
import csv, logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, output_path):
        """Initialize data manager."""
        self.output_path = Path(output_path)
        # Update fieldnames to match what we actually want to save
        self.fieldnames = [
            'time',
            'left_angle',
            'right_angle', 
            'delta_angle',
            'frame',
            'phase',
            'speed',
            'gain'
        ]
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
        """Write a row of data to CSV."""
        if not self.writer:
            raise RuntimeError("CSV writer not initialized")
            
        current_time = perf_counter_ns() / 1_000_000  
        
        if self.first_timestamp is None:
            self.first_timestamp = current_time
            
        elapsed_time = current_time - self.first_timestamp
        
        # Ensure all required fields are present with default values if needed
        row_data = {
            'time': elapsed_time,
            'left_angle': data.get('left_angle', 0),
            'right_angle': data.get('right_angle', 0),
            'delta_angle': data.get('delta_angle', 0),
            'frame': data.get('frame', 0),
            'phase': data.get('phase', 0),
            'speed': data.get('speed', 0),
            'gain': data.get('gain', 0)
        }
        
        self.writer.writerow(row_data)
        self.csv_file.flush()  # Ensure data is written immediately
        
    def close(self):
        """Close the CSV file"""
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"Closed data recording file: {self.output_path}") 