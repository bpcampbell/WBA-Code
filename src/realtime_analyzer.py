from src.wingbeat_analyser import WingbeatAnalyzer
from src.video_handler import VideoHandler
from multiprocessing import Process, Queue
from queue import Full, Empty
import cv2, logging, time
from src.config import CONFIG
from datetime import datetime
from pathlib import Path
import numpy as np

class RealTimeWingbeatAnalyzer(Process):
    def __init__(self, output_queue: Queue, ready_queue: Queue, video_path: str, points: dict):
        super().__init__()
        self.output_queue = output_queue
        self.ready_queue = ready_queue
        self.video_path = video_path
        self.points = points
        self.running = True
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Main process loop"""
        self.logger.info("Starting RealTimeWingbeatAnalyzer process")
        frame_count = 0  # Initialize frame_count at the start
        
        # Add frequency tracking
        frame_times = []
        last_frame_time = time.perf_counter()
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            # Initialize video writer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = Path("output") / f"wing_analysis_{timestamp}.mp4"
            video_handler = VideoHandler(
                source_path=self.video_path,
                output_path=output_video_path,
                make_video=True
            )
                
            # Initialize analyzer with provided points
            analyzer = WingbeatAnalyzer(CONFIG)
            analyzer.points = self.points  # Set points directly
            first_frame = video_handler.read_frame()
            if first_frame is None:
                raise RuntimeError("Failed to read first frame")
            analyzer.initialize_processing(first_frame)
            
            # Signal that analyzer is ready
            self.ready_queue.put(True)
            
            # Main processing loop
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("Reached end of video")
                    break
                    
                frame_count += 1
                
                # Trackframe processing frequency
                current_frame_time = time.perf_counter()
                frame_times.append(current_frame_time - last_frame_time)
                last_frame_time = current_frame_time
                
                # Process frame
                results = analyzer.process_frame(frame)
                
                # Write the cropped frame to video if available
                cropped_frame = analyzer.get_last_cropped_frame()
                if cropped_frame is not None:
                    video_handler.write_frame(cropped_frame)
                
                # Put latest wingbeat amplitude in queue (non-blocking)
                try:
                    self.output_queue.put_nowait({
                        'left_angle': results['left_angle'],
                        'right_angle': results['right_angle'],
                        'delta_angle': results['delta_angle'],
                        'frame': frame_count
                    })
                    if frame_count % 30 == 0:  # Log every 30 frames
                        self.logger.info(f"Frame {frame_count}: delta_angle = {results['delta_angle']:.2f}")
                except Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait({
                            'left_angle': results['left_angle'],
                            'right_angle': results['right_angle'],
                            'delta_angle': results['delta_angle'],
                            'frame': frame_count
                        })
                    except Empty:
                        pass
                        
            # Calculate and log frequency
            if frame_times:
                frame_freq = 1.0 / np.mean(frame_times)
                self.logger.info(f"Average frame processing frequency: {frame_freq:.2f} Hz")
                
            self.logger.info(f"Processed {frame_count} frames")
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            self.ready_queue.put(False)
        finally:
            cap.release()
            if 'video_handler' in locals():
                video_handler.release()
            
    def stop(self):
        """Stop the analyzer process"""
        self.running = False 