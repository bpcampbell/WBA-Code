from multiprocessing import Process, Queue
from queue import Full, Empty
import cv2
from src.wingbeat_analyser import WingbeatAnalyzer
from src.config import CONFIG
import logging

class RealTimeWingbeatAnalyzer(Process):
    def __init__(self, output_queue: Queue, ready_queue: Queue, video_path: str):
        super().__init__()
        self.output_queue = output_queue
        self.ready_queue = ready_queue
        self.video_path = video_path
        self.running = True
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Main process loop"""
        self.logger.info("Starting RealTimeWingbeatAnalyzer process")
        
        # Initialize video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
            
        self.logger.info("Successfully opened video file")
        
        # Initialize analyzer
        analyzer = WingbeatAnalyzer(CONFIG)
        frame_count = 0
        
        try:
            # Get first frame for setup
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read first frame")
                
            # Setup points through UI
            analyzer.setup_points_ui(frame)
            self.logger.info("Points setup complete")
            
            # Signal that points are selected and analyzer is ready
            self.ready_queue.put(True)
            
            # Main processing loop
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("Reached end of video")
                    break
                    
                frame_count += 1
                # Process frame
                results = analyzer.process_frame(frame)
                
                # Put latest wingbeat amplitude in queue (non-blocking)
                try:
                    self.output_queue.put_nowait(results['delta_angle'])
                    if frame_count % 30 == 0:  # Log every 30 frames
                        self.logger.info(f"Frame {frame_count}: delta_angle = {results['delta_angle']:.2f}")
                except Full:
                    self.logger.debug("Queue full, updating with new value")
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(results['delta_angle'])
                    except Empty:
                        pass
                        
        finally:
            self.logger.info(f"Processed {frame_count} frames")
            cap.release()
            
    def stop(self):
        """Stop the analyzer process"""
        self.running = False 