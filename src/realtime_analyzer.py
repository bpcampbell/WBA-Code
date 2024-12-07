from multiprocessing import Process, Queue
from queue import Full, Empty
import cv2
from src.wingbeat_analyser import WingbeatAnalyzer
from src.config import CONFIG
import logging
import time
from datetime import datetime
from pathlib import Path
from src.video_handler import VideoHandler

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
                
            self.logger.info("Successfully opened video file")
            
            # Initialize analyzer
            analyzer = WingbeatAnalyzer(CONFIG)
            frame_count = 0
            
            # Buffer first 10 frames for setup
            initial_frames = []
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read initial frames")
                initial_frames.append(frame)
                
            # Setup points through UI using averaged frame
            try:
                if not analyzer.setup_points_ui(initial_frames):
                    self.logger.info("Point selection cancelled by user")
                    self.ready_queue.put(False)  # Signal that setup failed
                    return
                # Add a small delay to ensure window cleanup
                time.sleep(0.2)  # 200ms delay
            except Exception as e:
                self.logger.error(f"Point selection failed: {str(e)}")
                self.ready_queue.put(False)
                return
            
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
                
                # Write the cropped frame to video
                cropped_frame = analyzer.get_last_cropped_frame()
                if cropped_frame is not None:
                    video_handler.write_frame(cropped_frame)
                
                # Put latest wingbeat amplitude in queue (non-blocking)
                try:
                    self.output_queue.put_nowait(results['delta_angle'])
                    if frame_count % 30 == 0:
                        self.logger.info(f"Frame {frame_count}: delta_angle = {results['delta_angle']:.2f}")
                except Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(results['delta_angle'])
                    except Empty:
                        pass
                        
        finally:
            self.logger.info(f"Processed {frame_count} frames")
            cap.release()
            if 'video_handler' in locals():
                video_handler.release()
            
    def stop(self):
        """Stop the analyzer process"""
        self.running = False 