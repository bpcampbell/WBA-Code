from multiprocessing import Process, Queue
from queue import Full, Empty
import cv2
from src.wingbeat_analyser import WingbeatAnalyzer
from src.config import CONFIG

class RealTimeWingbeatAnalyzer(Process):
    def __init__(self, output_queue: Queue, video_path: str):
        super().__init__()
        self.output_queue = output_queue
        self.video_path = video_path
        self.running = True
        
    def run(self):
        """Main process loop"""
        # Initialize video instead of camera
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
            
        # Initialize analyzer
        analyzer = WingbeatAnalyzer(CONFIG)
        
        try:
            # Get first frame for setup
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read first frame")
                
            # Setup points through UI
            analyzer.setup_points_ui(frame)
            
            # Main processing loop
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                results = analyzer.process_frame(frame)
                
                # Put latest wingbeat amplitude in queue (non-blocking)
                try:
                    self.output_queue.put_nowait(results['delta_angle'])
                except Full:
                    # Queue is full, remove old value and put new one
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(results['delta_angle'])
                    except Empty:
                        pass
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def stop(self):
        """Stop the analyzer process"""
        self.running = False 