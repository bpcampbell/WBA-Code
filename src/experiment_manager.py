import cv2
import time
import logging
from multiprocessing import Queue
from queue import Empty
import numpy as np
from src.realtime_analyzer import RealTimeWingbeatAnalyzer
from src.optic_flow import OpticFlowGenerator

logger = logging.getLogger(__name__)

class ExperimentManager:
    def __init__(self, frame_size, video_path, min_amplitude=30, max_amplitude=120):
        logger.info(f"Initializing ExperimentManager with frame size: {frame_size}")
        self.frame_size = frame_size
        self.flow_generator = OpticFlowGenerator(frame_size)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.wing_beat_amplitude = 50
        self.cumulative_phase = 0
        
        # Setup analyzer communication
        self.wingbeat_queue = Queue(maxsize=1)
        self.analyzer = RealTimeWingbeatAnalyzer(self.wingbeat_queue, video_path)
        
    def map_wingbeat_to_speed(self, wingbeat_amplitude, gain):
        """Maps wingbeat amplitude to speed using gain."""
        wingbeat_amplitude = max(min(wingbeat_amplitude, self.max_amplitude), 
                               self.min_amplitude)
        normalized_amplitude = ((wingbeat_amplitude - self.min_amplitude) / 
                              (self.max_amplitude - self.min_amplitude))
        return gain * normalized_amplitude
        
    def run_experiment(self, experiment_name, gain_cycle):
        """Run experiment with given gain cycle."""
        # Start analyzer process
        self.analyzer.start()
        
        exp_start_time = time.perf_counter()
        last_time = time.perf_counter()
        data = []
        
        try:
            for duration, gain in gain_cycle:
                cycle_start_time = time.perf_counter()
                logger.info(f"Starting cycle with gain: {gain}")
                
                while time.perf_counter() - cycle_start_time < duration:
                    try:
                        self.wing_beat_amplitude = self.wingbeat_queue.get_nowait()
                        logger.debug(f"Current wingbeat amplitude: {self.wing_beat_amplitude}")
                    except Empty:
                        pass
                    
                    current_time = time.perf_counter() - exp_start_time
                    delta_time = time.perf_counter() - last_time
                    
                    speed = self.map_wingbeat_to_speed(self.wing_beat_amplitude, gain)
                    self.cumulative_phase -= speed * delta_time
                    
                    if gain > 0:
                        pattern = self.flow_generator.generate_pattern(self.cumulative_phase)
                    else:
                        pattern = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                    
                    # Display pattern
                    cv2.imshow('Optic Flow', pattern)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return data
                    
                    data.append([current_time, self.wing_beat_amplitude, speed])
                    last_time = time.perf_counter()
                    
        finally:
            self.analyzer.stop()
            self.analyzer.join()
            cv2.destroyAllWindows()
            
        return data 