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
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.wing_beat_amplitude = 50
        self.cumulative_phase = 0
        self.video_path = video_path
        
        # Add speed scaling factor
        self.speed_scale = 100.0  # Adjust this to make movement more visible
        
        # Setup analyzer communication
        self.wingbeat_queue = Queue(maxsize=1)
        self.ready_queue = Queue(maxsize=1)
        
        # Don't create flow generator yet
        self.flow_generator = None
        self.analyzer = None
        
    def initialize_analyzer(self):
        """Initialize and start the wingbeat analyzer"""
        logger.info("Starting wingbeat analyzer for point selection...")
        self.analyzer = RealTimeWingbeatAnalyzer(
            self.wingbeat_queue, 
            self.ready_queue, 
            self.video_path
        )
        self.analyzer.start()
        
        # Wait for points to be selected
        logger.info("Waiting for point selection to complete...")
        self.ready_queue.get()
        logger.info("Point selection complete")
        
    def initialize_optic_flow(self):
        """Initialize optic flow after point selection"""
        logger.info("Initializing optic flow display...")
        # Adjust wavelength to change pattern size
        self.flow_generator = OpticFlowGenerator(
            frame_size=self.frame_size,
            wavelength=50,  # Smaller wavelength = more pattern cycles
            amplitude=255   # Maximum contrast
        )
        cv2.namedWindow("Optic Flow", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Optic Flow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def map_wingbeat_to_speed(self, wingbeat_amplitude, gain):
        """Maps wingbeat amplitude to speed using gain."""
        # Take absolute value of wingbeat amplitude since negative values shouldn't affect direction
        wingbeat_amplitude = abs(wingbeat_amplitude)
        
        # Clamp to valid range
        wingbeat_amplitude = max(min(wingbeat_amplitude, self.max_amplitude), 
                               self.min_amplitude)
        
        normalized_amplitude = ((wingbeat_amplitude - self.min_amplitude) / 
                              (self.max_amplitude - self.min_amplitude))
        
        # Apply speed scaling and ensure it's not zero
        speed = self.speed_scale * gain * normalized_amplitude
        return max(speed, 0.1)  # Ensure minimum movement speed
        
    def run_experiment(self, experiment_name, gain_cycle):
        """Run experiment with given gain cycle."""
        # First initialize and wait for wingbeat analyzer
        self.initialize_analyzer()
        
        # Then initialize optic flow
        self.initialize_optic_flow()
        
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
                        logger.info(f"Got wingbeat amplitude: {self.wing_beat_amplitude:.2f}")
                    except Empty:
                        logger.debug("No new wingbeat data available")
                        pass
                    
                    current_time = time.perf_counter() - exp_start_time
                    delta_time = time.perf_counter() - last_time

                    speed = self.map_wingbeat_to_speed(self.wing_beat_amplitude, gain)
                    self.cumulative_phase -= speed * delta_time
                    
                    if gain > 0:
                        pattern = self.flow_generator.generate_pattern(self.cumulative_phase)
                        logger.debug(f"Generated pattern with phase: {self.cumulative_phase:.2f}")
                    else:
                        logger.info('No gain, setting pattern to zero')
                        pattern = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                    
                    # Display pattern without delay
                    cv2.imshow('Optic Flow', pattern)
                    cv2.pollKey()  # Non-blocking window update

                    data.append([current_time, self.wing_beat_amplitude, speed])
                    last_time = time.perf_counter()
                    
        finally:
            self.analyzer.stop()
            self.analyzer.join()
            cv2.destroyAllWindows()
            
        return data 