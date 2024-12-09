from multiprocessing import Queue
from queue import Empty
import cv2, time, logging
import numpy as np
from src.realtime_analyzer import RealTimeWingbeatAnalyzer
from src.optic_flow import OpticFlowGenerator
from src.video_handler import VideoHandler
from src.point_selector import PointSelector
from src.config import CONFIG

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
        
        # Use optic flow config
        self.speed_scale = CONFIG['optic_flow']['speed_scale']
        
        # Setup analyzer communication
        self.wingbeat_queue = Queue(maxsize=1)
        self.ready_queue = Queue(maxsize=1)
        
        # Don't create flow generator yet
        self.flow_generator = None
        self.analyzer = None
        
        self.points = None
        self.video_handler = VideoHandler(video_path)
        
    def setup_points(self):
        """Initialize UI and get points from user"""
        logger.info("Starting point selection UI...")
        
        # Get initial frames for point selection
        initial_frames = []
        for _ in range(10):
            frame = self.video_handler.read_frame()
            if frame is None:
                raise RuntimeError("Failed to read initial frames")
            initial_frames.append(frame)

        # Launch point selector UI
        selector = PointSelector(initial_frames)
        result = selector.get_points()
        if result is None or result[0] is None:
            logger.info("Point selection cancelled by user")
            return False
            
        self.points, self.timestamp = result
        logger.info("Point selection complete")
        return True

    def initialize_analyzer(self):
        """Initialize and start the wingbeat analyzer"""
        if self.points is None:
            raise RuntimeError("Points must be selected before initializing analyzer")
            
        logger.info("Starting wingbeat analyzer...")
        self.analyzer = RealTimeWingbeatAnalyzer(
            self.wingbeat_queue,
            self.ready_queue,
            self.video_path,
            self.points  # Pass points to analyzer
        )
        self.analyzer.start()
        
        # Wait for analyzer to be ready
        try:
            if not self.ready_queue.get(timeout=60):
                raise RuntimeError("Analyzer initialization failed")
        except Empty:
            self.analyzer.stop()
            self.analyzer.join()
            raise RuntimeError("Analyzer initialization timed out")
            
        logger.info("Analyzer initialization complete")

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
        wingbeat_amplitude = abs(wingbeat_amplitude)
        
        # Clamp to valid range
        wingbeat_amplitude = max(min(wingbeat_amplitude, self.max_amplitude), 
                               self.min_amplitude)
        
        normalized_amplitude = ((wingbeat_amplitude - self.min_amplitude) / 
                              (self.max_amplitude - self.min_amplitude))
        
        # Apply speed scaling and ensure it's not zero
        speed = self.speed_scale * gain * normalized_amplitude
        
        return speed 
        
    def run_experiment(self, experiment_name, gain_cycle):
        """Run experiment with given gain cycle."""
        try:
            # First get points through UI
            if not self.setup_points():
                return None
                
            # Then initialize analyzer with points
            self.initialize_analyzer()
            
            # Initialize optic flow
            self.initialize_optic_flow()
            
            data = []
            exp_start_time = time.perf_counter()
            last_time = time.perf_counter()
            
            # Add frequency tracking
            pattern_times = []
            last_pattern_time = time.perf_counter()
            
            # Pre-allocate the zero pattern
            zero_pattern = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            
            # Initialize cycle tracking
            current_cycle_index = 0
            current_cycle_start = exp_start_time
            current_duration, current_gain = gain_cycle[0]
            
            # Main loop
            while True:
                # Check if we need to move to next cycle
                current_time = time.perf_counter()
                if current_time - current_cycle_start >= current_duration:
                    current_cycle_index += 1
                    if current_cycle_index >= len(gain_cycle):
                        break
                    current_duration, current_gain = gain_cycle[current_cycle_index]
                    current_cycle_start = current_time
                    logger.info(f"Starting cycle with gain: {current_gain}")
                
                # Try to get new wingbeat data
                try:
                    wingbeat_data = self.wingbeat_queue.get_nowait()
                    
                    if wingbeat_data is not None:
                        delta_time = current_time - last_time
                        average_wingbeat_amplitude = 0.5*(wingbeat_data['left_angle'] + wingbeat_data['right_angle'])
                        speed = self.map_wingbeat_to_speed(average_wingbeat_amplitude, current_gain)
                        self.cumulative_phase -= speed * delta_time
                        
                        # Create data point
                        data_point = {
                            'time': current_time - exp_start_time,
                            'left_angle': wingbeat_data['left_angle'],
                            'right_angle': wingbeat_data['right_angle'],
                            'delta_angle': wingbeat_data['delta_angle'],
                            'frame': wingbeat_data['frame'],
                            'phase': self.cumulative_phase,
                            'speed': speed,
                            'gain': current_gain
                        }
                        data.append(data_point)
                        last_time = current_time
                        
                except Empty:
                    pass
                
                # Always update pattern (regardless of new wingbeat data)
                current_pattern_time = time.perf_counter()
                pattern_times.append(current_pattern_time - last_pattern_time)
                last_pattern_time = current_pattern_time
                
                # Use pre-allocated zero pattern
                if current_gain > 0:
                    pattern = self.flow_generator.generate_pattern(self.cumulative_phase)
                else:
                    pattern = zero_pattern
                
                cv2.imshow('Optic Flow', pattern)
                cv2.pollKey()
            
            # Calculate and log frequencies
            if pattern_times:
                pattern_freq = 1.0 / np.mean(pattern_times)
                logger.info(f"Average pattern display frequency: {pattern_freq:.2f} Hz")
            
            return data, self.timestamp
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            if self.analyzer:
                self.analyzer.stop()
                self.analyzer.join()
            return None, None