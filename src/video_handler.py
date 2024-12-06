import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoHandler:
    def __init__(self, source_path, output_path=None, make_video=False, frame_size=None):
        """Initialize video capture and writer.
        
        Args:
            source_path: Path to source video file
            output_path: Path to output video file (if recording)
            make_video: Whether to record output video
            frame_size: Tuple of (width, height) for output video
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path) if output_path else None
        self.make_video = make_video
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Use provided frame size for output if specified
        if frame_size:
            self.output_width, self.output_height = frame_size
        else:
            self.output_width, self.output_height = self.width, self.height
        
        # Initialize video writer if needed
        self.writer = None
        if self.make_video:
            self.setup_video_writer()
            
    def setup_video_writer(self):
        """Initialize video writer with source video properties"""
        if not self.output_path:
            raise ValueError("Output path not specified for video recording")
            
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.output_width, self.output_height)
        )
        
    def read_frame(self):
        """Read and return a frame from the video source.
        
        Returns:
            numpy.ndarray: Video frame or None if no frames left
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def write_frame(self, frame):
        """Write frame to output video if enabled.
        
        Args:
            frame: Video frame to write
        """
        if self.make_video and self.writer:
            self.writer.write(frame)
            
    def release(self):
        """Release video capture and writer resources"""
        self.cap.release()
        if self.writer:
            self.writer.release()
            
    def update_frame_size(self, frame_size):
        """Update the output frame size and reinitialize writer if needed."""
        self.output_width, self.output_height = frame_size
        if self.make_video:
            if self.writer is not None:
                self.writer.release()
            self.setup_video_writer()