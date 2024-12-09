import customtkinter as ctk
from PIL import Image, ImageTk
import logging, json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from src.config import CONFIG

logger = logging.getLogger(__name__)

class PointSelector:
    def __init__(self, frames):
        # Set theme once at application level
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Store frames
        self.frames = frames
        self.current_frame_idx = 0
        
        # Initialize main window
        self.root = self._setup_main_window()
        
        # Create main container
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color="black"
        )
        self.main_container.pack(fill="both", expand=True)
        
        # Load and validate frame
        self.frame = self._load_frame(self.frames[self.current_frame_idx])
        
        # Create UI components
        self.canvas = self._create_canvas()
        self.notes_panel = self._create_notes_panel()
        self.control_panel = self._create_control_panel()
        
        # Initialize state
        self._init_state()
        
        # Bind events
        self._bind_events()
        
        # Center window
        self._center_window()

    def _setup_main_window(self):
        root = ctk.CTk()
        root.title("Wing Analysis Setup")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.configure(fg_color="black")
        
        # Set a reasonable initial window size
        root.geometry("1024x768")  # You can adjust this default size
        
        # Allow window to be resizable
        root.resizable(True, True)
        return root

    def _load_frame(self, frame):
        try:
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(self.image)
            return frame
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def _create_canvas(self):
        # Create a canvas container to handle resizing
        canvas_container = ctk.CTkFrame(
            self.main_container,
            fg_color="black"
        )
        canvas_container.pack(fill="both", expand=True)
        
        # Create the canvas
        canvas = tk.Canvas(
            canvas_container,
            bg='black',
            highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)
        
        # Initial image placement
        self.update_canvas_image(canvas)
        
        # Bind resize event
        canvas_container.bind("<Configure>", lambda e: self.update_canvas_image(canvas))
        
        return canvas
        
    def update_canvas_image(self, canvas):
        """Update canvas and image size to fill window while maintaining aspect ratio"""
        # Get container size
        container_width = self.main_container.winfo_width()
        container_height = self.main_container.winfo_height()
        
        if container_width <= 1 or container_height <= 1:  # Skip invalid sizes
            return
            
        # Calculate aspect ratios
        image_ratio = self.frame.shape[1] / self.frame.shape[0]
        container_ratio = container_width / container_height
        
        # Calculate new dimensions
        if container_ratio > image_ratio:
            # Container is wider than image
            height = container_height
            width = int(height * image_ratio)
        else:
            # Container is taller than image
            width = container_width
            height = int(width / image_ratio)
            
        # Resize canvas
        canvas.configure(width=container_width, height=container_height)
        
        # Resize image
        img_resized = self.image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_resized)
        
        # Calculate center position
        x = (container_width - width) // 2
        y = (container_height - height) // 2
        
        # Update image on canvas
        canvas.delete("all")  # Remove old image and all points/lines
        canvas.create_image(x, y, anchor="nw", image=self.photo, tags="image")
        
        # Redraw points and lines if they exist
        if hasattr(self, 'points') and self.click_count > 0:
            self._redraw_points_and_lines()

    def _create_control_panel(self):
        panel = ctk.CTkFrame(
            self.root,  # Attach to root instead of main_container
            corner_radius=10,
            width=280,
            height=90,
            fg_color="#1a1a1a"
        )
        # Position panel at bottom center
        panel.place(
            relx=0.5,
            rely=1.0,
            anchor="s",
            y=-20  # 20 pixels from bottom
        )
        
        # Prevent frame from shrinking
        panel.grid_propagate(False)
        panel.pack_propagate(False)
        
        # Add instruction label first
        self.instruction_text = ctk.StringVar()
        ctk.CTkLabel(
            panel,
            textvariable=self.instruction_text,
            font=("Helvetica", 14),
            text_color="white"
        ).pack(pady=(5, 0))
        
        # Create button container and center it
        button_container = ctk.CTkFrame(
            panel,
            fg_color="transparent",
            width=260
        )
        button_container.pack(expand=True)
        button_container.pack_propagate(False)
        
        # Add new frame button
        self.new_frame_button = ctk.CTkButton(
            button_container,
            text="New Frame",
            command=self.next_frame,
            font=("Helvetica", 14),
            corner_radius=6,
            width=120,
            height=28,
            fg_color="black",
            text_color="white",
            hover_color="#333333"
        )
        self.new_frame_button.pack(side="left", padx=2)
        
        # Add start button
        self.start_button = ctk.CTkButton(
            button_container,
            text="Start Experiment",
            command=self.start_experiment,
            state="disabled",
            font=("Helvetica", 14),
            corner_radius=6,
            width=120,
            height=28,
            fg_color="black",
            text_color="white",
            hover_color="#333333"
        )
        self.start_button.pack(side="right", padx=2)
        
        return panel

    def _create_notes_panel(self):
        """Create floating notes panel"""
        panel = ctk.CTkFrame(
            self.root,
            corner_radius=10,
            width=300,
            height=460,  # Increased height to accommodate new field
            fg_color="#1a1a1a"
        )
        # Position panel at right side
        panel.place(
            relx=1.0,
            rely=0.5,
            anchor="e",
            x=-20  # 20 pixels from right edge
        )
        
        # Prevent frame from shrinking
        panel.grid_propagate(False)
        panel.pack_propagate(False)
        
        # Add experiment name entry
        name_frame = ctk.CTkFrame(
            panel,
            fg_color="transparent"
        )
        name_frame.pack(pady=(15, 0), padx=20, fill="x")
        
        ctk.CTkLabel(
            name_frame,
            text="Experiment Name:",
            font=("Helvetica", 12),
            text_color="white"
        ).pack(side="left", padx=(0, 10))
        
        self.name_entry = ctk.CTkEntry(
            name_frame,
            font=("Helvetica", 12),
            width=140,
            fg_color="black",
            text_color="white"
        )
        self.name_entry.pack(side="right")
        self.name_entry.insert(0, CONFIG['experiment']['name'])  # Set default name
        
        # Add title label for notes
        ctk.CTkLabel(
            panel,
            text="Experiment Notes",
            font=("Helvetica", 16, "bold"),
            text_color="white"
        ).pack(pady=(15, 5))
        
        # Add text box
        self.notes_text = ctk.CTkTextbox(
            panel,
            width=260,
            height=330,
            font=("Helvetica", 12),
            fg_color="black",
            text_color="white",
            corner_radius=6
        )
        self.notes_text.pack(padx=20, pady=(5, 15))
        
        return panel

    def _init_state(self):
        """Initialize internal state"""
        self.points = {f'x{i}': None for i in range(5)}
        self.points.update({f'y{i}': None for i in range(5)})
        self.point_markers = []
        self.lines = []
        self.click_count = 0
        self.active_point = None
        self._drag_data = {"x": 0, "y": 0}
        self.experiment_started = False
        
        self.update_instructions()

    def _bind_events(self):
        """Bind all event handlers"""
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

    def _center_window(self):
        """Center window on screen"""
        self.root.update()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"+{x}+{y}")

    def update_instructions(self):
        instructions = [
            "1. Click the centre of the body",
            "2. Click the centre of the head",
            "3. Click a point to track on the wing",
            "4. Click the left wing hinge",
            "5. Click the right wing hinge"
        ]
        if self.click_count < 5:
            self.instruction_text.set(instructions[self.click_count])
        else:
            self.instruction_text.set("Points selected - Ready to start!")
            self.start_button.configure(state="normal")
        
    def on_press(self, event):
        """Handle mouse press events"""
        # Get the current image position and size
        container_width = self.main_container.winfo_width()
        container_height = self.main_container.winfo_height()
        
        # Calculate current image dimensions
        image_ratio = self.frame.shape[1] / self.frame.shape[0]
        container_ratio = container_width / container_height
        
        if container_ratio > image_ratio:
            height = container_height
            width = int(height * image_ratio)
        else:
            width = container_width
            height = int(width / image_ratio)
            
        # Calculate image offset
        x_offset = (container_width - width) // 2
        y_offset = (container_height - height) // 2
        
        # Get event coordinates in display space
        display_x = event.x
        display_y = event.y
        
        # Transform click coordinates to original image space
        x = int((display_x - x_offset) * (self.frame.shape[1] / width))
        y = int((display_y - y_offset) * (self.frame.shape[0] / height))
        
        # Validate coordinates are within image bounds
        if not (0 <= x < self.frame.shape[1] and 0 <= y < self.frame.shape[0]):
            logger.warning("Click outside image bounds")
            return
        
        # Store the initial click position
        self._drag_data["x"] = x
        self._drag_data["y"] = y
        
        # Check for existing point click with improved hit detection
        for i, point in enumerate(self.point_markers):
            coords = self.canvas.coords(point)
            if coords:
                # Calculate center of point in display coordinates
                point_center_x = (coords[0] + coords[2]) / 2
                point_center_y = (coords[1] + coords[3]) / 2
                
                # Check if click is within point area using display coordinates
                if ((display_x - point_center_x) ** 2 + 
                    (display_y - point_center_y) ** 2 < 100):  # Reduced radius for better accuracy
                    self.active_point = i
                    return
        
        # Create new point if we haven't placed all points
        if self.click_count < 5:
            point_num = self.click_count
            self.points[f'x{point_num}'] = x
            self.points[f'y{point_num}'] = y
            
            # Transform point coordinates for display
            display_x = int(x * (width / self.frame.shape[1])) + x_offset
            display_y = int(y * (height / self.frame.shape[0])) + y_offset
            
            point = self.canvas.create_oval(
                display_x-6, display_y-6, display_x+6, display_y+6,
                fill='#00ff00',
                outline='white',
                width=2
            )
            self.point_markers.append(point)
            
            # Draw lines
            if self.click_count == 1:
                self.draw_line(0, 1)
            elif self.click_count == 3:
                self.draw_line(0, 3)
            elif self.click_count == 4:
                self.draw_line(0, 4)
                
            self.click_count += 1
            self.update_instructions()

    def on_drag(self, event):
        """Handle mouse drag events"""
        if self.active_point is not None:
            # Get current image dimensions and position
            container_width = self.main_container.winfo_width()
            container_height = self.main_container.winfo_height()
            
            image_ratio = self.frame.shape[1] / self.frame.shape[0]
            container_ratio = container_width / container_height
            
            if container_ratio > image_ratio:
                height = container_height
                width = int(height * image_ratio)
            else:
                width = container_width
                height = int(width / image_ratio)
                
            x_offset = (container_width - width) // 2
            y_offset = (container_height - height) // 2
            
            # Transform event coordinates to image space
            x = int((event.x - x_offset) * (self.frame.shape[1] / width))
            y = int((event.y - y_offset) * (self.frame.shape[0] / height))
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.frame.shape[1] - 1))
            y = max(0, min(y, self.frame.shape[0] - 1))
            
            point_num = self.active_point
            
            # Update stored coordinates
            self.points[f'x{point_num}'] = x
            self.points[f'y{point_num}'] = y
            
            # Transform back to display coordinates
            display_x = int(x * (width / self.frame.shape[1])) + x_offset
            display_y = int(y * (height / self.frame.shape[0])) + y_offset
            
            # Move the point
            self.canvas.coords(
                self.point_markers[point_num],
                display_x-6, display_y-6, display_x+6, display_y+6
            )
            
            # Update connected lines
            if point_num == 0:  # Center point
                self.update_line(0, 1)
                if self.click_count > 3:
                    self.update_line(0, 3)
                if self.click_count > 4:
                    self.update_line(0, 4)
            elif point_num == 1:  # Head point
                self.update_line(0, 1)
            elif point_num == 3:  # Left wing
                self.update_line(0, 3)
            elif point_num == 4:  # Right wing
                self.update_line(0, 4)

    def on_release(self, event):
        """Handle mouse release events"""
        self.active_point = None
        self._drag_data = {"x": 0, "y": 0}
    
    def draw_line(self, point1_idx, point2_idx):
        """Draw line between two points"""
        # Get original coordinates
        x1 = self.points[f'x{point1_idx}']
        y1 = self.points[f'y{point1_idx}']
        x2 = self.points[f'x{point2_idx}']
        y2 = self.points[f'y{point2_idx}']
        
        # Transform coordinates for display
        container_width = self.main_container.winfo_width()
        container_height = self.main_container.winfo_height()
        
        image_ratio = self.frame.shape[1] / self.frame.shape[0]
        container_ratio = container_width / container_height
        
        if container_ratio > image_ratio:
            height = container_height
            width = int(height * image_ratio)
        else:
            width = container_width
            height = int(width / image_ratio)
            
        x_offset = (container_width - width) // 2
        y_offset = (container_height - height) // 2
        
        # Transform to display coordinates
        display_x1 = int(x1 * (width / self.frame.shape[1])) + x_offset
        display_y1 = int(y1 * (height / self.frame.shape[0])) + y_offset
        display_x2 = int(x2 * (width / self.frame.shape[1])) + x_offset
        display_y2 = int(y2 * (height / self.frame.shape[0])) + y_offset
        
        line = self.canvas.create_line(
            display_x1, display_y1, display_x2, display_y2,
            fill='#00ff00',
            width=2
        )
        self.lines.append(line)

    def update_line(self, point1_idx, point2_idx):
        """Update line position between two points"""
        # Get original coordinates
        x1 = self.points[f'x{point1_idx}']
        y1 = self.points[f'y{point1_idx}']
        x2 = self.points[f'x{point2_idx}']
        y2 = self.points[f'y{point2_idx}']
        
        # Transform coordinates for display
        container_width = self.main_container.winfo_width()
        container_height = self.main_container.winfo_height()
        
        image_ratio = self.frame.shape[1] / self.frame.shape[0]
        container_ratio = container_width / container_height
        
        if container_ratio > image_ratio:
            height = container_height
            width = int(height * image_ratio)
        else:
            width = container_width
            height = int(width / image_ratio)
            
        x_offset = (container_width - width) // 2
        y_offset = (container_height - height) // 2
        
        # Transform to display coordinates
        display_x1 = int(x1 * (width / self.frame.shape[1])) + x_offset
        display_y1 = int(y1 * (height / self.frame.shape[0])) + y_offset
        display_x2 = int(x2 * (width / self.frame.shape[1])) + x_offset
        display_y2 = int(y2 * (height / self.frame.shape[0])) + y_offset
        
        # Find the line index based on the points it connects
        line_idx = None
        if point1_idx == 0 and point2_idx == 1:
            line_idx = 0
        elif point1_idx == 0 and point2_idx == 3:
            line_idx = 1
        elif point1_idx == 0 and point2_idx == 4:
            line_idx = 2
            
        if line_idx is not None and line_idx < len(self.lines):
            self.canvas.coords(
                self.lines[line_idx],
                display_x1, display_y1, display_x2, display_y2
            )
    
    def start_experiment(self):
        """Modified to save metadata before closing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_experiment_metadata(timestamp)
        self.root.destroy()
        self.experiment_started = True
        self.timestamp = timestamp  # Store timestamp for retrieval

    def get_points(self):
        self.root.mainloop()
        if self.experiment_started:
            return self.points, self.timestamp
        return None, None

    def on_closing(self):
        self.experiment_started = False
        self.root.destroy()

    def next_frame(self):
        """Load the next frame from the video"""
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        self.frame = self._load_frame(self.frames[self.current_frame_idx])
        
        # Update canvas with new frame
        self.canvas.delete("all")  # Clear canvas
        self.update_canvas_image(self.canvas)
        
        # Redraw points and lines
        self._redraw_points_and_lines()

    def _redraw_points_and_lines(self):
        """Redraw all points and lines on the canvas"""
        # Get current image dimensions and position
        container_width = self.main_container.winfo_width()
        container_height = self.main_container.winfo_height()
        
        image_ratio = self.frame.shape[1] / self.frame.shape[0]
        container_ratio = container_width / container_height
        
        if container_ratio > image_ratio:
            height = container_height
            width = int(height * image_ratio)
        else:
            width = container_width
            height = int(width / image_ratio)
            
        x_offset = (container_width - width) // 2
        y_offset = (container_height - height) // 2
        
        # Clear existing points and lines
        for point in self.point_markers:
            self.canvas.delete(point)
        for line in self.lines:
            self.canvas.delete(line)
        
        self.point_markers.clear()
        self.lines.clear()
        
        # Redraw points
        for i in range(self.click_count):
            x = self.points[f'x{i}']
            y = self.points[f'y{i}']
            
            # Transform coordinates for display
            display_x = int(x * (width / self.frame.shape[1])) + x_offset
            display_y = int(y * (height / self.frame.shape[0])) + y_offset
            
            point = self.canvas.create_oval(
                display_x-6, display_y-6, display_x+6, display_y+6,
                fill='#00ff00',
                outline='white',
                width=2
            )
            self.point_markers.append(point)
        
        # Redraw lines
        if self.click_count > 1:
            self.draw_line(0, 1)
        if self.click_count > 3:
            self.draw_line(0, 3)
        if self.click_count > 4:
            self.draw_line(0, 4)

    def save_experiment_metadata(self, timestamp):
        """Save experiment metadata and notes to JSON file"""
        # Update config with new experiment name
        CONFIG['experiment']['name'] = self.name_entry.get()
        
        # Create metadata dictionary
        metadata = {
            "timestamp": timestamp,
            "experiment_name": self.name_entry.get(),
            "notes": self.notes_text.get("1.0", "end-1c"),
            "config": CONFIG,
            "points": {
                "body_center": {"x": self.points['x0'], "y": self.points['y0']},
                "head": {"x": self.points['x1'], "y": self.points['y1']},
                "wing_point": {"x": self.points['x2'], "y": self.points['y2']},
                "left_hinge": {"x": self.points['x3'], "y": self.points['y3']},
                "right_hinge": {"x": self.points['x4'], "y": self.points['y4']}
            }
        }
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save to JSON file
        json_path = output_dir / f"experiment_metadata_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved experiment metadata to: {json_path}")