import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import logging
import tkinter as tk

logger = logging.getLogger(__name__)

class PointSelector:
    def __init__(self, frames):
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.root = ctk.CTk()
        self.root.title("Wing Analysis Setup")
        
        # Use just the first frame instead of averaging
        self.frame = frames[0]
        
        # Convert to PIL format for tkinter
        self.image = Image.fromarray(self.frame)
        self.photo = ImageTk.PhotoImage(self.image)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=self.frame.shape[1],
            height=self.frame.shape[0],
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        
        # Display the averaged frame
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        # Create rounded container frame
        self.container = ctk.CTkFrame(
            self.main_frame,
            corner_radius=10,
            width=400,
            height=100
        )
        self.container.place(
            relx=0.5,
            rely=1.0,
            anchor="s",
            y=-20
        )
        
        # Prevent frame from shrinking
        self.container.grid_propagate(False)
        self.container.pack_propagate(False)
        
        # Instructions label in container
        self.instruction_text = ctk.StringVar()
        self.instruction_label = ctk.CTkLabel(
            self.container,
            textvariable=self.instruction_text,
            font=("Helvetica", 12)
        )
        self.instruction_label.pack(pady=(10, 5))
        
        # Start button
        self.start_button = ctk.CTkButton(
            self.container,
            text="Start Experiment",
            command=self.start_experiment,
            state="disabled",
            font=("Helvetica", 12),
            corner_radius=8
        )
        self.start_button.pack(pady=(0, 10))
        
        # Initialize points and lines
        self.points = {'x0': None, 'y0': None, 'x1': None, 'y1': None,
                      'x2': None, 'y2': None, 'x3': None, 'y3': None,
                      'x4': None, 'y4': None}
        self.point_markers = []
        self.lines = []
        self.click_count = 0
        self.active_point = None
        self._drag_data = {"x": 0, "y": 0}  # Store initial click position
        
        # Bind events in correct order
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
        self.update_instructions()
        
        # Flag for experiment start
        self.experiment_started = False
        
        # Center the window
        self.root.update()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Add window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
        x, y = event.x, event.y
        
        # Store the initial click position
        self._drag_data["x"] = x
        self._drag_data["y"] = y
        
        # Check for existing point click with improved hit detection
        for i, point in enumerate(self.point_markers):
            coords = self.canvas.coords(point)
            if coords:
                # Calculate center of point
                px = (coords[0] + coords[2]) / 2
                py = (coords[1] + coords[3]) / 2
                # Check if click is within point area
                if (x - px) ** 2 + (y - py) ** 2 < 400:  # 20px radius
                    self.active_point = i
                    return
        
        # Create new point if we haven't placed all points
        if self.click_count < 5:
            point_num = self.click_count
            self.points[f'x{point_num}'] = x
            self.points[f'y{point_num}'] = y
            
            point = self.canvas.create_oval(
                x-6, y-6, x+6, y+6,
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
            # Calculate how far the mouse has moved
            x, y = event.x, event.y
            point_num = self.active_point
            
            # Update stored coordinates
            self.points[f'x{point_num}'] = x
            self.points[f'y{point_num}'] = y
            
            # Move the point
            self.canvas.coords(
                self.point_markers[point_num],
                x-6, y-6, x+6, y+6
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
            
            # Update the drag data
            self._drag_data["x"] = x
            self._drag_data["y"] = y

    def on_release(self, event):
        """Handle mouse release events"""
        self.active_point = None
        self._drag_data = {"x": 0, "y": 0}
    
    def draw_line(self, point1_idx, point2_idx):
        x1 = self.points[f'x{point1_idx}']
        y1 = self.points[f'y{point1_idx}']
        x2 = self.points[f'x{point2_idx}']
        y2 = self.points[f'y{point2_idx}']
        
        line = self.canvas.create_line(
            x1, y1, x2, y2,
            fill='#00ff00',  # Bright green
            width=2
        )
        self.lines.append(line)
        
    def update_line(self, point1_idx, point2_idx):
        """Update line position between two points"""
        x1 = self.points[f'x{point1_idx}']
        y1 = self.points[f'y{point1_idx}']
        x2 = self.points[f'x{point2_idx}']
        y2 = self.points[f'y{point2_idx}']
        
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
                x1, y1, x2, y2
            )
    
    def start_experiment(self):
        self.root.destroy()
        self.experiment_started = True
        
    def get_points(self):
        self.root.mainloop()
        return self.points if self.experiment_started else None 
    
    def on_closing(self):
        self.experiment_started = False
        self.root.destroy()