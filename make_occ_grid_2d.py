import tkinter as tk
from tkinter import Button, Canvas, filedialog
import numpy as np
from PIL import Image, ImageTk

class ImageColoringApp:
    def __init__(self, root, image_width=640, image_height=640):
        self.root = root
        self.image_width = image_width
        self.image_height = image_height
        self.image = Image.new("RGB", (self.image_width, self.image_height), "black")  # Start with a blank white image
        self.pixels = np.array(self.image)  # Convert to NumPy array to keep track of changes
        
        # Canvas to display the image
        self.canvas = Canvas(root, width=self.image_width, height=self.image_height, bg="white")
        self.canvas.pack()
        
        # Convert image to a format that can be displayed on Tkinter canvas
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Bind left mouse click to draw on the canvas
        self.canvas.bind("<Button-1>", self.on_click)  # Left click to draw
        self.canvas.bind("<B1-Motion>", self.on_drag)  # Click and drag to draw
        
        # Buttons to change color, clear the image, and save the file
        self.color = (255, 0, 0)  # Default color is red
        
        self.red_button = Button(root, text="Red", bg="red", command=lambda: self.set_color((255, 0, 0)))
        self.red_button.pack(side="left")
        
        self.green_button = Button(root, text="Green", bg="green", command=lambda: self.set_color((0, 255, 0)))
        self.green_button.pack(side="left")
        
        self.blue_button = Button(root, text="Blue", bg="blue", command=lambda: self.set_color((0, 0, 255)))
        self.blue_button.pack(side="left")
        
        self.blue_button = Button(root, text="White", bg="White", command=lambda: self.set_color((255, 255, 255)))
        self.blue_button.pack(side="left")
        
        self.clear_button = Button(root, text="Clear", command=self.clear_image)
        self.clear_button.pack(side="left")
        
        self.save_button = Button(root, text="Save", command=self.save_as_numpy_array)
        self.save_button.pack(side="left")

    def set_color(self, color):
        """Sets the current drawing color."""
        self.color = color
    
    def on_click(self, event):
        """Draw a point where the user clicks."""
        self.draw_point(event.x, event.y)
    
    def on_drag(self, event):
        """Draw continuously as the user drags the mouse."""
        self.draw_point(event.x, event.y)
    
    def draw_point(self, x, y):
        """Draws a small square at (x, y) to simulate a brush stroke."""
        brush_size = 5  # Size of the brush
        x0 = max(0, x - brush_size)
        y0 = max(0, y - brush_size)
        x1 = min(self.image_width, x + brush_size)
        y1 = min(self.image_height, y + brush_size)
        
        # Draw on the image
        for i in range(y0, y1):
            for j in range(x0, x1):
                self.pixels[i, j] = self.color
        
        # Update the canvas image
        self.update_image()
    
    def update_image(self):
        """Updates the image on the canvas."""
        self.image = Image.fromarray(self.pixels)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
    
    def clear_image(self):
        self.pixels = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.update_image()
    
    def save_as_numpy_array(self):
        """Saves the image as a .npy file."""
        file_path = 'occ_grid.npy'
        if file_path:
            img = self.pixels[::-1, :, :]
            np.save(file_path, img)
            print(f"Image saved as NumPy array at: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageColoringApp(root)
    root.mainloop()
