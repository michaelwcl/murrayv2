import numpy as np
from PIL import Image, ImageDraw
import os

def visualize_lake(lake_array, agent_pos, epoch, save_dir="visuals"):
    """
    Create a PNG visualization of the lake state
    0 (unexplored) = white
    1 (land) = black
    2 (explored) = blue
    agent position = red dot
    Note: Image is flipped vertically to match coordinate system
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create RGB image
    height, width = lake_array.shape
    img = Image.new('RGB', (width, height), color='black')
    pixels = img.load()
    
    # Fill pixels based on lake state (note y is flipped)
    for y in range(height):
        for x in range(width):
            # Flip y-coordinate for visualization
            flipped_y = height - 1 - y
            if lake_array[y, x] == 0:  # Land
                pixels[x, flipped_y] = (255, 255, 255)  # White
            elif lake_array[y, x] == 1:  # Unexplored water
                pixels[x, flipped_y] = (0, 0, 0)  # Black
            elif lake_array[y, x] == 2:  # Explored water
                pixels[x, flipped_y] = (0, 0, 255)  # Blue
    
    # Mark agent position with a red dot (remember to flip y)
    draw = ImageDraw.Draw(img)
    agent_x, agent_y = agent_pos
    flipped_agent_y = height - 1 - agent_y
    dot_size = 2
    draw.ellipse([agent_x - dot_size, flipped_agent_y - dot_size, 
                  agent_x + dot_size, flipped_agent_y + dot_size], 
                 fill='red')
    
    # Save image in visuals folder
    filename = f"lake_epoch{epoch}.png"
    filepath = os.path.join(save_dir, filename)
    img.save(filepath)
    return filepath