import os
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_synthetic_dataset(root_dir="data/mlcc_synthetic", num_train=50, num_val=20):
    classes = ["good", "scratch", "crack", "discoloration", "chip_off"]
    
    for split, num in [("train", num_train), ("val", num_val)]:
        for cls_name in classes:
            os.makedirs(os.path.join(root_dir, split, cls_name), exist_ok=True)
            
            for i in range(num):
                # Create a "chip" (grey rectangle)
                img = Image.new('RGB', (64, 64), color=(200, 200, 200))
                draw = ImageDraw.Draw(img)
                
                # Add "features" based on class
                if cls_name == "scratch":
                    x1, y1 = random.randint(0, 64), random.randint(0, 64)
                    x2, y2 = random.randint(0, 64), random.randint(0, 64)
                    draw.line((x1, y1, x2, y2), fill=(50, 50, 50), width=1)
                elif cls_name == "crack":
                    x1, y1 = random.randint(10, 54), random.randint(10, 54)
                    draw.line((x1, y1, x1+10, y1+10), fill=(0, 0, 0), width=2)
                elif cls_name == "discoloration":
                    draw.rectangle((20, 20, 40, 40), fill=(180, 180, 150))
                elif cls_name == "chip_off":
                    draw.polygon([(0,0), (10,0), (0,10)], fill=(255, 255, 255))
                
                # Save
                img.save(os.path.join(root_dir, split, cls_name, f"{i}.jpg"))
                
    print(f"Generated synthetic dataset at {root_dir}")

if __name__ == "__main__":
    generate_synthetic_dataset()
