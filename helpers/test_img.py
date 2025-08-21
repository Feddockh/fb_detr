from PIL import Image
import sys

def create_black_image(height, width, filename):
    # Create a new black image with the given dimensions.
    img = Image.new("RGB", (width, height), "black")
    img.save(filename, "PNG")
    print(f"Saved black image of size {height}x{width} as '{filename}'")

if __name__ == "__main__":
    # Usage: python test_img.py <height> <width> <filename>
    if len(sys.argv) == 4:
        try:
            height = int(sys.argv[1])
            width = int(sys.argv[2])
        except ValueError:
            print("Height and width must be integers.")
            sys.exit(1)
        filename = sys.argv[3]
    else:
        # Default values if no command line arguments are given.
        height, width = 100, 200
        filename = "black_image.png"

    create_black_image(height, width, filename)