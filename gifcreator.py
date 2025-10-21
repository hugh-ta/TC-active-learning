from PIL import Image

def create_gif(image_files, output_path, duration=500, loop=0):
    # Open images
    images = [Image.open(img) for img in image_files]
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )

# Example usage
if __name__ == "__main__":
    # Make sure all files exist and have correct extensions
    imgs = ["1.png", "2.png", "3.png", "4.png", "5.png"]
    create_gif(imgs, "output69.gif", duration=300, loop=0)