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
    imgs = ["Figure_1gif.png", "Figure_2gif.png", "Figure_3gif.png", "Figure_4gif.png", "Figure_5gif.png"]
    create_gif(imgs, "output.gif", duration=300, loop=0)