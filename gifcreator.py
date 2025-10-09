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
    imgs = ["plot1.png", "plot2.png", "plot3.png", "plot4.png"]
    create_gif(imgs, "output.gif", duration=300, loop=0)