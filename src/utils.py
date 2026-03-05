from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def load_image(image_path, target_size = None):
    image = Image.open(image_path).convert('RGB')

    # Convert to tensor (3, H, W) with values in [0, 1]
    image_tensor = TF.to_tensor(image)
    
    if target_size is not None:
        # Resize to (3, height, width)
        image_tensor = TF.resize(image_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    
    return image_tensor

def imshow(image, target, predicted):
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(predicted, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    plt.show()