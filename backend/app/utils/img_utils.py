from PIL import Image
from torchvision import transforms
import torch
from pathlib import Path

def img_path_to_tensor(img_path, crop=True, scale=True):
    # open image
    image = Image.open(img_path)
    
    # Crop to a 1:1 aspect ratio
    if crop:
        width, height = image.size
        if width > height:
            margin = (width - height) // 2
            left, upper, right, lower = margin, 0, width - margin, height
        else:
            margin = (height - width) // 2
            left, upper, right, lower = 0, margin, width, height - margin
        image = image.crop((left, upper, right, lower))
        # ensure that image is square
        assert abs(image.size[0] - image.size[1]) <= 1

    # Resize the image to 224x224
    if scale:
        image = image.resize((224, 224), Image.LANCZOS)

    # convert to tensor (ToTensor scales to [0,1])
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image) * 255.0  # shape: (3, 224, 224)
    
    # subtract the VGG mean for each channel for normalization
    VGG_MEAN = torch.tensor([123.68, 116.779, 103.939]).reshape(3,1,1)
    tensor = tensor - VGG_MEAN

    # add a batch dimension: (1, 3, 224, 224)
    return tensor.unsqueeze(0)

# saves tensor as image in given path
def save_tensor_as_image(img_tensor, path):
    
    # Ensure the parent directory exists
    save_dir = Path(path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Make sure we are not tracking gradients and work on a CPU copy
    img_tensor = img_tensor.detach().cpu().clone()

    # If there's a batch dimension (N, C, H, W), remove it
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)

    # Add the VGG mean back
    img_tensor = img_tensor + torch.tensor([123.68, 116.779, 103.939]).reshape(3,1,1)

    # Clamp values to [0,255]
    img_tensor = torch.clamp(img_tensor, 0, 255)

    # Convert to uint8
    img_tensor = img_tensor.byte()

    # Convert from (C,H,W) to (H,W,C)
    img_array = img_tensor.permute(1, 2, 0).numpy()

    # Convert to PIL Image
    image = Image.fromarray(img_array)
    image.save(path)    
