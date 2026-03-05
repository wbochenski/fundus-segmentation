import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F

from ..utils import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None,
    in_channels=3, 
    classes=2
)

weights_path = "models/gabriel_lepetitaimon_fundus_vessel.pt"
model.load_state_dict(torch.load(weights_path))
model.eval()
model.to(device)

def predict(image_path: str):
    x = load_image(image_path).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    vessel_map = probs[0, 1].cpu().numpy()
    return vessel_map
