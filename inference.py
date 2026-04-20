import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from model import LiteSeg

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(input_dir, output_dir):

    model = LiteSeg(num_classes=21).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    transform = T.Compose([
        T.Resize((300,300)),
        T.ToTensor()
    ])

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):

        img_path = os.path.join(input_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)

        # Binary mask (IMPORTANT for evaluation)
        binary_mask = (pred > 0).int().squeeze().cpu().numpy()

        save_name = img_name.replace(".jpg", "_mask.png")
        save_path = os.path.join(output_dir, save_name)

        Image.fromarray(binary_mask.astype("uint8") * 255).save(save_path)

    print("Inference completed!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)