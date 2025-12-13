import os
import glob
import torch
import importlib
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# =========================
# 1. Load configuration
# =========================
ckpt = "pre-trained/autoencoder/svg_autoencoder_P_1024.yaml"

encoder_config = OmegaConf.load(ckpt)
epoch = encoder_config.model.params.ckpt_path.split("/")[-1].split('.')[0]
encoder_config.model.params.is_train = False
dinov3 = instantiate_from_config(encoder_config.model).cuda().eval()


# =========================
# 2. Image path & output directory
# =========================
for resol in [1024, 256, 512]:
    image_dir = "./test"
    basename = ckpt.split('/')[-1].split('.')[0]
    out_dir = f"./output/output-{basename}-{epoch}-{resol}"
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 3. Image preprocessing & resize helper
    # =========================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            inplace=True)
    ])

    def resize_to_even_half(img: Image.Image) -> Image.Image:
        """Resize image to target resolution (resol x resol)."""
        w_half, h_half = resol, resol
        print(w_half, h_half)
        return img.resize((w_half, h_half), Image.BICUBIC)

    def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization for a tensor of shape (C,H,W) or (B,C,H,W)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, -1, 1, 1)
        return tensor * std + mean

    # =========================
    # 4. Load all image paths
    # =========================
    image_paths = (
        glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True) +
        glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(image_dir, "**", "*.JPEG"), recursive=True)
    )
    print(f"Found {len(image_paths)} images")

    # =========================
    # 5. Batch processing
    # =========================
    batch_size = 16  # Adjust based on GPU memory
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images", ncols=100):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []

        # -------------------------
        # 5.1 Load & preprocess images
        # -------------------------
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = resize_to_even_half(img)
                img_tensor = transform(img).unsqueeze(0).to(device)
                batch_imgs.append(img_tensor)
            except Exception as e:
                print(f"[Warning] Failed to process image {img_path}: {e}")
                continue

        if not batch_imgs:
            continue

        batch_list = batch_imgs  # shape: [B, C, H, W]

        # -------------------------
        # 5.2 Encode
        # -------------------------
        imgs_encoded = []
        for j in range(len(batch_list)):
            try:
                print(batch_list[j].shape)
                latent = dinov3.encode(batch_list[j])
                imgs_encoded.append(latent)
            except Exception as e:
                print(f"[Warning] Failed to encode {batch_paths[j]}: {e}")
                continue

        if not imgs_encoded:
            continue

        # -------------------------
        # 5.3 Decode
        # -------------------------
        try:
            with torch.no_grad():
                decoded_list = dinov3.decode(torch.cat(imgs_encoded, dim=0))
                print(decoded_list.shape)
        except Exception as e:
            print(f"[Warning] Failed to decode: {e}")
            continue

        # -------------------------
        # 5.4 Save results
        # -------------------------
        for j, decoded in enumerate(decoded_list):
            try:
                decoded = torch.clamp(decoded, -1, 1)
                basename = os.path.basename(batch_paths[j]).split('.')[0]

                # Save reconstructed image (decode output assumed to be [-1, 1])
                save_image(
                    decoded,
                    os.path.join(out_dir, f"{basename}_rec.png"),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )

                # Save original image (unnormalized)
                orig_img = unnormalize(batch_imgs[j].clone())
                orig_img = torch.clamp(orig_img, 0, 1)
                save_image(orig_img, os.path.join(out_dir, f"{basename}.png"), nrow=1)

            except Exception as e:
                print(f"[Warning] Failed to save image {batch_paths[j]}: {e}")
                continue

        # -------------------------
        # 5.5 Clear GPU memory
        # -------------------------
        del batch_imgs, batch_list, imgs_encoded, decoded_list
        torch.cuda.empty_cache()
