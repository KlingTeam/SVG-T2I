import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from safetensors import safe_open

def get_img_to_safefile_map(files):
    """Create a mapping from image index to safetensor file and position"""
    img_to_file = {}
    for safe_file in files:
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            labels = f.get_slice('labels')
            labels_shape = labels.get_shape()
            num_imgs = labels_shape[0]
            cur_len = len(img_to_file)
            for i in range(num_imgs):
                img_to_file[cur_len+i] = {
                    'safe_file': safe_file,
                    'idx_in_file': i
                }
    return img_to_file

def get_latent_stats(data_dir):
    """Load latent statistics (mean and std) from cache file"""
    latent_stats_cache_file = os.path.join(data_dir, "latents_stats.pt")
    latent_stats = torch.load(latent_stats_cache_file)
    return latent_stats['mean'], latent_stats['std']

def load_latent_data(safetensor_files, cache_file=None, sample_num=10000):
    """Load latent vectors from safetensor files or cache if available"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading latent data from cache file: {cache_file}")
        return torch.load(cache_file)
    
    # Get directory from first file path
    data_dir = os.path.dirname(safetensor_files[0])
    
    # Load statistics
    latent_mean, latent_std = get_latent_stats(data_dir)
    latent_mean = latent_mean.squeeze(dim=(2,3))
    latent_std = latent_std.squeeze(dim=(2,3))
    
    # Sample latent vectors
    data = sample_latents(safetensor_files, latent_mean, latent_std, sample_num)
    
    # Save cache if needed
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(data, cache_file)
    
    return data

def calculate_uniformity_metrics(tsne_results):
    """Calculate uniformity metrics for t-SNE results"""
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)
    
    # Density statistics
    density_mean = np.mean(density)
    density_std = np.std(density)
    density_cv = density_std / density_mean
    
    # Entropy calculation
    density_norm = density / np.sum(density)
    entropy = -np.sum(density_norm * np.log2(density_norm + 1e-10))
    max_entropy = np.log2(len(density))
    normalized_entropy = entropy / max_entropy
    
    # Gini coefficient
    sorted_density = np.sort(density)
    index = np.arange(1, len(sorted_density) + 1)
    n = len(sorted_density)
    gini = ((np.sum((2 * index - n - 1) * sorted_density)) / 
            (n * np.sum(sorted_density)))
    
    return {
        'density_std': density_std,
        'density_cv': density_cv,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini
    }

def plot_tsne_visualization(safetensor_files, output_path="tsne_visualization.png", 
                           n_components=2, perplexity=30, n_iter=1000, cache_file=None):
    """Generate t-SNE visualization for latent vectors from safetensor files"""
    # Load data
    data = load_latent_data(safetensor_files, cache_file)

    # Print the shape of the data
    print(f"Data shape: {data.shape}")
    
    # Compute t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=n_components, random_state=42,
                perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data.numpy())
    
    # Calculate and print uniformity metrics
    metrics = calculate_uniformity_metrics(tsne_results)
    print(f"Uniformity metrics: {metrics}")

    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Calculate density for coloring
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)
    
    # Plot scatter with density coloring
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=density, cmap='viridis', alpha=0.6)
    
    # Clean up plot appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return tsne_results, metrics

def sample_latents(safetensor_files, latent_mean, latent_std, sample_num):
    """Sample latent vectors from safetensor files"""
    img_to_file_map = get_img_to_safefile_map(safetensor_files)
    total_imgs = len(img_to_file_map.keys())
    sample_idx = np.random.choice(total_imgs, sample_num, replace=False)
    
    data = []
    for idx in tqdm(sample_idx, desc="Sampling latent vectors"):
        img_info = img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            # Randomly choose between original and flipped latents for data augmentation
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            feature = features[img_idx:img_idx+1]
            
        # Sample a random pixel from the feature map
        h, w = feature.shape[2], feature.shape[3]
        hi = np.random.randint(0, h)
        wi = np.random.randint(0, w)
        pixel_feat = feature[:, :, hi, wi]
        
        # Normalize the feature
        pixel_feat = (pixel_feat - latent_mean) / latent_std
        data.append(pixel_feat)
    
    return torch.cat(data, dim=0)

# Usage example
if __name__ == "__main__":

    # Demo Usage:

    # We provide latent cache demos of ``f16d32`` and ``f16d32_vfdinov2``
    # They are randomly sampled VAE latents. You can also generate them by yourself.
    # https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants/tree/main
    # You can run the following code to visualize the t-SNE of the demo dataset.
    # ---------------------------------------------------
    # run:
    # cd LightningDiT
    # python tools/latent_vis.py
    # ---------------------------------------------------

    tsne_results, metrics = plot_tsne_visualization(
        [],
        output_path="tools/latent_demos/latent_tsne_f16d32.png",
        cache_file="tools/latent_demos/latents_cache_f16d32.pt"
    )

    tsne_results, metrics = plot_tsne_visualization(
        [],
        output_path="tools/latent_demos/latent_tsne_f16d32_vfdinov2.png",
        cache_file="tools/latent_demos/latents_cache_f16d32_vfdinov2.pt"
    )

    # Regular Usage:

    # Extracting latents with ``extract_features.py`` and get the safetensor files (including latents and mean, std)
    # Modify the ``latent_path_vf_dinov2`` to your own path
    # Run this script to get the t-SNE visualization
    # We randomly sample 10000 latents from the dataset and use t-SNE to visualize them. We run 10 times and take the average to 
    # reduce the randomness.

    # uncomment the following code to use your own dataset

    # latent_path = 'path/to/your/imagenet_train_256'
    
    # # Example usage with safetensor files
    # safetensor_files = sorted(glob(f"{latent_path}/*.safetensors"))
    
    # # Generate visualization
    # tsne_results, metrics = plot_tsne_visualization(
    #     safetensor_files,
    #     output_path="path/to/your/output.png",            # Output path
    #     cache_file="path/to/your/latents_cache.pt"        # Cache file
    # )
