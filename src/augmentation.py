import numpy as np
import matplotlib.pyplot as plt
from config import PARTICLE_COLORS



def apply_augmentations(image, mask, transformations):
    """Applique les transformations de manière synchronisée sur l'image et son masque."""
    transformed = transformations(image=image, mask=mask)
    augmented_image = transformed['image']
    augmented_mask = transformed['mask']
    
    # S'assurer que l'image et le masque ont la même taille que l'original
    if augmented_image.shape != image.shape:
        augmented_image = np.resize(augmented_image, image.shape)
    if augmented_mask.shape != mask.shape:
        augmented_mask = np.resize(augmented_mask, mask.shape)
    
    return augmented_image, augmented_mask


def visualize_augmented_samples(X_aug, Y_aug, all_targets, voxel_size, num_samples=5):
    """Affiche plusieurs exemples d'images augmentées avec leurs masques et protéines associées."""
    indices = np.random.choice(len(X_aug), num_samples, replace=False)
    colors = PARTICLE_COLORS
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 5))
    
    for i, idx in enumerate(indices):
        image = X_aug[idx].squeeze()
        mask = Y_aug[idx].argmax(axis=-1)  # Convertir one-hot en index de classe
        slice_idx = image.shape[0] // 2  # Slice médiane sur l'axe Z
        
        axes[i, 0].imshow(image[slice_idx, :, :], cmap='gray')
        axes[i, 0].set_title("Image Augmentée (Slice Z={})".format(slice_idx))
        
        # Ajouter les protéines avec des couleurs associées
        legend_entries = {}
        for molecule, points in all_targets.items():
            color = colors.get(molecule, "yellow")
            for point in points:
                if isinstance(point, dict) and "location" in point:
                    x = point["location"]["x"] / voxel_size
                    y = point["location"]["y"] / voxel_size
                    z = point["location"]["z"] / voxel_size
                    if abs(z - slice_idx) < 1:
                        sc = axes[i, 0].scatter(x, y, s=50, edgecolors=color, facecolors='none', linewidths=1.5)
                        if molecule not in legend_entries:
                            legend_entries[molecule] = sc
        
        if legend_entries:
            axes[i, 0].legend(legend_entries.values(), legend_entries.keys(), loc='upper right')
        
        axes[i, 1].imshow(mask[slice_idx, :, :], cmap='jet')
        axes[i, 1].set_title("Masque Associé (Slice Z={})".format(slice_idx))
    
    plt.tight_layout()
    plt.show()
