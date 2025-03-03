import os
import json
import zarr
import numpy as np
from config import VOXEL_SIZES, PARTICLE_COLORS, CLASS_MAPPING
from tensorflow.keras.utils import to_categorical


def prepare_dataset(image_path, target_path):
    """
    Prépare un dataset associant les données d'images aux targets (protéines et positions).
    
    Args:
        image_path (str): Chemin vers le dossier contenant les images Zarr.
        target_path (str): Chemin vers le dossier contenant les targets (fichiers JSON).

    Returns:
        list: Liste de dictionnaires, où chaque élément contient les données d'un tomogramme :
            - "name": Nom du tomogramme.
            - "images": Liste des résolutions (volumes 3D).
            - "targets": Dictionnaire {type_molécule: [positions (x, y, z)]}.
    """
    dataset = []

    # Parcourir les tomogrammes dans le dossier des images
    for tomogram_name in os.listdir(image_path):
        tomogram_image_path = os.path.join(image_path, tomogram_name, "VoxelSpacing10.000/denoised.zarr")
        tomogram_target_path = os.path.join(target_path, tomogram_name, "Picks")

        # Vérifier que les données Zarr et les targets existent
        if not os.path.exists(tomogram_image_path):
            print(f"Images non trouvées pour {tomogram_name}, ignoré.")
            continue

        if not os.path.exists(tomogram_target_path):
            print(f"Targets non trouvés pour {tomogram_name}, ignoré.")
            continue

        # Charger les images (volumes 3D à plusieurs résolutions)
        zgroup = zarr.open_group(tomogram_image_path, mode='r')
        sorted_keys = sorted(zgroup.keys(), key=lambda k: np.prod(zgroup[k].shape), reverse=True)
        images = [zgroup[key][:] for key in sorted_keys]

        # Charger les targets (localisations des particules)
        targets = {}
        for json_file in os.listdir(tomogram_target_path):
            if json_file.endswith(".json"):
                json_path = os.path.join(tomogram_target_path, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    molecule_name = data.get("pickable_object_name", "unknown")
                    points = [
                        [point["location"]["x"], point["location"]["y"], point["location"]["z"]]
                        for point in data["points"]
                    ]
                    if molecule_name not in targets:
                        targets[molecule_name] = []
                    targets[molecule_name].extend(points)

        # Ajouter les données du tomogramme au dataset
        dataset.append({
            "name": tomogram_name,
            "images": images,  # Liste des résolutions
            "targets": targets  # Localisations des particules par type
        })

    return dataset

# print tomogram shapes for each resolution in the order of the dataset
def print_shapes(dataset):
    for tomogram in dataset:
        print(f"Tomogramme {tomogram['name']}:")
        for i, image in enumerate(tomogram['images']):
            print(f"  - Résolution {i}: {image.shape}")


def generate_mask(tomogram, resolution, sphere_radius=2):
    """
    Crée un masque de segmentation (de dimensions volume_shape) à partir des targets.
    Pour chaque point, on convertit la position physique en indice voxel et on dessine
    une petite sphère (de rayon sphere_radius voxels) avec la classe correspondante.
    
    Args:
        tomogram (dict): Dictionnaire contenant les données du tomogramme.
        resolution (str): Résolution à utiliser ('0', '1' ou '2').
        sphere_radius (int): Rayon de la sphère autour de chaque point.
        
    Returns:
        ndarray: Masque de segmentation de dimensions volume_shape.
    """
    class_mapping = CLASS_MAPPING
    volume_shape = tomogram['images'][int(resolution)].shape
    targets = tomogram['targets']
    voxel_size = VOXEL_SIZES.get(resolution)
    mask = np.zeros(volume_shape, dtype=np.uint8)  # Fond = 0

    for molecule, points in targets.items():
        if molecule not in class_mapping:
            continue
        class_idx = class_mapping[molecule]
        for point in points:
            # Si le point est un dictionnaire avec la clé "location", on l'utilise
            if isinstance(point, dict) and "location" in point:
                x_coord = point["location"]["x"]
                y_coord = point["location"]["y"]
                z_coord = point["location"]["z"]
            else:
                # Sinon, on suppose que le point est une liste/tuple de coordonnées [x, y, z]
                x_coord, y_coord, z_coord = point
            # print(f"Point {molecule} at ({x_coord}, {y_coord}, {z_coord})")
            x_center = int(round(x_coord / voxel_size))
            y_center = int(round(y_coord / voxel_size))
            z_center = int(round(z_coord / voxel_size))
            
            # Dessiner une petite sphère autour du point
            for dz in range(-sphere_radius, sphere_radius + 1):
                for dy in range(-sphere_radius, sphere_radius + 1):
                    for dx in range(-sphere_radius, sphere_radius + 1):
                        if dx**2 + dy**2 + dz**2 <= sphere_radius**2:
                            z_idx = z_center + dz
                            y_idx = y_center + dy
                            x_idx = x_center + dx
                            if (0 <= z_idx < volume_shape[0] and
                                0 <= y_idx < volume_shape[1] and
                                0 <= x_idx < volume_shape[2]):
                                mask[z_idx, y_idx, x_idx] = class_idx
    return mask


def extract_grid_patches(tomogram, mask, resolution, num_cubes_axis=None):
    """
    Extrait tous les patches qui couvrent entièrement le volume en divisant chaque dimension en num_cubes_axis segments.
    On suppose que le volume est exactement divisible par num_cubes_axis sur chaque axe.
    
    Args:
        volume (ndarray): Volume 3D d'entrée, de forme (Z, Y, X).
        mask (ndarray): Masque associé, de même forme.
        num_cubes_axis (int): Nombre de segments (patchs) par axe (exemple : 4 donnera 4x4x4 = 64 patches).
        
    Returns:
        tuple: (patches_img, patches_mask) 
               - patches_img : liste de patches d'image.
               - patches_mask : liste de patches de masque.
    """
    if num_cubes_axis is None:
        # based on the resolution, we can set the number of cubes
        num_resolutions = len(tomogram['images'])
        num_cubes_axis = 2**int(num_resolutions-resolution)
    volume = tomogram['images'][int(resolution)]
    z_dim, y_dim, x_dim = volume.shape
    patch_size_z = z_dim // num_cubes_axis
    patch_size_y = y_dim // num_cubes_axis
    patch_size_x = x_dim // num_cubes_axis
    
    patches_img = []
    patches_mask = []
    
    for i in range(num_cubes_axis):
        for j in range(num_cubes_axis):
            for k in range(num_cubes_axis):
                z0 = i * patch_size_z
                y0 = j * patch_size_y
                x0 = k * patch_size_x
                patch_img = volume[z0:z0+patch_size_z, y0:y0+patch_size_y, x0:x0+patch_size_x]
                patch_mask = mask[z0:z0+patch_size_z, y0:y0+patch_size_y, x0:x0+patch_size_x]
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
    
    return patches_img, patches_mask

def generate_training_data(dataset, resolution='2', num_cubes_axis=8):
    """
    Génère les données d'entraînement à partir du dataset en divisant chaque tomogramme en patchs.
    
    Args:
        dataset (list): Liste des tomogrammes.
        resolution (str): Résolution du tomogramme.
        num_cubes_axis (int): Nombre de cubes par axe (détermine la taille des patchs).
        
    Returns:
        X_train (numpy array): Les images des patchs.
        Y_train_int (numpy array): Les masques des patchs.
    """
    X_train_list = []
    Y_train_list = []
    
    for tomogram in dataset:
        # print(f"Traitement de {tomogram['name']} - Cibles : {tomogram['targets']}")
        mask_full = generate_mask(tomogram, resolution, sphere_radius=2)
        # print(f"Volume: {tomogram['images'][int(resolution)].shape} - Masque: {mask_full.shape}")
        
        patches_img, patches_mask = extract_grid_patches(tomogram, mask_full, resolution, num_cubes_axis=num_cubes_axis)
        
        for patch_img, patch_mask in zip(patches_img, patches_mask):
            patch_img = patch_img[..., np.newaxis]  # Ajouter la dimension de canal
            X_train_list.append(patch_img)
            Y_train_list.append(patch_mask)

    X_train = np.array(X_train_list, dtype=np.float32)
    Y_train_int = np.array(Y_train_list, dtype=np.uint8)
    Y_train = to_categorical(Y_train_int, num_classes=len(CLASS_MAPPING))
    
    return X_train, Y_train, Y_train_int

def balance_dataset(X_train, Y_train_int, num_classes=7):
    """
    Équilibre le dataset en sélectionnant aléatoirement les patchs sans protéines pour avoir un équilibre 50/50.
    
    Args:
        X_train (numpy array): Les images des patchs.
        Y_train_int (numpy array): Les masques des patchs.
        num_classes (int): Nombre de classes dans les masques.
        
    Returns:
        X_train_balanced (numpy array): Les images des patchs équilibrés.
        Y_train_balanced (numpy array): Les masques des patchs équilibrés en one-hot.
    """
    patch_contains_protein = (Y_train_int > 0).any(axis=(1, 2, 3))
    
    indices_with_protein = np.where(patch_contains_protein)[0]
    indices_without_protein = np.where(~patch_contains_protein)[0]
    
    num_with_protein = len(indices_with_protein)
    num_without_protein_needed = num_with_protein  # Pour équilibrer à 50/50
    
    indices_without_protein_sampled = np.random.choice(indices_without_protein, num_without_protein_needed)
    balanced_indices = np.concatenate([indices_with_protein, indices_without_protein_sampled])
    
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = X_train[balanced_indices]
    Y_train_int_balanced = Y_train_int[balanced_indices]
    
    Y_train_balanced = to_categorical(Y_train_int_balanced, num_classes=num_classes)
    
    return X_train_balanced, Y_train_balanced, Y_train_int_balanced