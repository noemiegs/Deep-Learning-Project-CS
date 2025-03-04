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


def extract_cubic_patches(tomogram, resolution, mask, dim_in):
    """
    Extrait des patches cubiques de taille (dim_in, dim_in, dim_in) à partir du volume.
    Si la dimension finale dépasse la taille du volume, le patch est complété par du padding.

    Args:
        tomogram (dict): Dictionnaire contenant l'image et le masque.
        mask (ndarray): Masque associé.
        dim_in (int): Taille des patches (cubiques).

    Returns:
        tuple: (patches_img, patches_mask)
               - patches_img : liste de patches image.
               - patches_mask : liste de patches masque.
    """
    volume = tomogram['images'][int(resolution)]
    z_dim, y_dim, x_dim = volume.shape
    
    patches_img = []
    patches_mask = []

    # Calcul des indices de départ pour chaque axe
    z_steps = list(range(0, z_dim, dim_in))
    y_steps = list(range(0, y_dim, dim_in))
    x_steps = list(range(0, x_dim, dim_in))

    # Assurez-vous de couvrir toute la zone avec padding si nécessaire
    if z_dim % dim_in != 0:
        z_steps.append(z_dim - dim_in)
    if y_dim % dim_in != 0:
        y_steps.append(y_dim - dim_in)
    if x_dim % dim_in != 0:
        x_steps.append(x_dim - dim_in)

    # Extraction des patches
    for z0 in z_steps:
        for y0 in y_steps:
            for x0 in x_steps:
                patch_img = volume[z0:min(z0 + dim_in, z_dim),
                                   y0:min(y0 + dim_in, y_dim),
                                   x0:min(x0 + dim_in, x_dim)]
                
                patch_mask = mask[z0:min(z0 + dim_in, z_dim),
                                  y0:min(y0 + dim_in, y_dim),
                                  x0:min(x0 + dim_in, x_dim)]

                # Padding si la taille est inférieure à dim_in
                pad_z = dim_in - patch_img.shape[0]
                pad_y = dim_in - patch_img.shape[1]
                pad_x = dim_in - patch_img.shape[2]

                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    patch_img = np.pad(patch_img, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                    patch_mask = np.pad(patch_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

                patches_img.append(patch_img)
                patches_mask.append(patch_mask)

    return patches_img, patches_mask


def generate_training_data(dataset, resolution='2', dim_in=64):
    """
    Génère les données d'entraînement avec des patches cubiques.

    Args:
        dataset (list): Liste des tomogrammes.
        resolution (str): Résolution du tomogramme.
        dim_in (int): Taille des patches cubiques.

    Returns:
        X_train (numpy array): Les images des patches.
        Y_train (numpy array): Les masques des patches.
    """
    X_train_list = []
    Y_train_list = []

    for tomogram in dataset:
        mask = generate_mask(tomogram, resolution, sphere_radius=2)
        patches_img, patches_mask = extract_cubic_patches(tomogram, mask, dim_in=dim_in)

        for img, mask in zip(patches_img, patches_mask):
            img = img[..., np.newaxis].astype(np.float32)
            X_train_list.append(img)
            Y_train_list.append(mask)

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