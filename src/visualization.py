import matplotlib.pyplot as plt
import numpy as np
from config import VOXEL_SIZES, PARTICLE_COLORS

def display_tomogram_slice(all_data, tomogram_folder, slice_index, all_targets=None, resolution=None, threshold=30):
    """
    Affiche une ou plusieurs coupes d'un tomogramme avec superposition optionnelle des cibles moléculaires.

    Args:
        all_data (dict): Dictionnaire contenant les données des tomogrammes.
        tomogram_folder (str): Nom du dossier du tomogramme.
        slice_index (int): Index de la coupe sur l'axe Z.
        all_targets (dict, optional): Dictionnaire contenant les cibles moléculaires. Si None, aucune cible n'est affichée.
        resolution (str, optional): Résolution à afficher. Si None, affiche les trois résolutions.
        threshold (int, optional): Distance maximale entre la coupe et la cible pour l'affichage. Par défaut 30 voxels.

    Returns:
        None
    """
    # Si la résolution est None, on affiche les trois résolutions
    if resolution is None:
        resolutions = ['0', '1', '2']
    else:
        resolutions = [resolution]

    fig, axs = plt.subplots(1, len(resolutions), figsize=(6 * len(resolutions), 6))
    if len(resolutions) == 1:
        axs = [axs]

    legend_entries = {}
    for i, res in enumerate(resolutions):
        if tomogram_folder not in all_data or res not in all_data[tomogram_folder]:
            print(f"Données introuvables pour {tomogram_folder} avec la résolution {res}.")
            continue

        tomogram_volume = all_data[tomogram_folder][res]['data']
        image_slice = tomogram_volume[slice_index, :, :]

        axs[i].imshow(image_slice, cmap='gray')
        axs[i].set_title(f"Résolution {res} - {tomogram_volume.shape[2]}x{tomogram_volume.shape[1]}x{tomogram_volume.shape[0]}")
        axs[i].axis('off')

        voxel_size = VOXEL_SIZES.get(res)
        if voxel_size is None:
            print(f"⚠️ Avertissement : Voxel size non défini pour la résolution {res}. Les cibles peuvent être mal positionnées.")
            continue

        if all_targets and tomogram_folder in all_targets:
            molecules = all_targets[tomogram_folder]
            for molecule, points in molecules.items():
                color = PARTICLE_COLORS.get(molecule, 'red')
                for point in points:
                    x = point["location"]["x"] / voxel_size
                    y = point["location"]["y"] / voxel_size
                    z = point["location"]["z"] / voxel_size
                    if abs(z - slice_index) < (threshold / voxel_size):
                        sc = axs[i].scatter(x, y, s=20, color=color, label=molecule if molecule not in legend_entries else "")
                        if molecule not in legend_entries:
                            legend_entries[molecule] = sc

    if legend_entries:
        fig.legend(legend_entries.values(), legend_entries.keys(), loc='upper right')

    plt.suptitle(f"Tomogramme {tomogram_folder} - Coupe Z={slice_index}")
    plt.show()
