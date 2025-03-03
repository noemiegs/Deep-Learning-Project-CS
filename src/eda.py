import matplotlib.pyplot as plt
import math


def calculate_global_counts(all_targets):
    """ Calcule le nombre global d'exemplaires pour chaque type de protéine. """
    global_counts = {}
    for molecules in all_targets.values():
        for molecule, points in molecules.items():
            global_counts[molecule] = global_counts.get(molecule, 0) + len(points)
    return global_counts


def plot_global_histogram(all_targets):
    """ Affiche l'histogramme global des protéines. """
    global_counts = calculate_global_counts(all_targets)
    plt.figure(figsize=(10, 6))
    plt.bar(global_counts.keys(), global_counts.values(), color='skyblue')
    plt.xlabel("Type de protéine")
    plt.ylabel("Nombre d'exemplaires")
    plt.title("Histogramme global du nombre d'exemplaires de protéines")
    plt.show()


def plot_tomogram_histograms(all_targets):
    """ Affiche les histogrammes détaillés par tomogramme. """
    num_tomograms = len(all_targets)
    cols = 2
    rows = math.ceil(num_tomograms / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 6, rows * 4), squeeze=False)
    tomogram_names = list(all_targets.keys())

    for idx, tomogram in enumerate(tomogram_names):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        counts = {molecule: len(points) for molecule, points in all_targets[tomogram].items()}

        ax.bar(counts.keys(), counts.values(), color='lightgreen')
        ax.set_title(tomogram)
        ax.set_xlabel("Protéine")
        ax.set_ylabel("Nombre d'exemplaires")
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)

    # Supprimer les sous-graphes vides
    total_plots = rows * cols
    for idx in range(num_tomograms, total_plots):
        row, col = divmod(idx, cols)
        fig.delaxes(axes[row][col])

    plt.suptitle("Histogrammes par tomogramme du nombre d'exemplaires de protéines", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()