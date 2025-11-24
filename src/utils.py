import json
import roller_design as RD
import numpy as np
import matplotlib.pyplot as plt
import optim_rollerbearing as opt
from scipy.spatial import KDTree
import os

def make_rollerbearing(x, rext=30.0, rshaft=6.5, L=15.0):
    y_ctrl = x[1:]
    rmid = x[0]
    rol = RD.Roller()
    rol.mid_width = L /2
    rol.rmid = rmid
    y_ctrl_final = np.clip(rmid - y_ctrl, rmid - 10, rmid - 2.5)
    rol.set_Rxf(y_ctrl_final)
    oring = RD.OuterRing(rol)
    iring = RD.InnerRing(rol)
    system = RD.BearingSimulation(rol, iring, oring, rext=rext, rshaft=rshaft)
    return system

def save_result_to_json(path, res, **kwargs):
    # Extraire les données utiles de la solution
    data = {
        "X": res.X.tolist(),  # variables
        "F": res.F.tolist(),  # objectifs
        "CV": res.CV.tolist(),  # violations des contraintes
    }
    for key, value in kwargs.items():
        data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Résultat sauvegardé dans {path}")

def load_result_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Reconstruire les tableaux NumPy
    X = np.array(data["X"])
    F = np.array(data["F"])
    CV = np.array(data["CV"])
    print(f"✅ Résultat chargé depuis {path}")
    return X, F, CV

def getbyrank(path, rank, w=None, e=None, rescaled=False, **kwargs):
    """Rank from 0 to len(F)"""
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])

    # Somme des carrés par ligne
    scores = weighted_scores(F, w=w, e=e, rescaled=rescaled)
    ranks = np.argsort(scores)  # ordre croissant
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(ranks))
    idx = int(np.where(rank_values == rank)[0])

    return make_rollerbearing(X[idx], **kwargs)

def select_best(path, w=None, e=None, rescaled=False, **kwargs):
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])

    # Calculate scores using the helper function
    scores = weighted_scores(F, w=w, e=e, rescaled=rescaled)

    # 1. Create a mask for the constraints
    # We select rows where the 2nd objective (index 1) and 3rd objective (index 2) are < 0.8
    valid_mask = (F[:, 1] < 0.8) & (F[:, 2] < 0.8)

    # 2. Handle cases where no solution meets the criteria
    if not np.any(valid_mask):
        print("Warning: No design satisfies the constraints. Returning the best unconstrained score.")
        best_idx = np.argmin(scores)
    else:
        # 3. Apply the mask to find the best among the valid designs
        # We copy scores to avoid modifying the original array
        masked_scores = scores.copy()

        # Set the score of invalid designs to infinity so they are not selected by argmin
        masked_scores[~valid_mask] = np.inf

        # Find the index of the lowest score among the valid entries
        best_idx = np.argmin(masked_scores)

    # Return the generated object for the best design
    return make_rollerbearing(X[best_idx], **kwargs)

def getbyindex(path, idx):
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    return make_rollerbearing(X[idx])

def weighted_scores(F, w=None, e=None, rescaled=False):
    if w is None:
        w = [1.0, 1.0, 1.0, 0.1]
    if e is None:
        e = [1.0, 1.0, 1.0, 1.0]
    weights = np.array(w)
    exponents = np.array(e)
    if F[0].shape != weights.shape or F[0].shape != exponents.shape:
        raise ValueError("Vector and weights and exponents must have the same shape.")

    if rescaled:
        # Here the goal is to estimate the lifetime
        F = (1 / F) - 1
        weighted_vector = weights * np.power(F, exponents)
        L = np.min(weighted_vector, axis=1, where=(weighted_vector > 0), initial=np.inf)
        return 1 / L

    # Apply weights element-wise
    weighted_vector = weights * np.power(F, exponents)

    # Calculate the standard L2 norm of the weighted vector
    return np.linalg.norm(weighted_vector, axis=1)

def save_all_bearing_plots(data_path: str, output_dir: str):
    """
    Génère et enregistre les plots de tous les roulements optimisés
    dans un dossier spécifié avec un fond transparent. USED FOR FIGURE 1.

    :param data_path: Chemin vers le fichier JSON des résultats d'optimisation.
    :param output_dir: Chemin du dossier où enregistrer les images PNG.
    """
    # --- Charger résultats optimisés ---
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {data_path} n'a pas été trouvé.")
        return

    X = np.array(data["X"])
    F = np.array(data["F"])
    n_solutions = X.shape[0]

    # --- Créer le répertoire de sortie s'il n'existe pas ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Dossier de sortie créé: {output_dir}")
    else:
        print(f"Utilisation du dossier de sortie existant: {output_dir}")

    # --- Itérer et enregistrer chaque roulement ---
    for i in range(n_solutions):
        x = X[i]
        f = F[i]

        # Recréer l'objet 'rollerbearing' (nécessite la librairie 'utils')
        # Si vous n'avez pas 'utils', il faudra fournir une implémentation
        try:
            rb = utils.make_rollerbearing(x)
        except NameError:
            print("Erreur: La fonction utils.make_rollerbearing n'est pas définie ou importée.")
            return

        # Logique pour déterminer la couleur du plot (type_fail)
        # Basée sur la valeur la plus élevée dans le vecteur d'objectifs F
        type_fail_idx = np.argmax(f)
        type_fail = [1, 2, 2, 0][type_fail_idx]  # Réutilise la logique de votre code Dash

        # Création du plot Matplotlib
        fig, ax = plt.subplots()
        rb.roller.render(ax=ax, show=False, color=type_fail)
        ax.set_aspect('equal')
        ax.axis("off")

        # Configuration pour le fond transparent
        plt.tight_layout()
        output_filename = os.path.join(output_dir, f"bearing_solution_{i}.png")

        # Le paramètre 'transparent=True' est la clé
        plt.savefig(output_filename, format="png", transparent=True)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

        if (i + 1) % 10 == 0 or i == n_solutions - 1:
            print(f"Progression: {i + 1}/{n_solutions} plots enregistrés.")

    print(f"\nTerminé. Tous les {n_solutions} plots ont été enregistrés dans {output_dir}")

def weibul_curve_and_tR(lifetimes, R=0.90):
    import numpy as np
    import matplotlib.pyplot as plt

    lifetimes = np.sort(np.array(lifetimes))
    n = len(lifetimes)
    ranks = np.arange(1, n + 1)
    failure_probs = (ranks - 0.3) / (n + 0.4)

    weibull_y = np.log(np.log(1 / (1 - failure_probs)))
    log_lifetimes = np.log(lifetimes)
    # weibull_y = 1 / (1 - failure_probs)
    # log_lifetimes = lifetimes

    slope, intercept = np.polyfit(log_lifetimes, weibull_y, 1)
    beta = slope
    eta = np.exp(-intercept / beta)

    t_R = eta * (-np.log(R))**(1.0 / beta)

    print("beta: ", slope, " | eta: ", eta)
    # print("weigth: ", slope, " | bias: ", intercept)

    # Optionnel : tracé (comme dans ton code)
    plt.figure(figsize=(8,6))
    plt.scatter(lifetimes, weibull_y, color="k")
    plt.plot(lifetimes, slope * np.log(lifetimes) + intercept, color="k")
    plt.xscale('log')
    # plt.xlabel('Cycles to Failure (log scale)')
    # plt.ylabel('ln(ln(1 / (1 - P)))')
    # plt.title('Weibull Probability Plot (échelle transformée)')
    # plt.grid(True, which='both')
    # plt.legend()
    plt.show()
    # plt.savefig("weibull_breaking")

    return beta, eta, t_R

def get_kb(ptx, pty):
    # Linéarisation
    X = np.log(ptx)
    Y = np.log(pty)

    # Régression linéaire
    beta, lnK = np.polyfit(X, Y, 1)
    K = np.exp(lnK)

    print(f"β = {beta:.4f}")
    print(f"K = {K:.4f}")

    # === 3. Création de la courbe ajustée ===
    x_fit = np.linspace(min(ptx) * 0.8, max(ptx) * 1.2, 200)
    y_fit = K * x_fit ** beta

    # === 4. Tracé ===
    plt.figure(figsize=(8, 6))
    plt.scatter(ptx, pty, color='k', label='Points expérimentaux')
    plt.plot(x_fit, y_fit, color='k', label=f"Ajustement : y = {K:.2f}·x^{beta:.2f}")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Ajustement de la loi de puissance y = K·x^β')
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # plt.legend()
    plt.show()

def plot_score_conv(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruire les tableaux NumPy
    X = np.array(data["history"][0])
    # Le code original a commenté l'indice 1 (mean), nous le laissons ainsi
    # hist_datamean = np.array(data["history"][1])
    hist_datamin = np.array(data["history"][2])

    s1 = hist_datamin[:, 0]
    s2 = hist_datamin[:, 1]
    s3 = hist_datamin[:, 2]
    s4 = hist_datamin[:, 3]

    plt.plot(X, s1, label="disloc", linestyle='-', color="k")  # Trait plein
    plt.plot(X, s2, label="block", linestyle='--', color="k")  # Pointillés/Tirets
    plt.plot(X, s3, label="thetay", linestyle=':', color="k")  # Points
    plt.plot(X, s4, label="break", linestyle='-.', color="k")  # Point-Trait
    # plt.plot(X, hist_datamean, label="mean")

    plt.legend()
    plt.show()

def var_multiple_optim(*numpaths):
    """Function of figure 16.B"""
    XS, FS = list(), list()
    trees = list()

    # 1. Chargement des données (identique à votre code)
    for num in numpaths:
        # Attention : Assurez-vous que le chemin est correct dans votre environnement
        path = f"../data/optim_results/optim_cylbearing{num}.json"
        with open(path, "r") as f:
            data = json.load(f)
        X = np.array(data["X"])
        F = np.array(data["F"])
        XS.append(X)
        FS.append(F)
        trees.append(KDTree(F))

    # 2. Calcul des variances (identique à votre code)
    vars_list = list()
    # Récupérer les indices qui trient le premier run selon le 1er objectif (colonne 0)
    sort_indices = np.argsort(FS[0][:, 0])
    ref_front = FS[0][sort_indices]

    for sol in ref_front:
        all_x = []
        for data, tree in zip(XS, trees):
            distance, index = tree.query(sol, k=1)
            nearest = data[index]
            all_x.append(nearest)

        # Utilisation de numpy pour simplifier le calcul de variance (plus rapide)
        all_x_np = np.array(all_x)
        # Variance sur l'axe 0 (entre les seeds) pour chaque paramètre
        var = np.var(all_x_np, axis=0)
        vars_list.append(var)

    vars_array = np.array(vars_list)  # Forme: (Nb_Solutions, Nb_Paramètres)

    # 3. Affichage (MODIFIÉ)
    plt.figure(figsize=(10, 6))

    # # Tracer les lignes de variance
    # nb_params = vars_array.shape[1]
    # for ictrl in range(nb_params):
    #     plt.plot(vars_array[:, ictrl], label=f'Param {ictrl + 1}')
    plt.plot(np.mean(vars_array, axis=1), color="k")

    # --- NOUVEAU CODE : Identifier et marquer les extrêmes ---

    # Trouver les indices des minimums pour chaque objectif dans le run de référence
    # axis=0 cherche le min dans chaque colonne (chaque fonction objectif)
    min_indices = np.argmin(ref_front, axis=0)

    # Couleurs ou labels pour différencier les objectifs si besoin
    objectifs_labels = [f"Obj {i + 1}" for i in range(ref_front.shape[1])]

    # Ajouter du texte pour dire quel objectif est minimisé ici
    for i, idx in enumerate(min_indices):
        # On décale légèrement le texte en Y pour qu'il soit lisible
        # Si la variance est faible, on écrit au-dessus, sinon attention aux chevauchements
        max_var_at_idx = np.max(vars_array[idx])
        # plt.text(idx, 0 - (max(vars_array.flatten()) * 0.05),
        #          objectifs_labels[i],
        #          color='red',
        #          ha='center',
        #          rotation=45,
        #          fontweight='bold')

        # Optionnel : Ajouter une ligne verticale pointillée pour bien voir la coupe
        plt.axvline(x=idx, color='k', linestyle='--', alpha=0.8)

    # ---------------------------------------------------------

    # plt.title("Variance des paramètres le long du Front de Pareto (Run 0)")
    # plt.xlabel("Index de la solution (dans FS[0])")
    # plt.ylabel("Variance du paramètre")
    # plt.legend()
    # plt.tight_layout()
    plt.show()

def plot_optim(path, w=None, e=None, rescaled=False):
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array(data["X"])
    F = np.array(data["F"])

    x_axis = "block_score"
    y_axis = "thetay0"
    labels_objectives = ["disloc_score", "block_score", "thetay0", "FBB"]
    axis_options = [labels_objectives[i] for i in range(len(labels_objectives))]
    x_idx = axis_options.index(x_axis)
    y_idx = axis_options.index(y_axis)
    ranks = np.argsort(weighted_scores(F, w=w, e=e, rescaled=rescaled))  # ordre croissant
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(ranks))
    # Scatter avec palette de couleurs
    # Création de la figure (optionnel, pour définir la taille)
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(F[:, x_idx], F[:, y_idx], c=rank_values, cmap='jet', s=60, alpha=0.9)
    cbar = plt.colorbar(sc)
    cbar.set_label("Rank")

    # 3. Titres et Labels
    # plt.xlabel(x_axis)
    # plt.ylabel(y_axis)
    # plt.title("Pareto Front")

    # Optionnel : Ajouter une grille pour faciliter la lecture
    # plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

if __name__=="__main__":
    # BREAKING FAILURE
    # BSPBRWN7#3: 24min, BSPBRWN7#2: 36min, BSPBRWN7#6: 49min, BSPBRWN7#5: 57min, BSPBRWN7#4: 67min
    # weibul_curve_and_tR([24 * 1850, 36 * 1850, 49 * 1850, 57 * 1850, 67 * 1850])

    # BSPBRWN7#2-3-4-5-6 FBB: 46min, BSPBRWN8#3 FBB: 88min, BSPBRWN9#1-2-3 FBB: 65/112/181min=119min, BSPBRWN11#3 FBB: 453min
    # get_kb([3 * 7, 3 * 8, 3 * 9, 3 * 11], [46 * 1850, 88 * 1850, 119.3 * 1850, 453 * 1850])

    # DISLOCATION FAILURE
    # BSPBWOPT5-88#0-3-4-5-6-7 FD
    # weibul_curve_and_tR([28 * 1850, 29 * 1850, 53 * 1850, 71 * 1850, 76 * 1850, 182 * 1850]) # , 1850 * 1850])

    # BRWOPT5#98:0min (Sd=1.0) ,
    # BRWOPT5#0: 25min (pSd=0.30899),
    # BSPBWOPT5-88#MEAN: 73min (Sd=0.24) (pSd=0.9),
    # BRWOPT5#2: 239min (Sd=0.114) (pSd=0.11137),
    # BRWOPT5#1:450min (Sd=0.141) (pSd=0.10918),
    # BSPBWOPT5-45#MEAN: 582min (Sd=0.254) (pSd=0.47113)
    # get_kb([(1/0.9) - 1, (1/0.4713) - 1], [73 * 1850, 582 * 1850])

    path = "../data/optim_results/optim_cylbearing6.json"
    opt.dashboard_roller(path, w=[9.7e5, 0, 0, 2.23e-2], e=[1, 1, 1, 4.95], rescaled=True)  # [disloc, b_eq, thetay, fbb]
    # plot_score_conv(path)
    # var_multiple_optim(14, 6, 16)
    # plot_optim(path, w=[9.7e5, 0, 0, 2.23e-2], e=[1, 1, 1, 4.95], rescaled=True)