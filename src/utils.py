import json
import roller_design as RD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import optim_rollerbearing as opt
from scipy.spatial import KDTree
import os

def make_rollerbearing(x):
    rext = RD.R_EXT
    rshaft = RD.R_SHAFT
    L = RD.L
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

def getbyrank(path, rank, Pf=0.5, **kwargs):
    """Rank from 0 to len(F)"""
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])

    lts = lifetimes(F, Pf=Pf)
    L = np.min(lts, axis=1)
    ranks = np.argsort(1 / (L + 0.01))  # ordre croissant
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(ranks))
    idx = int(np.where(rank_values == rank)[0])
    print("lifetimes:", lts[idx])
    return make_rollerbearing(X[idx], **kwargs)

def select_best(path):
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])

    lts = lifetimes(F)
    L = np.min(lts, axis=1)

    best_idx = np.argmax(L)

    # Return the generated object for the best design
    return make_rollerbearing(X[best_idx])

def getbyindex(path, idx):
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    return make_rollerbearing(X[idx])

def lifetimes(F, Pf=0.5):
    w = [6.4e6, 5.48, 0, 2.23e-2]  # 9.7e6 for opt 18 and after, 9.7e5 else
    n = [0.85, 6.53, 1, 4.95]
    weights = np.array(w)
    exponents = np.array(n)
    if Pf != 0.5:
        if Pf < 0.0:
            Pf = 0.0
        elif Pf >= 1.0:
            Pf = 0.99
        betas = np.array([1.41, 1.34, 1, 2.53])
        Kpf = (np.log(1-Pf)/np.log(1-0.5))**(1/betas)
    else:
        Kpf = np.array([1, 1, 1, 1])
    # Here the goal is to estimate the lifetime
    F = (1 / F) - 1
    weighted_vector = weights * np.power(F, exponents)
    weighted_vector[:, 2] = np.array([np.inf]*len(F))
    return weighted_vector * Kpf

def weighted_scores(F, rescaled=False, Pf=0.5):
    if rescaled:
        # Here the goal is to estimate the lifetime
        lts = lifetimes(F, Pf=Pf)
        L = np.min(lts, axis=1)
        return 1 / (L+1e-5)

    # Apply weights element-wise
    weighted_vector = F

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
    """Plot the 4 scores along all generations."""
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruire les tableaux NumPy
    X = np.array(data["history"][0])
    # Le code original a commenté l'indice 1 (mean), nous le laissons ainsi
    hist_datamin = np.array(data["history"][2])

    s1 = hist_datamin[:, 0]
    s2 = hist_datamin[:, 1]
    s3 = hist_datamin[:, 2]
    s4 = hist_datamin[:, 3]

    plt.plot(X[3:], s1[3:], label="disloc", linestyle='-', color="k")  # Trait plein
    plt.plot(X[3:], s2[3:], label="block", linestyle='--', color="k")  # Pointillés/Tirets
    plt.plot(X[3:], s3[3:], label="thetay", linestyle=':', color="k")  # Points
    plt.plot(X[3:], s4[3:], label="break", linestyle='-.', color="k")  # Point-Trait

    plt.legend()
    plt.show()

def var_multiple_optim(*numpaths):
    """Function of figure 16.B canceled. DEPRECATED"""
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

def plot_optim(path, rescaled=False):
    """Plot the pareto front with jet map"""
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array(data["X"])
    F = np.array(data["F"])

    x_axis = "disloc_score"
    y_axis = "FBB"
    labels_objectives = ["disloc_score", "block_score", "thetay0", "FBB"]
    axis_options = [labels_objectives[i] for i in range(len(labels_objectives))]
    x_idx = axis_options.index(x_axis)
    y_idx = axis_options.index(y_axis)
    ranks = np.argsort(weighted_scores(F, rescaled=rescaled))  # ordre croissant
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

def plot_lifetime_contribution(path, bearing_index):
    """plot lifetime of the bearing in function of the probability of failure."""
    with open(path, "r") as f:
        data = json.load(f)
    F = np.array(data["F"])

    Pfs = list()
    Lts = list()
    for k in np.linspace(0, 1, 20, endpoint=False):
        lts = lifetimes(F, Pf=k)
        Lts.append(lts[bearing_index])
        Pfs.append(k)

    plt.plot(Pfs, [l[0] for l in Lts], label="disloc", linestyle='-', color="k")  # Trait plein
    plt.plot(Pfs, [l[1] for l in Lts], label="block", linestyle='--', color="k")  # Pointillés/Tirets
    plt.plot(Pfs, [l[3] for l in Lts], label="break", linestyle='-.', color="k")  # Point-Trait

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_lifetime_chart(path, bearing_index, Pf=0.5):
    """
    Plot the roller bearing geometry on the left (30% width)
    and the horizontal lifetime chart on the right (70% width),
    with simplified styling.
    """

    # --- 1. Préparation des données ---

    try:
        # Chargement des données
        with open(path, "r") as f:
            data = json.load(f)
        X = np.array(data["X"])
        F = np.array(data["F"])
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return

    # Calcul des durées de vie
    try:
        # NOTE: Cette ligne dépend de votre fonction 'lifetimes'
        lts = lifetimes(F, Pf=Pf)
        lt = lts[bearing_index]
        lifetime = [lt[3], lt[0], lt[1]]
        # type_fail = ["break", "disloc", "block"]
        type_fail = ["br", "d", "bl"]
        c = lifetime.index(min(lifetime))
    except NameError:
        print("Erreur: La fonction 'lifetimes' n'est pas définie.")
        return
    except IndexError:
        print(f"Erreur: 'bearing_index' {bearing_index} est hors limites.")
        return

    # Récupération des couleurs
    try:
        bar_colors = RD.COLORS
    except NameError:
        print("Avertissement: 'RD.COLORS' non défini. Utilisation des couleurs par défaut.")
        bar_colors = plt.cm.get_cmap('Set1', len(type_fail)).colors  # Fallback colors

    # --- 2. Création de la Figure et des Sous-graphiques avec Mise en Page Personnalisée ---

    # Crée une figure
    fig = plt.figure(figsize=(12, 4))  # Taille ajustée pour l'affichage 30/70

    # Axes pour la géométrie (30% de la largeur)
    # plt.subplot2grid((rows, columns), (row, col), colspan=w, rowspan=h)
    ax_geometry = plt.subplot2grid((1, 10), (0, 0), colspan=4)  # Utilise 3 colonnes sur 10 (soit 30%)

    # Axes pour le graphique de durée de vie (70% de la largeur)
    ax_chart = plt.subplot2grid((1, 10), (0, 4), colspan=6)  # Utilise 7 colonnes sur 10 (soit 70%)

    # --- 3. Affichage de la Géométrie (Sous-graphique de gauche) ---

    # Construction du roulement
    # NOTE: X[bearing_index] contient les paramètres géométriques
    rb = make_rollerbearing(X[bearing_index])
    # Affichage du roulement sur le premier axe
    rb.roller.render(ax=ax_geometry, show=False, plot_rotcenter=False,
                     color=c, alpha=1.0, filled=True)
    # Supprimer le titre
    ax_geometry.set_title("")
    ax_geometry.set_aspect('equal', adjustable='box')
    ax_geometry.axis('off')  # Cache les axes


    # --- 4. Affichage du Graphique de Durée de Vie (Sous-graphique de droite) ---

    # Tracé des barres horizontales sur le second axe
    # Ajout de 'edgecolor' et 'linewidth' pour l'encadrement noir
    ax_chart.barh(type_fail, lifetime, color=bar_colors, height=0.7,
                  edgecolor='black', linewidth=1.0)

    # Configuration des axes pour le graphique de durée de vie
    ax_chart.set_xscale('log')
    ax_chart.set_xlim(1e2, 1e15)
    ax_chart.tick_params(axis='x', labelsize=28)
    ax_chart.tick_params(axis='y', labelleft=False)

    ax_chart.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Ajustement de l'espacement
    plt.tight_layout()
    plt.show()

def plot_generations_comparison(path, generations, top_k=5, Pf=0.5):
    """
    Visualise les 'top_k' meilleures géométries et leurs durées de vie pour plusieurs générations.

    :param path: Chemin vers le fichier JSON contenant les données de simulation.
    :param generations: Liste des numéros d'indices de génération à afficher (ex: [10, 50, 100]).
    :param top_k: Nombre de meilleures géométries à afficher par génération.
    :param Pf: Probabilité de défaillance pour le calcul de la durée de vie.
    """

    # --- 1. Préparation et Chargement des Données ---

    if not generations:
        print("Erreur: La liste 'generations' est vide.")
        return


    # Chargement des données
    with open(path, "r") as f:
        data = json.load(f)
    # Récupération des données pour chaque génération demandée
    generation_data = {}
    for i_gen in generations:
        # i_gen est l'index de la génération
        if i_gen < len(data["history"][5]):
            # Récupère l'ensemble X (géométrie) et F pour cette génération
            X = np.array(data["history"][5][i_gen])
            F = np.array(data["history"][3][i_gen])
            # Calcul des durées de vie (lts) pour tous les roulements de cette génération
            lts = lifetimes(F, Pf=Pf)
            L = np.min(lts, axis=1)
            # Ajout de l'index du roulement pour le suivi
            indices = np.arange(len(L))
            top_solutions = indices[L.argsort()[::-1]][:top_k]

            generation_data[i_gen] = {
                'X_top': X[top_solutions],
                'LTS_top': lts[top_solutions, :]
            }
        else:
            print(f"Avertissement: Génération {i_gen} non trouvée dans les données.")


    # --- 2. Configuration de la Figure ---

    num_generations = len(generation_data)
    if num_generations == 0:
        print("Aucune donnée valide à afficher.")
        return

    # La figure sera large pour accueillir toutes les générations côte à côte
    # Chaque génération a 1 colonne de géométrie (30% de la largeur du bloc) et 1 colonne de barres (70% du bloc)
    # Total de colonnes dans la grille principale = num_generations * 2

    # Largeur relative pour chaque bloc (géométrie vs. barres) dans une génération: 3 (Géom) : 7 (Barres)
    col_ratios = [3, 7] * num_generations

    # Crée la figure et les axes, en utilisant GridSpec pour un contrôle total
    fig = plt.figure(figsize=(4 * num_generations, 1 + 2 * top_k))

    # Création du GridSpec (grille de base)
    gs = GridSpec(nrows=top_k, ncols=num_generations * 2, figure=fig, width_ratios=col_ratios)

    # Couleurs pour les barres
    bar_colors = RD.COLORS
    type_fail = ["br", "d", "bl"]  # L'ordre des barres

    # --- 3. Boucle d'Affichage ---

    for gen_col_index, (i_gen, data) in enumerate(generation_data.items()):
        X_top = data['X_top']
        LTS_top = data['LTS_top']

        # Pour chaque top_k solution
        for row_index in range(top_k):
            if row_index >= len(LTS_top):
                break  # Arrête si on n'a pas atteint top_k solutions

            lt_data = LTS_top[row_index]
            # Durées de vie spécifiques : [lt_d, lt_bl, lt_br]
            # L'exemple utilisateur utilise: lt[3], lt[0], lt[1] -> overall, disloc, block
            # Je prends les trois valeurs spécifiques (index 0, 1, 2) pour la barre chart: [disloc, block, break]
            lifetime_values = [lt_data[3], lt_data[0], lt_data[1]]
            c = np.argmin(lifetime_values)  # Index de l'échec le plus court

            # --- A. Sous-graphique Géométrie (30% de la colonne) ---

            # Définition des axes de la géométrie
            # gs[row, col*2] donne l'axe dans la grille
            ax_geometry = fig.add_subplot(gs[row_index, gen_col_index * 2])

            # Affichage de la géométrie
            rb = make_rollerbearing(X_top[row_index])
            rb.roller.render(ax=ax_geometry, show=False, plot_rotcenter=False,
                             color=c, alpha=1.0, filled=True)

            ax_geometry.set_aspect('equal', adjustable='box')
            ax_geometry.axis('off')

            # --- B. Sous-graphique Bar Chart (70% de la colonne) ---

            # Définition des axes du graphique
            # gs[row, col*2 + 1] donne l'axe du bar chart
            ax_chart = fig.add_subplot(gs[row_index, gen_col_index * 2 + 1])

            # Tracé des barres horizontales
            ax_chart.barh(type_fail, lifetime_values,
                          color=bar_colors[:len(type_fail)],
                          height=0.7,
                          edgecolor='black', linewidth=1.0)

            # Mise à l'échelle logarithmique pour les durées de vie
            ax_chart.set_xscale('log')
            ax_chart.set_xlim(1e1, 1e16)

            # Configuration des ticks et labels
            ax_chart.tick_params(axis='y', labelleft=False)
            ax_chart.tick_params(axis='x', labelsize=10)
            ax_chart.grid(True, axis='x', linestyle='--', alpha=0.6)

            # Simplification des labels Y (ne les affiche que pour la première rangée)
            ax_chart.tick_params(axis='y', labelleft=True, labelsize=10)
            ax_chart.set_yticklabels(type_fail)

            # Afficher l'axe X (échelle) uniquement pour la dernière ligne de chaque colonne
            if row_index < top_k - 1:
                ax_chart.tick_params(axis='x', labelbottom=False)

            # Ajout de la valeur de durée de vie la plus courte
            min_lt = min(lt_data)  # Durée de vie globale minimale (overall_min)
            ax_chart.text(
                ax_chart.get_xlim()[1] * 0.5,  # Position X (proche de la limite droite)
                0.5,  # Position Y (au centre)
                f"Lifetime:\n{min_lt:.2e}",
                ha='right', va='center',
                transform=ax_chart.transData,
                fontsize=10,
                color='k',
                # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )

    # Ajustement final
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])  # Laisser de l'espace pour le suptitle
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
    # get_kb([((1/0.9) - 1)/10, ((1/0.4713) - 1)/10, (1/0.776) - 1], [73 * 1850, 582 * 1850, 1147 * 1850])

    # BLOCKING FAILURE
    # weibul_curve_and_tR([20/60 * 1850, 1.26 * 1850, 0.25 * 1850, 35/60 * 1850])

    # get_kb([(1 / 0.309) - 1, (1 / 0.184) - 1, (1 / 0.16) - 1], [0.65 * 1850, 25 * 1850, 259 * 1850])

    path = "../data/optim_results/optim_cylbearing20.json"

    # plot_generations_comparison(path, [3, 10, 20, 29], top_k=5, Pf=0.5)
    # plot_lifetime_contribution(path, 2)
    # plot_lifetime_chart(path, 20)
    opt.dashboard_roller(path, rescaled=True)  # [disloc, b_eq, thetay, fbb]
    # plot_score_conv(path)
    # var_multiple_optim(14, 6, 16)
    # plot_optim(path, rescaled=True)