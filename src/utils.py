import json
import roller_design as RD
import numpy as np

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

def getbyrank(path, rank, w=None):
    """Rank from 0 to len(F)"""
    # --- Charger résultats optimisés ---
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])

    # Somme des carrés par ligne
    scores = weighted_scores(F, weights=w)
    ranks = np.argsort(scores)  # ordre croissant
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(ranks))
    idx = int(np.where(rank_values == rank)[0])

    return make_rollerbearing(X[idx])

def getbyindex(path, idx):
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    return make_rollerbearing(X[idx])

def weighted_scores(F, weights=None):
    #TODO: Add exponents n for calculing them
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    weights = np.array(weights)
    if F[0].shape != weights.shape:
        raise ValueError("Vector and weights must have the same shape.")

    # Apply weights element-wise
    weighted_vector = F * weights

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

    # weibull_y = np.log(np.log(1 / (1 - failure_probs)))
    # log_lifetimes = np.log(lifetimes)
    weibull_y = 1 / (1 - failure_probs)
    log_lifetimes = lifetimes

    slope, intercept = np.polyfit(log_lifetimes, weibull_y, 1)
    beta = slope
    eta = np.exp(-intercept / beta)

    t_R = eta * (-np.log(R))**(1.0 / beta)

    print("beta: ", slope, " | eta: ", eta)

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
    import numpy as np
    import matplotlib.pyplot as plt

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

if __name__=="__main__":
    # BREAKING FAILURE
    # BSPBRWN7#3: 24min, BSPBRWN7#2: 36min, BSPBRWN7#6: 49min, BSPBRWN7#5: 57min, BSPBRWN7#4: 67min
    # weibul_curve_and_tR([24 * 1850, 36 * 1850, 49 * 1850, 57 * 1850, 67 * 1850])

    # BSPBRWN7#2-3-4-5-6 FBB: 46min, BSPBRWN8#3 FBB: 88min, BSPBRWN9#1-2-3 FBB: 65/112/181min=119min, BSPBRWN11#3 FBB: 453min
    # get_kb([3 * 7, 3 * 8, 3 * 9, 3 * 11], [46 * 1850, 88 * 1850, 119.3 * 1850, 453 * 1850])

    # DISLOCATION FAILURE
    weibul_curve_and_tR([28 * 1850, 29 * 1850, 53 * 1850, 182 * 1850, 1850 * 1850])