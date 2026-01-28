from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
import utils as utils
import roller_design as rd
import matplotlib
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output
import numpy as np
import json
import time
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions


# ROLLER BEARINGS
class RollerFunctionProblem(ElementwiseProblem):
    def __init__(self, n_ctrl=5, **kwargs):
        self.n_ctrl = n_ctrl
        RB_MAX0 = (np.tan(70 * np.pi / 180) * (rd.L / (n_ctrl * 2))) + rd.RB_MIN  # = 6.62 but set to 8 before
        RB_MAX = min([(rd.R_EXT - (rd.T_OUT_MIN + rd.T_IN_MIN + rd.R_SHAFT + 2 * rd.CLEARANCE)) / 2, RB_MAX0])
        x = np.random.uniform(rd.RB_MIN, RB_MAX, (n_ctrl,))
        super().__init__(
            n_var=n_ctrl + 1,
            n_obj=4,
            n_constr=3,  # 3
            xl=np.array([rd.RB_MIN] * (n_ctrl + 1)),
            xu = np.array([RB_MAX] * (n_ctrl + 1)),
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        rb = utils.make_rollerbearing(x)
        # rb.score_weight["fbb_score"] = 0.5
        out["G"] = rb.constraints()  # contraintes doivent être <= 0
        out["F"] = rb.score()  # the 4 scores
        # print(f"[OK] ob: {[float(round(value, 3)) for value in out["F"]]}, co: {[int(c) for c in out['G']]}, {str(rb)}")

class StoreObjectivesCallback(Callback):
    """
    Callback pour stocker les valeurs minimales de chaque objectif
    à la fin de chaque génération.
    """
    def __init__(self):
        super().__init__()
        self.minobjectives = []
        self.meanobjectives = []
        self.objectives = []
        self.constraints = []
        self.generations = []
        self.X = []

    def notify(self, algorithm):
        # Récupère la population actuelle
        pop = algorithm.pop

        # Récupère les objectifs de la population
        F = pop.get("F")
        G = pop.get("G")
        X = pop.get("X")

        # Stocke les données
        self.objectives.append(F.tolist())
        self.X.append(X.tolist())
        self.meanobjectives.append(float(np.mean(F)))
        self.minobjectives.append(np.min(F, axis=0).tolist())
        self.constraints.append(G.tolist())
        self.generations.append(algorithm.n_gen)

def run_optimisation(output_dir):
    # PARAMETERS
    n_pop = 50
    n_gen = 30
    n_ctrl = 5
    seed = 42

    info = "S: [disloc_score(SAF (/10)), irotblock_score(pi/4 traj), thetay0, fbb_score(Nb*Rmin)], constr: [rout_constr, rin_constr, maxslope_contr] \n"
    info += "Nctrl: " + str(n_ctrl) + "\n"
    info += "Npop: " + str(n_pop) + "\n"
    info += "Ngen: " + str(n_gen) + "\n"
    info += "L: " + str(rd.L) + "\n"
    info += "R_EXT: " + str(rd.R_EXT) + "\n"
    info += "CLEARANCE: " + str(rd.CLEARANCE) + "\n"
    info += "R_SHAFT: " + str(rd.R_SHAFT) + "\n"
    info += "T_OUT_MIN: " + str(rd.T_OUT_MIN) + "\n"
    info +="T_IN_MIN: " + str(rd.T_IN_MIN) + "\n"
    info += "ALPHA_LIM (deg) : " + str(rd.ALPHA_LIM) + "\n"
    info += "RB_MIN: " + str(rd.RB_MIN) + "\n"
    info += "SEED: " + str(seed) + "\n"
    t0 = time.time()

    hc = StoreObjectivesCallback()
    problem = RollerFunctionProblem(n_ctrl=n_ctrl)  # , elementwise_runner=runner)
    algorithm = NSGA2(pop_size=n_pop)
    res = minimize(problem, algorithm, termination=('n_gen', n_gen), seed=seed, verbose=True, callback=hc)

    info += "T_OPTIM (min): " + str((time.time() - t0) / 60) + "\n"
    utils.save_result_to_json(output_dir, res,
                              infos=info,
                              history=[hc.generations,
                                       hc.meanobjectives,
                                       hc.minobjectives,
                                       hc.objectives,
                                       hc.constraints,
                                       hc.X])

def run_optimisation_moead(output_dir):
    # PARAMETERS
    n_pop = 35
    n_gen = 30
    n_ctrl = 5

    # ... (Le bloc 'info' reste le même) ...
    info = "S: [disloc_score(SAF /10), irotblock_score(pi/4 traj), thetay0, fbb_score(Nb*RBmin)], constr: [rout_constr, rin_constr, maxslope_contr] \n"
    info += "Nctrl: " + str(n_ctrl) + "\n"
    info += "Npop: " + str(n_pop) + "\n"
    info += "Ngen: " + str(n_gen) + "\n"
    info += "L: " + str(rd.L) + "\n"
    info += "R_EXT: " + str(rd.R_EXT) + "\n"
    info += "CLEARANCE: " + str(rd.CLEARANCE) + "\n"
    info += "R_SHAFT: " + str(rd.R_SHAFT) + "\n"
    info += "T_OUT_MIN: " + str(rd.T_OUT_MIN) + "\n"
    info += "T_IN_MIN: " + str(rd.T_IN_MIN) + "\n"
    info += "ALPHA_LIM (deg) : " + str(rd.ALPHA_LIM) + "\n"
    info += "RB_MIN: " + str(rd.RB_MIN) + "\n"
    t0 = time.time()

    history_callback = StoreObjectivesCallback()
    problem = RollerFunctionProblem(n_ctrl=n_ctrl)

    # *** CHANGEMENT ICI : Utilisation de MOEAD ***
    # La population de MOEAD est souvent égale au nombre de sous-problèmes (ici n_pop)
    algorithm = MOEAD(
        ref_dirs=get_reference_directions("das-dennis", problem.n_obj, n_points=n_pop),
        prob_neighbor=0.9,
        n_neighbors=15
    )
    # **********************************************

    res = minimize(problem, algorithm, termination=('n_gen', n_gen), seed=42, verbose=True, callback=history_callback)

    info += "T_OPTIM (min): " + str((time.time() - t0) / 60) + "\n"
    info += "ALGO: MOEAD"
    utils.save_result_to_json(output_dir, res, infos=info,
                              history=[history_callback.generations, history_callback.objectives])

def dashboard_roller(path, rescaled=False):
    matplotlib.use('Agg')

    # --- Charger résultats optimisés ---
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])
    n_objectives = F.shape[1]

    labels_objectives = ["disloc_score", "block_score", "thetay0", "FBB"]
    axis_options = [labels_objectives[i] for i in range(n_objectives)]

    app = dash.Dash(__name__)
    app.title = "Bearings Explorer"

    # Style des tableaux défini une fois pour toutes
    table_header_style = {'borderBottom': '1px solid #ddd', 'padding': '8px', 'textAlign': 'left', 'fontWeight': 'bold'}
    table_cell_style = {'padding': '8px', 'borderBottom': '1px solid #eee'}

    app.layout = html.Div([
        html.H1("Multi-objectives optimization explorer", style={"textAlign": "center"}),

        html.Div([
            # --- COLONNE GAUCHE : Contrôles et Graphique ---
            html.Div([
                html.Label("Axe X"),
                dcc.Dropdown(id="x-axis", options=[{"label": ax, "value": ax} for ax in axis_options], value="F1"),

                html.Label("Axe Y", style={'marginTop': '10px'}),
                dcc.Dropdown(id="y-axis", options=[{"label": ax, "value": ax} for ax in axis_options], value="F2"),

                # --- AJOUT DU SLIDER ICI ---
                html.Label("Failure probability (Pf)", style={'marginTop': '20px', 'fontWeight': 'bold'}),
                dcc.Slider(
                    id='pf-slider',
                    min=0,
                    max=1,
                    step=0.01,  # Pas de 1%
                    value=0.5,  # Valeur par défaut
                    marks={0: '0', 0.5: '0.5', 1: '1'},  # Repères visuels
                    tooltip={"placement": "bottom", "always_visible": True}  # Affiche la valeur au survol
                ),
                # ---------------------------

                dcc.Graph(id="pareto-plot", style={"height": "400px"})
            ], style={"width": "40%", "padding": "20px"}),

            # --- COLONNE DROITE : Détails ---
            html.Div([
                html.H3("Selected bearing details"),
                html.Div(id="solution-info"),

                html.Div([
                    html.Img(id="bearing-plot", style={"width": "70%", "display": "inline-block"})
                ], style={"width": "100%", "textAlign": "center", "marginTop": "10px", "marginBottom": "50px"}),

            ], style={"width": "58%", "padding": "20px", "borderLeft": "1px solid #ccc"})

        ], style={"display": "flex", "flexDirection": "row"})
    ])

    # --- Callback principal ---
    @app.callback(
        Output("pareto-plot", "figure"),
        Output("solution-info", "children"),
        Output("bearing-plot", "src"),
        Input("x-axis", "value"),
        Input("y-axis", "value"),
        Input("pareto-plot", "clickData"),
        Input("pf-slider", "value")  # --- AJOUT DE L'INPUT SLIDER ---
    )
    def update_plot(x_axis, y_axis, clickData, pf):  # --- AJOUT DE L'ARGUMENT pf ---
        # Gestion des axes par défaut si la sélection est invalide
        if x_axis not in labels_objectives:
            x_axis = "disloc_score"
        if y_axis not in labels_objectives:
            y_axis = "block_score"

        x_idx = axis_options.index(x_axis)
        y_idx = axis_options.index(y_axis)

        # --- UTILISATION DE pf DYNAMIQUE ---
        lifetimes = utils.lifetimes(F, Pf=pf)
        ranks = np.argsort(utils.weighted_scores(F, rescaled=rescaled, Pf=pf))
        # -----------------------------------

        rank_values = np.empty_like(ranks)
        rank_values[ranks] = np.arange(len(ranks))

        # Scatter plot
        fig = px.scatter(
            x=F[:, x_idx],
            y=F[:, y_idx],
            color=rank_values,
            labels={"x": x_axis, "y": y_axis, "color": "Rank"},
            title=f"Pareto Front (Pf={pf})",  # J'ai ajouté Pf dans le titre pour confirmation visuelle
            color_continuous_scale="Cividis",
        )
        fig.update_traces(marker=dict(size=8))

        # Gestion du point sélectionné
        selected_index = clickData["points"][0]["pointIndex"] if clickData else 0

        # Surbrillance du point sélectionné
        fig.add_trace(px.scatter(
            x=[F[selected_index, x_idx]],
            y=[F[selected_index, y_idx]],
        ).update_traces(marker=dict(size=12, color='red', symbol='circle-open', line=dict(width=2))).data[0])

        # --- Partie Info (Mise à jour avec le tableau) ---
        x = X[selected_index]
        rb = utils.make_rollerbearing(x)

        # Calcul du lifetime pour le point sélectionné
        estimated_lifetime = min([l for l in lifetimes[selected_index] if l > 0.0], default=0)

        info = [
            html.H4(f"Solution #{selected_index} (Rank {rank_values[selected_index]})", style={'marginBottom': '10px'}),
            html.Div([
                html.Span(f"RB: {rb}", style={'marginRight': '20px', 'fontWeight': 'bold'}),
                html.Span(f"Lifetime Total: {estimated_lifetime:.2e}", style={'fontWeight': 'bold', 'color': '#007bff'})
            ], style={'marginBottom': '15px'}),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Name", style=table_header_style),
                        html.Th("Score (S)", style=table_header_style),
                        html.Th("Rescaled (1/S-1)", style=table_header_style),
                        html.Th("Lifetime (Cycles)", style=table_header_style),
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(name, style=table_cell_style),
                        html.Td(f"{F[selected_index, i]:.2e}", style=table_cell_style),
                        html.Td(f"{(1/F[selected_index, i])-1:.2e}", style=table_cell_style),
                        html.Td(f"{lifetimes[selected_index][i]:.2e}", style=table_cell_style),
                    ]) for i, name in enumerate(axis_options)
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.9em'})
        ]

        # --- Génération de l'image ---
        type_fail = np.argmin(lifetimes[selected_index])
        type_fail = [1, 2, 2, 0][type_fail]  # Mapping des couleurs selon l'échec

        fig2, ax2 = plt.subplots()
        rb.roller.render(ax=ax2, show=False, color=type_fail)
        ax2.axis("off")

        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format="png", bbox_inches='tight')  # bbox_inches='tight' coupe mieux les bords blancs
        plt.close(fig2)
        img2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

        return fig, info, f"data:image/png;base64,{img2}"

    app.run(debug=True)


if __name__ == "__main__":
    path = "../data/optim_results/optim_cylbearing_mainwheels_team1.json"
    run_optimisation(path)

    # dashboard_roller(path, w=[9.7e5, 0, 0, 2.23e-2], e=[1, 1, 1, 4.95], rescaled=True)