from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import multiprocessing
import numpy as np
import pandas as pd
import utils as utils
import roller_design as rd
import matplotlib
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output
import numpy as np
import json
import plotly.express as px
from io import BytesIO
import base64


# ROLLER BEARINGS
class RollerFunctionProblem(ElementwiseProblem):
    def __init__(self, n_ctrl=5, **kwargs):
        self.n_ctrl = n_ctrl

        RB_MAX = (np.tan(70 * np.pi / 180) * (rd.L / (n_ctrl * 2))) + rd.RB_MIN  # = 6.62 but set to 8 before
        super().__init__(
            n_var=n_ctrl + 1,
            n_obj=4,
            n_constr=3,
            xl=np.array([rd.RB_MIN] * (n_ctrl + 1)),
            xu = np.array([RB_MAX] * (n_ctrl + 1)),
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        rb = utils.make_rollerbearing(x, rext=rd.R_EXT, rshaft=rd.R_SHAFT, L=rd.L)
        # rb.score_weight["fbb_score"] = 0.5
        out["G"] = rb.constraints()  # contraintes doivent être <= 0
        out["F"] = rb.score()  # the 4 scores
        # print(f"[OK] ob: {[float(round(value, 3)) for value in out["F"]]}, co: {[int(c) for c in out['G']]}, {str(rb)}")


def dashboard_roller(path):
    matplotlib.use('Agg')  # backend sans interface graphique

    # --- Charger résultats optimisés ---
    with open(path, "r") as f:
        data = json.load(f)

    X = np.array(data["X"])
    F = np.array(data["F"])
    n_objectives = F.shape[1]

    # Options d’axes à afficher
    labels_objectives = ["disloc_score", "block_score","thetay0", "FBB"]
    axis_options = [labels_objectives[i] for i in range(n_objectives)]

    # --- App Dash ---
    app = dash.Dash(__name__)
    app.title = "Bearings Explorer"

    app.layout = html.Div([
        html.H1("Multi-objectives optimization explorer", style={"textAlign": "center"}),

        html.Div([
            html.Div([
                html.Label("Axe X"),
                dcc.Dropdown(id="x-axis", options=[{"label": ax, "value": ax} for ax in axis_options], value="F1"),
                html.Label("Axe Y"),
                dcc.Dropdown(id="y-axis", options=[{"label": ax, "value": ax} for ax in axis_options], value="F2"),
                dcc.Graph(id="pareto-plot", style={"height": "400px"})
            ], style={"width": "40%", "padding": "20px"}),

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
        Input("pareto-plot", "clickData")
    )
    def update_plot(x_axis, y_axis, clickData):
        if x_axis not in labels_objectives:
            x_axis = "disloc_score"
        if y_axis not in labels_objectives:
            y_axis = "block_score"
        x_idx = axis_options.index(x_axis)
        y_idx = axis_options.index(y_axis)

        ranks = np.argsort(utils.weighted_scores(F))  # ordre croissant
        rank_values = np.empty_like(ranks)
        rank_values[ranks] = np.arange(len(ranks))

        # Scatter avec palette de couleurs
        fig = px.scatter(
            x=F[:, x_idx],
            y=F[:, y_idx],
            color=rank_values,  # valeur numérique = classement
            labels={"x": x_axis, "y": y_axis, "color": "Rank"},
            title="Pareto Front",
            color_continuous_scale="Cividis",  # ou "Plasma", "Cividis", etc.
        )

        fig.update_traces(marker=dict(size=8))

        selected_index = clickData["points"][0]["pointIndex"] if clickData else 0
        fig.update_traces(marker=dict(size=8))
        fig.add_trace(px.scatter(x=[F[selected_index, x_idx]], y=[F[selected_index, y_idx]],
                                 labels={"x": x_axis, "y": y_axis}).data[0])

        # Extraire la solution
        x = X[selected_index]
        rb = utils.make_rollerbearing(x)

        # Texte d'infos
        info = [
            html.P(f"Solution #{selected_index} #Rank{rank_values[selected_index]}"),
            html.P(f"RB = {str(rb)}"),
            html.Ul([html.Li(f"{name} = {F[selected_index, i]:.5f}") for i, name in enumerate(axis_options)])
        ]

        type_fail = np.argmax(F[selected_index])
        type_fail = [1, 2, 2, 0][type_fail]
        fig2, ax2 = plt.subplots()
        rb.roller.render(ax=ax2, show=False, color=type_fail)
        ax2.set_aspect('equal')
        ax2.axis("off")

        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format="png")
        plt.close(fig2)
        img2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

        return fig, info, f"data:image/png;base64,{img2}"

    # --- Lancement ---
    app.run(debug=True)


def run_optimisation(output_dir):
    problem = RollerFunctionProblem(n_ctrl=5)  # , elementwise_runner=runner)
    algorithm = NSGA2(pop_size=50)
    res = minimize(problem, algorithm, termination=('n_gen', 70), seed=42, verbose=True)
    info = "70gen, popsize50, score: [disloc_score(realSDF), irotblock_score(pi/4 traj), thetay0, fbb_score(N*Rmin)], constr: [rout_constr, rin_constr, maxslope_contr], info:L = 2 * INCH_TO_MM  # 15. CLEARANCE = 0.3. R_EXT = 1.5 * INCH_TO_MM  # 30. R_SHAFT = 1 * INCH_TO_MM  # 6.5. T_OUT_MIN = 5. T_IN_MIN = 3. ALPHA_LIM = 60  # (deg). RB_MIN = 2.5"
    utils.save_result_to_json(output_dir, res, infos=info)


if __name__ == "__main__":
    path = "../data/optim_results/optim_cylbearing7.json"
    run_optimisation(path)
    # dashboard_roller(path)

    # rb = utils.getbyrank(path, 45)
    # rb.roller.render()
    # rb.acc_grid = 200
    # rb.d_score(N=5, clearance_factor=2.0)
    # print()
    # rb = utils.getbyindex(path, 3)
    # rb.acc_grid = 50
    # rb.r_score(n_samples=5, tol=1e-2, max_iter=20, clearance_factor=2.0, silent=False)
    # rb.b_function(N=20, show=True)

#
    # ext=HE.Extruder(rb)
    # rmax = max(best_bp.loop_sun.rmax, best_bp.loop_sat.rmax, best_bp.loop_or.rmax)
    # alpha_max = 60
#
    # ext.twist_func = HE.linear_rev(alpha_max=alpha_max, rmax=rmax)
    # ext.extrude(nsat=None, nslices=15)
    # ext.export("opt2.stl")