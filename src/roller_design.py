#Faire un script qui prend une skecth freecad de roller en entrée et qui calcule de disassembly possible.
import matplotlib
matplotlib.use("TkAgg")   # backend interactif basé sur Tkinter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour projection 3d
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d, make_interp_spline
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from bisect import bisect_right
import utils
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm, norm
import copy
import json
import time
import math

COLORS = ["#F0D6C9", "#E2F3FA", "#B4E5A2"]
INCH_TO_MM = 25.4

# DIMENSIONAL CONSTRAINTS (in mm)
L = 0.63 * INCH_TO_MM  # 15
# L = 15
CLEARANCE = 0.3
# R_EXT = 30
# R_SHAFT = 6.5
R_EXT = (2.44 * INCH_TO_MM) / 2  # 30
R_SHAFT = (1.18 * INCH_TO_MM) / 2  # 6.5
T_OUT_MIN = 5
T_IN_MIN = 3
ALPHA_LIM = 60  # (deg)
RB_MIN = 2.5

class Element:
    def __init__(self, fcshape, plane=(0, 1)):
        self.fcshape = fcshape
        translate = ["X", "Y", "Z"]
        self.c = [translate[c] for c in plane]

    def get(self, strtype):
        if strtype == "xmin":
            return getattr(self.fcshape.BoundBox, self.c[0] + "Min")
        elif strtype == "xmax":
            return getattr(self.fcshape.BoundBox, self.c[0] + "Max")
        elif strtype == "ymin":
            return getattr(self.fcshape.BoundBox, self.c[1] + "Min")
        elif strtype == "ymax":
            return getattr(self.fcshape.BoundBox, self.c[1] + "Max")
        else:
            raise AssertionError("strtype not recognized")

class Part(ABC):
    def __init__(self):
        self.position = np.array([0.0, 0.0])  # Translation (x, y)
        self.rotation = 0.0  # En radians
        self.rot_center = np.array([0.0, 0.0])

        # Attributes for cache SDF
        self.sdf_grid = None
        self.sdf_local_origin = None  # Coordonnées locales (xmin, ymin) de la grille
        self.sdf_resolution = 0.05
        self.sdf_real_or_not = True

    def __str__(self):
        return self.__class__.__name__

    def set_transform(self, position, rotation, center="0"):
        if type(center) == np.ndarray:
            self.rot_center = center
        else:
            self.rot_center = self.get_rot_center(center=center)
        self.position = np.array(position)
        self.rotation = rotation

    def world_to_local(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # Décalage par rapport à la position globale
        dx = x - self.position[0]
        dy = y - self.position[1]
        # Décalage par rapport au centre de rotation
        dx -= self.rot_center[0]
        dy -= self.rot_center[1]
        # Rotation inverse autour du pivot
        c = np.cos(-self.rotation)
        s = np.sin(-self.rotation)
        x_local = c * dx - s * dy + self.rot_center[0]
        y_local = s * dx + c * dy + self.rot_center[1]
        return x_local, y_local

    def world_to_local_3D(self, x_global, y_global, z_global, thetay=0):
        """
        Transforme des coordonnées globales (x,y,z) en coordonnées locales
        en appliquant la translation et la rotation inverse autour de l'axe Y.
        Le centre de rotation self.rot_center est défini dans le plan XY,
        avec z fixé à 0.
        """

        x_global = np.asarray(x_global)
        y_global = np.asarray(y_global)
        z_global = np.asarray(z_global)

        # Étend le centre de rotation en 3D
        rot_center_3d = np.array([self.rot_center[0], self.rot_center[1], 0.0])

        # 1. Décalage par rapport à la position globale
        dx = x_global - self.position[0]
        dy = y_global - self.position[1]
        dz = z_global # centre fixé à 0 en z

        # 2. Décalage par rapport au pivot
        dx -= rot_center_3d[0]
        dy -= rot_center_3d[1]
        dz -= rot_center_3d[2]

        # 3. Rotation inverse autour de Y
        c = np.cos(-thetay)
        s = np.sin(-thetay)

        x_local = c * dx + s * dz + rot_center_3d[0]
        y_local = dy + rot_center_3d[1]
        z_local = -s * dx + c * dz + rot_center_3d[2]

        return x_local, y_local, z_local

    @abstractmethod
    def SDF_local(self):
        pass

    @abstractmethod
    def xy_local(self):
        # Must return the polyline exterior
        pass

    @abstractmethod
    def SDF3D_local(self):
        pass

    @abstractmethod
    def get_rot_center(self, center=None):
        pass

    def polygon(self):
        contour = list(zip(*self.xy_local()))
        base_poly = Polygon(contour)
        return translate(
            rotate(base_poly, self.rotation, origin=tuple(self.rot_center), use_radians=True),
            xoff=self.position[0], yoff=self.position[1]
        )

    def BB(self, dim=2, z="rev"):
        """
        Return the Boundig Box of the object in 2 or 3 dimmensions.
        :param dim: Number of dimmensions (2 or 3).
        :param z: "rev" by default. Rev if revolution along x-axis (roller) or const for constant z.
        :return:
        """
        assert z in ["rev", "const"]
        poly = self.polygon()
        xmin, ymin, xmax, ymax = poly.bounds
        if dim == 2:
            return xmin, ymin, xmax, ymax
        elif dim == 3:
            if z=="rev":
                # z = y because of revolution on x axis
                return xmin, ymin, ymin, xmax, ymax, ymax
            else:
                raise Warning("If z set to const, choose zmin zmax manually (can be infinite)")
        else:
            raise AssertionError("Dimension not possible")

    def SDF(self, x, y, real=False):
        # t0=time.time()
        x_local, y_local = self.world_to_local(x, y)
        if real:
            sdf_local = self.realSDF_local(x_local, y_local)
        else:
            sdf_local = self.SDF_local(x_local, y_local)
        # print("SDF time calcul:", time.time() - t0)
        return sdf_local

    def SDF2(self, x, y, real=False):
        """
        Calcule le SDF en coordonnées globales. Utilise la grille SDF mise en cache si disponible.
        """
        x_local, y_local = self.world_to_local(x, y)

        if real != self.sdf_real_or_not:
            print("(Part.SDF2) Warning: real arg is different from cache sdf --> SDF recomputed.")
            self.cache_sdf_grid(real=real)


        # Récupération des paramètres de la grille
        x_min_local, y_min_local = self.sdf_local_origin
        res = self.sdf_resolution
        grid_shape = self.sdf_grid.shape  # (Ny, Nx)
        max_i = grid_shape[1] - 1  # Index max en X
        max_j = grid_shape[0] - 1  # Index max en Y
        # Calcul des indices de la grille (Approche du plus proche voisin)
        # x_local et y_local sont déjà des numpy arrays (scalaires ou arrays)
        i_float = (x_local - x_min_local) / res
        j_float = (y_local - y_min_local) / res
        i = np.round(i_float).astype(int)
        j = np.round(j_float).astype(int)
        # Vérification des limites
        is_oob = (i < 0) | (i > max_i) | (j < 0) | (j > max_j)
        # Initialisation du tableau de résultats
        if np.isscalar(x):
            # Si l'entrée est scalaire, is_oob est aussi scalaire
            if not is_oob:
                print("SCALAR IN BOUNDS")
                # Dans les limites: lookup
                return self.sdf_grid[j.item(), i.item()]
            else:
                # Hors limites: calcul
                print("SCALAR OUT OF BOUNDS")
                return self.SDF(x_local, y_local, real=True).item()
        else:
            # Si l'entrée est un array, on utilise un mix de cache et de calcul
            sdf_result = np.full(x_local.shape[0], 10.0, dtype=float)
            # 1. Remplissage des valeurs dans les limites (Lookup)
            in_bounds = np.logical_not(is_oob)
            if np.any(in_bounds):
                i_in = i[in_bounds]
                j_in = j[in_bounds]
                # La grille est indexée [y_index, x_index]
                sdf_result[in_bounds] = self.sdf_grid[j_in, i_in]
            return sdf_result

    def SDF3D(self, x, y, z, theta=0):
        xl, yl, zl = self.world_to_local_3D(x, y, z, thetay=theta)
        return self.SDF3D_local(xl, yl, zl)

    def cache_sdf_grid(self, resolution=0.05, real=True, silent=True):
        """
        Calcule le SDF_local sur une grille définie et le stocke dans self.sdf_grid.
        :param resolution: Pas d'échantillonnage de la grille.
        :param real: SAF if false, SDF if true.
        """
        xmin_local, ymin_local, xmax_local, ymax_local = -L, -15, L, 15
        self.sdf_real_or_not = real
        self.sdf_resolution = resolution
        self.sdf_local_origin = np.array([xmin_local, ymin_local])

        # Crée la grille de coordonnées locales
        x_local_1d = np.arange(xmin_local, xmax_local + resolution / 2, resolution)
        y_local_1d = np.arange(ymin_local, ymax_local + resolution / 2, resolution)

        X_local, Y_local = np.meshgrid(x_local_1d, y_local_1d)

        # Calcule les valeurs SDF
        try:
            if real:
                sdf_values = self.realSDF_local(X_local, Y_local)
            else:
                sdf_values = self.SDF_local(X_local, Y_local)
            self.sdf_grid = sdf_values
            print(f"SDF grid cached successfully (shape: {self.sdf_grid.shape})") if not silent else None
        except Exception as e:
            self.sdf_grid = None
            print(f"Error during SDF grid caching: {e}") if not silent else None

    def render(self, ax=None, show=True, plot_rotcenter=False, color=0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        poly = self.polygon()
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black")
        ax.fill(x, y, color=COLORS[color], alpha=0.7, label=self.__str__())
        if plot_rotcenter:
            ax.scatter(self.rot_center[0], self.rot_center[1], c="k", marker="o", s=100, alpha=0.7)
            ax.grid(True)
        ax.set_aspect("equal")
        if show:
            plt.show()

class Roller(Part):
    def __init__(self):
        super().__init__()
        self.fcsketch = None
        self.Rxp = [[], []]  # [X, Y]
        self.Rxf = None  # Function

        self.rmid = 3.0  # Radius at the beginning of the function
        self.mid_width = L / 2
        self.clearance = CLEARANCE

        self.rfinal = None
        self.rmax = None

        self.rb_parameters = None  # To access to dimensions

    def importSketch(self, sketch):
        self.fcsketch = sketch

        c1, c2 = self.c()
        revolution_line_index = None
        border_line_index = None

        ymax_sketch = -float('inf')
        shapes = list()
        for i, geo in enumerate(self.fcsketch.Geometry):
            if self.fcsketch.getConstruction(i):
                continue  # Ignorer les lignes de construction

            shape_element = Element(geo.toShape(), plane=(c1, c2))

            if shape_element.get("xmax") <= 0:
                continue

            if geo.TypeId.endswith("LineSegment"):
                if (abs(shape_element.get("ymin") - shape_element.get("ymax")) < 1e-5
                        and abs(shape_element.get("xmin") + self.mid_width) < 1e-5
                        and abs(shape_element.get("xmax") - self.mid_width) < 1e-5
                        and shape_element.get("ymax") > ymax_sketch):  # Revolution line
                    revolution_line_index = len(shapes)
                    ymax_sketch = shape_element.get("ymax")

                elif (abs(shape_element.get("xmin") - shape_element.get("xmax")) < 1e-5
                      and abs(shape_element.get("xmax") - self.mid_width) < 1e-5):  # Border line
                    border_line_index = len(shapes)

            shapes.append(shape_element)

        if border_line_index is None or revolution_line_index is None:
            raise Warning("Borderline or Revolution line not detected")

        for i, shape_element in enumerate(shapes):
            if i == border_line_index or i == revolution_line_index:
                continue
            curve = shape_element.fcshape
            parameters = np.linspace(curve.FirstParameter, curve.LastParameter, 100)
            points = [curve.valueAt(t) for t in parameters]
            X = [p[c1] for p in points]
            Y = [p[c2] - ymax_sketch + self.rmid for p in points]

            if shape_element.get("xmin") < 0 and shape_element.get("xmax") > 0:
                is_increasing = X[1] > X[0]
                null_point = find_first_positive_index(X, is_increasing)
                if is_increasing:
                    X = X[null_point:]
                    Y = Y[null_point:]
                else:
                    X = X[:null_point]
                    Y = Y[:null_point]

            self.Rxp[0].extend(X)
            self.Rxp[1].extend(Y)

        # Convertir les points Rxp en tableau numpy
        X = np.array(self.Rxp[0])
        Y = np.array(self.Rxp[1])
        indices = np.argsort(X)
        X_sorted = X[indices]
        Y_sorted = Y[indices]
        self.Rxp = np.array([X_sorted, Y_sorted])

        # Étape 1 : trier par X
        sorted_indices = np.argsort(X)
        X = X[sorted_indices]
        Y = Y[sorted_indices]

        # Étape 2 : supprimer les points trop proches (distance < tolérance)
        tol = 1e-6
        dx = np.diff(X)
        dy = np.diff(Y)
        dist2 = dx ** 2 + dy ** 2
        mask = np.concatenate(([True], dist2 > tol))  # garde le premier point + ceux éloignés

        x_clean = X[mask]
        y_clean = Y[mask]

        # Étape 4 : interpolation finale
        self.Rxf = interp1d(x_clean, y_clean, fill_value='extrapolate', assume_sorted=True)
        X = np.linspace(0, self.mid_width, 100)
        self.Rxp = [X, self.Rxf(X)]
        self.rmax = -min(self.Rxp[1])
        self.rfinal = -self.Rxp[1][-1]

    def rminmax(self):
        rmin = self.rmid - np.max(self.Rxp[1])
        rmax = self.rmid - np.min(self.Rxp[1])
        return rmin, rmax

    def c(self):
        if not self.fcsketch:
            raise ValueError("Aucune esquisse importée")

        # Déduction automatique du plan (XY, XZ, YZ)
        support = self.fcsketch.Support if hasattr(self.fcsketch, 'Support') else self.fcsketch.AttachmentSupport
        if not support:
            print('Support non trouvé')
            return 0, 1
        plane = support[0][0].Name[:2]
        c1, c2 = {"X": 0, "Y": 1, "Z": 2}[plane[0]], {"X": 0, "Y": 1, "Z": 2}[plane[1]]
        return c1, c2

    def plot(self):
        if not self.fcsketch:
            raise ValueError("Aucune esquisse importée")

        # Déduction automatique du plan (XY, XZ, YZ)
        support = self.fcsketch.Support if hasattr(self.fcsketch, 'Support') else self.fcsketch.AttachmentSupport
        if not support:
            raise ValueError("Impossible de détermider le plan d'attachement")
        plane = support[0][0].Name[:2]
        c1, c2 = {"X": 0, "Y": 1, "Z": 2}[plane[0]], {"X": 0, "Y": 1, "Z": 2}[plane[1]]

        for i, geo in enumerate(self.fcsketch.Geometry):
            if self.fcsketch.getConstruction(i):
                continue  # Ignorer les lignes de construction

            if geo.TypeId.endswith("LineSegment"):
                X = [geo.StartPoint[c1], geo.EndPoint[c1]]
                Y = [geo.StartPoint[c2], geo.EndPoint[c2]]
                plt.plot(X, Y, 'b')

            elif geo.TypeId[10:] == "Circle" or geo.TypeId[10:] == "ArcOfCircle":
                center = geo.Center
                radius = geo.Radius

                if geo.TypeId[10:] == "Circle":
                    angles = np.linspace(0, 2 * np.pi, 100)
                else:
                    # ArcOfCircle : trouver les angles de début et fin
                    v1 = geo.StartPoint.sub(geo.Center)
                    v2 = geo.EndPoint.sub(geo.Center)
                    angle1 = np.arctan2(v1[c2], v1[c1])
                    angle2 = np.arctan2(v2[c2], v2[c1])

                    # Assurer un balayage dans le bon sens
                    if angle2 < angle1:
                        angle2 += 2 * np.pi
                    angles = np.linspace(angle1, angle2, 100)

                X = center[c1] + radius * np.cos(angles)
                Y = center[c2] + radius * np.sin(angles)
                plt.plot(X, Y, 'g')

            elif geo.TypeId.endswith("BSplineCurve"):
                poles = geo.getPoles()
                # Tracer les points de contrôle
                Xp = [p[c1] for p in poles]
                Yp = [p[c2] for p in poles]
                plt.plot(Xp, Yp, 'ro--', alpha=0.5, label='Points de contrôle' if i == 0 else "")

                # Tracer la vraie courbe BSpline interpolée
                curve = geo.toShape()
                parameters = np.linspace(curve.FirstParameter, curve.LastParameter, 100)
                points = [curve.valueAt(t) for t in parameters]
                X = [p[c1] for p in points]
                Y = [p[c2] for p in points]
                plt.plot(X, Y, 'r', label='BSpline' if i == 0 else "")

            else:
                print(f"[INFO] Type non supporté : {geo.TypeId}")

        plt.axis('equal')
        plt.title(f"Sketch on {plane} plane")
        plt.xlabel(plane[0])
        plt.ylabel(plane[1])
        plt.grid(True)
        plt.show()

    def set_Rxf(self, y_ctrl):
        if type(y_ctrl) == np.ndarray:
            self.Rxf = make_spline_func(y_ctrl, self.mid_width)
        else:
            self.Rxf = y_ctrl
        X = np.linspace(0, self.mid_width, 100)
        self.Rxp = [X, self.Rxf(X)]
        self.rmax = -min(self.Rxp[1])
        self.rfinal = -self.Rxp[1][-1]

    def get_Rxf(self, x, offset=0.0, masked=False):
        x = np.asarray(x)
        x_abs = np.abs(x)

        if masked:
            result = np.zeros_like(x_abs, dtype=float)  # par défaut 0
            # masque des valeurs valides
            mask = (x_abs >= -self.mid_width) & (x_abs <= self.mid_width)
            # on remplit seulement sur le masque
            result[mask] = self.Rxf(x_abs[mask]) - offset
        else:
            result = self.Rxf(x_abs) - offset

        # si scalaire, on rend scalaire
        return result.item() if result.size == 1 else result

    def get_Rxp(self, offset=0.0):
        x, y = self.Rxp
        return x, y - offset

    def SDF_local(self, x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        x_abs = np.abs(x)

        # Cas où x <= mid_width
        mask_inner = x_abs <= self.mid_width
        sdf = np.empty_like(x_abs, dtype=float)

        # Partie intérieure : appel vectorisé à get_Rxf
        if np.any(mask_inner):
            fx_inner = self.get_Rxf(x_abs[mask_inner], offset=self.rmid)
            # fx_inner = np.vectorize(lambda xi: self.get_Rxf(xi, offset=self.rmid))(x_abs[mask_inner])
            y_inner = y[mask_inner]
            sdf_inner = np.where(y_inner >= 0, y_inner + fx_inner, -y_inner + fx_inner)
            sdf[mask_inner] = sdf_inner

        # Partie extérieure
        if np.any(~mask_inner):
            sdf_outer = self._sdf_vertical_segment(x_abs[~mask_inner], y[~mask_inner])
            sdf[~mask_inner] = sdf_outer

        # Si entrée était scalaire, sortie aussi
        return sdf.item() if np.isscalar(x) else sdf

    def _sdf_vertical_segment(self, x, y, y_min=None, y_max=None):
        x = np.asarray(x)
        y = np.asarray(y)

        if y_min is None or y_max is None:
            y0 = self.get_Rxf(self.mid_width, offset=self.rmid)
            y_min = y0
            y_max = -y0

        y_clamped = np.clip(y, y_min, y_max)
        x0 = self.mid_width

        dx = x - x0
        dy = y - y_clamped
        distance = np.sqrt(dx * dx + dy * dy)  # Sometime overflow encountered I dont know why.
        sign = np.where(x < x0, -1.0, 1.0)
        return sign * distance

    def xy_local(self):
        x1, y1 = self.get_Rxp(offset=self.rmid) # Tronçon en bas à droite
        x3, y3 = list(reversed(x1)), list(reversed(-y1))  # Tronçon haut droite
        x4, y4 = -x1, -y1  # Tronçon haut gauche
        x6, y6 = list(reversed(-x1)), list(reversed(y1)) # Tronçon en bas à gauche

        xf = np.concatenate((x1, x3, x4, x6))
        yf = np.concatenate((y1, y3, y4, y6))
        return xf, yf

    def get_rot_center(self, center="0"):
        if center == "0":
            rot_center = np.array([0.0, 0.0])
        elif center == "local":
            rot_center = np.array([0.0, 0.0])
        elif center == "bearing":
            rot_center = np.array([0.0, - self.rb_parameters["RBmax"] - self.rb_parameters["m"] - self.rb_parameters["Rin"]])
        else:
            raise AssertionError(center, " must be either '0', 'bearing or 'local'")
        return rot_center

    def SDF3D_local(self, x, y, z):
        r = np.sqrt(y ** 2 + z ** 2)
        return self.SDF_local(x, r)

    def save(self, filename: str):
        """Sauvegarde l'objet Part en JSON (sans la fonction, mais avec les points)."""
        data = {
            "position": self.position.tolist(),
            "rotation": self.rotation,
            "Rxp": self.Rxp,
            "rmid": self.rmid,
            "mid_width": self.mid_width,
            "clearance": self.clearance,
            "rfinal": float(self.rfinal),
            "rmax": float(self.rmax),
            "color": self.color,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename: str):
        """Charge un objet Part depuis un JSON et reconstruit l’interpolateur Rxf."""
        with open(filename, "r") as f:
            data = json.load(f)

        roller = cls()
        roller.position = np.array(data["position"])
        roller.rotation = data["rotation"]
        roller.Rxp = [np.array(data["Rxp"][0]), np.array(data["Rxp"][1])]

        # Recrée l'interpolateur à partir des points s'il y en a
        if roller.Rxp and len(roller.Rxp[0]) > 1:
            roller.Rxf = interp1d(roller.Rxp[0], roller.Rxp[1], fill_value='extrapolate', assume_sorted=True)

        roller.rmid = data["rmid"]
        roller.mid_width = data["mid_width"]
        roller.clearance = data["clearance"]
        roller.rfinal = data["rfinal"]
        roller.rmax = data["rmax"]
        roller.color = data["color"]

        return roller

    def realSDF_local(self, x, y, nt=300, chunk=5000):
        """
        Calcul du SDF (distance minimale) entre (x,y) et la courbe y = R(t), t in [0, mid_width],
        en tenant compte d'un éventuel segment vertical en x = mid_width.
        - nt : nombre de points sur la grille en t (augmenter pour plus de précision).
        - chunk : taille de chunk pour éviter d'allouer trop de mémoire quand il y a beaucoup de points.
        Retourne un scalaire si (x,y) scalaires, sinon un tableau de même forme que x, y.
        """
        # Normalisation des entrées
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        scalar_input = (x_arr.shape == ()) and (y_arr.shape == ())

        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()
        x_abs = np.abs(x_flat)
        y_abs = np.abs(y_flat)

        # grille en t (vectorisée)
        Nt = int(nt)
        t_grid = np.linspace(0.0, float(self.mid_width), Nt)
        R_grid = np.asarray(self.get_Rxf(t_grid, offset=self.rmid))  # get_Rxf doit accepter un tableau
        # calcul par "batch" pour limiter la mémoire : pour chaque batch on calcule la distance à tous les t_grid
        dist_min = np.empty_like(x_abs, dtype=float)
        dist_seg = np.empty_like(x_abs, dtype=float)
        dist_inner = np.empty_like(x_abs, dtype=float)

        # argmin_idx si besoin pour debug/raffinement
        chunk = max(1, int(chunk))
        for start in range(0, x_abs.size, chunk):
            stop = min(x_abs.size, start + chunk)
            dx = x_abs[start:stop, None] - t_grid[None, :]        # shape (batch, Nt)
            dy = -y_abs[start:stop, None] - R_grid[None, :]       # shape (batch, Nt)
            d2 = dx*dx + dy*dy                                # shape (batch, Nt)
            idx = np.argmin(d2, axis=1)
            dist_min[start:stop] = -np.sign(dy)[np.arange(d2.shape[0]), idx] * np.sqrt(d2[np.arange(d2.shape[0]), idx])
            dist_seg[start:stop] = np.asarray(self._sdf_vertical_segment(x_abs[start:stop] , y_abs[start:stop]))
            choice = np.abs(dist_min[start:stop]) <= np.abs(dist_seg[start:stop])
            dist_inner[start:stop] = np.where(choice, dist_min[start:stop], dist_seg[start:stop])

        sdf = dist_inner

        # remettre la forme originale et éventuellement retourner scalaire
        sdf = sdf.reshape(x_arr.shape)
        return float(sdf) if scalar_input else sdf

class OuterRing(Part):
    def __init__(self, roller):
        super().__init__()
        self.roller = roller

        self.color = 'grey'

        self.rb_parameters = None  # To access to dimensions

    def xy_local(self):
        if self.rb_parameters is None:
            rout = 8.0
        else:
            rout = self.rb_parameters["Rext"] - (self.rb_parameters["Rin"] + self.rb_parameters["m"] + self.rb_parameters["RBmax"])
        x1, y1 = self.roller.get_Rxp(offset=self.roller.rmid + self.roller.clearance)  # Tronçon en bas à droite
        x3, y3 = list(reversed(x1)), list(reversed(-y1))  # Tronçon haut droite
        x4, y4 = -x1, -y1  # Tronçon haut gauche
        x5, y5 = [-self.roller.mid_width, self.roller.mid_width], [rout] * 2

        xf = np.concatenate((x3, x4, x5))
        yf = np.concatenate((y3, y4, y5))
        return xf, yf

    def SDF_local(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x_abs = np.abs(x)

        # Préparation de la sortie
        sdf = np.empty_like(x_abs, dtype=float)

        # Masques pour détermider les cas
        mask_inner = x_abs <= self.roller.mid_width
        mask_outer = ~mask_inner

        # Partie intérieure
        if np.any(mask_inner):
            fx_inner = self.roller.get_Rxf(x_abs[mask_inner], offset=self.roller.rmid + self.roller.clearance)
            sdf_inner = -y[mask_inner] - fx_inner
            sdf[mask_inner] = sdf_inner

        # Partie extérieure
        if np.any(mask_outer):
            sdf_outer = self.roller._sdf_vertical_segment(
                x_abs[mask_outer],
                y[mask_outer],
                y_min=self.roller.rmid + self.roller.rfinal + self.roller.clearance,
                y_max=float('inf')
            )
            sdf[mask_outer] = sdf_outer

        return sdf.item() if np.isscalar(x) else sdf

    def realSDF_local(self, x, y, nt=300, chunk=5000):
        """
        Calcul du SDF (distance minimale) entre (x,y) et la courbe y = R(t), t in [0, mid_width],
        en tenant compte d'un éventuel segment vertical en x = mid_width.
        - nt : nombre de points sur la grille en t (augmenter pour plus de précision).
        - chunk : taille de chunk pour éviter d'allouer trop de mémoire quand il y a beaucoup de points.
        Retourne un scalaire si (x,y) scalaires, sinon un tableau de même forme que x, y.
        """
        # Normalisation des entrées
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        scalar_input = (x_arr.shape == ()) and (y_arr.shape == ())

        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()
        x_abs = np.abs(x_flat)
        y_abs = y_flat

        # grille en t (vectorisée)
        Nt = int(nt)
        t_grid = np.linspace(0.0, float(self.roller.mid_width), Nt)
        R_grid = np.asarray(self.roller.get_Rxf(t_grid, offset=self.roller.rmid + self.roller.clearance))  # get_Rxf doit accepter un tableau
        # calcul par "batch" pour limiter la mémoire : pour chaque batch on calcule la distance à tous les t_grid
        dist_min = np.empty_like(x_abs, dtype=float)
        dist_seg = np.empty_like(x_abs, dtype=float)
        dist_inner = np.empty_like(x_abs, dtype=float)

        # argmin_idx si besoin pour debug/raffinement
        chunk = max(1, int(chunk))
        for start in range(0, x_abs.size, chunk):
            stop = min(x_abs.size, start + chunk)
            dx = x_abs[start:stop, None] - t_grid[None, :]        # shape (batch, Nt)
            dy = -y_abs[start:stop, None] - R_grid[None, :]       # shape (batch, Nt)
            d2 = dx*dx + dy*dy                                # shape (batch, Nt)
            idx = np.argmin(d2, axis=1)
            dist_min[start:stop] = np.sign(dy)[np.arange(d2.shape[0]), idx] * np.sqrt(d2[np.arange(d2.shape[0]), idx])
            dist_seg[start:stop] = np.asarray(self.roller._sdf_vertical_segment(x_abs[start:stop],
                                                                                y_abs[start:stop],
                                                                                y_min=self.roller.rmid + self.roller.rfinal + self.roller.clearance,
                                                                                y_max=float('inf')))
            choice = np.abs(dist_min[start:stop]) <= np.abs(dist_seg[start:stop])
            dist_inner[start:stop] = np.where(choice, dist_min[start:stop], dist_seg[start:stop])

        sdf = dist_inner

        # remettre la forme originale et éventuellement retourner scalaire
        sdf = sdf.reshape(x_arr.shape)
        return float(sdf) if scalar_input else sdf

    def SDF3D_local(self, x, y, z):
        # Hypothesis of infinite radius
        return self.SDF_local(x, y)

    def get_rot_center(self, center="bearing"):
        if center == "0":
            rot_center = np.array([0.0, 0.0])
        elif center == "local":
            rout = self.rb_parameters["Rext"] - (
                        self.rb_parameters["Rin"] + self.rb_parameters["m"] + self.rb_parameters["RBmax"])
            rminout = self.rb_parameters["RBmin"] + self.rb_parameters["m"]
            rot_center = np.array([0.0, (rminout + rout) / 2])
        elif center == "bearing":
            rot_center = np.array([0.0, - self.rb_parameters["RBmax"] - self.rb_parameters["m"] - self.rb_parameters["Rin"]])
        else:
            raise AssertionError(center, " must be either '0', 'bearing or 'local'")
        return rot_center

class InnerRing(Part):
    def __init__(self, roller):
        super().__init__()
        self.roller = roller

        self.color = 'grey'

        self.rb_parameters = None  # To access to dimmension for rendering

    def SDF_local(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x_abs = np.abs(x)

        # Préparation de la sortie
        sdf = np.empty_like(x_abs, dtype=float)

        # Masques pour détermider les cas
        mask_inner = x_abs <= self.roller.mid_width
        mask_outer = ~mask_inner

        # Partie intérieure
        if np.any(mask_inner):
            fx_inner = self.roller.get_Rxf(x_abs[mask_inner], offset=self.roller.rmid + self.roller.clearance)
            sdf_inner = y[mask_inner] - fx_inner
            sdf[mask_inner] = sdf_inner

        # Partie extérieure
        if np.any(mask_outer):
            sdf_outer = self.roller._sdf_vertical_segment(
                x_abs[mask_outer],
                y[mask_outer],
                y_min=-float('inf'),
                y_max=-self.roller.rmid - self.roller.rfinal - self.roller.clearance
            )
            sdf[mask_outer] = sdf_outer

        return sdf.item() if np.isscalar(x) else sdf

    def xy_local(self):
        untilrshaft = self.rb_parameters["RBmax"] + self.rb_parameters["m"] + self.rb_parameters["Rin"] - self.rb_parameters["Rshaft"]
        x1, y1 = self.roller.get_Rxp(offset=self.roller.rmid + self.roller.clearance)  # Tronçon en bas à droite
        x2, y2 = [self.roller.mid_width, -self.roller.mid_width], [-untilrshaft] * 2
        x6, y6 = list(reversed(-x1)), list(reversed(y1)) # Tronçon en bas à gauche

        xf = np.concatenate((x1, x2, x6))
        yf = np.concatenate((y1, y2, y6))
        return xf, yf

    def SDF3D_local(self, x, y, z):
        # Hypothesis of infinite radius
        # Could be improved by taking into account Rin.
        return self.SDF_local(x, y)

    def realSDF_local(self, x, y, nt=300, chunk=5000):
        """
        Calcul du SDF (distance minimale) entre (x,y) et la courbe y = R(t), t in [0, mid_width],
        en tenant compte d'un éventuel segment vertical en x = mid_width.
        - nt : nombre de points sur la grille en t (augmenter pour plus de précision).
        - chunk : taille de chunk pour éviter d'allouer trop de mémoire quand il y a beaucoup de points.
        Retourne un scalaire si (x,y) scalaires, sinon un tableau de même forme que x, y.
        """
        # Normalisation des entrées
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        scalar_input = (x_arr.shape == ()) and (y_arr.shape == ())

        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()
        x_abs = np.abs(x_flat)
        y_abs = -y_flat # np.abs(y_flat)

        # grille en t (vectorisée)
        Nt = int(nt)
        t_grid = np.linspace(0.0, float(self.roller.mid_width), Nt)
        R_grid = np.asarray(self.roller.get_Rxf(t_grid, offset=self.roller.rmid + self.roller.clearance))  # get_Rxf doit accepter un tableau
        # calcul par "batch" pour limiter la mémoire : pour chaque batch on calcule la distance à tous les t_grid
        dist_min = np.empty_like(x_abs, dtype=float)
        dist_seg = np.empty_like(x_abs, dtype=float)
        dist_inner = np.empty_like(x_abs, dtype=float)

        # argmin_idx si besoin pour debug/raffinement
        chunk = max(1, int(chunk))
        for start in range(0, x_abs.size, chunk):
            stop = min(x_abs.size, start + chunk)
            dx = x_abs[start:stop, None] - t_grid[None, :]        # shape (batch, Nt)
            dy = -y_abs[start:stop, None] - R_grid[None, :]       # shape (batch, Nt)
            d2 = dx*dx + dy*dy                                # shape (batch, Nt)
            idx = np.argmin(d2, axis=1)
            dist_min[start:stop] = np.sign(dy)[np.arange(d2.shape[0]), idx] * np.sqrt(d2[np.arange(d2.shape[0]), idx])
            dist_seg[start:stop] = np.asarray(self.roller._sdf_vertical_segment(x_abs[start:stop],
                                                                                y_abs[start:stop],
                                                                                y_min=self.roller.rmid + self.roller.rfinal + self.roller.clearance,
                                                                                y_max=float('inf')))
            choice = np.abs(dist_min[start:stop]) <= np.abs(dist_seg[start:stop])
            dist_inner[start:stop] = np.where(choice, dist_min[start:stop], dist_seg[start:stop])

        sdf = dist_inner

        # remettre la forme originale et éventuellement retourner scalaire
        sdf = sdf.reshape(x_arr.shape)
        return float(sdf) if scalar_input else sdf

    def get_rot_center(self, center="bearing"):
        if center == "0":
            rot_center = np.array([0.0, 0.0])
        elif center == "local":
            rmaxin = self.rb_parameters["RBmax"] + self.rb_parameters["m"] + self.rb_parameters["Rin"] - self.rb_parameters["Rshaft"]
            rminin = self.rb_parameters["RBmin"] + self.rb_parameters["m"]
            rot_center = np.array([0.0, -(rmaxin + rminin) / 2])
        elif center == "bearing":
            rot_center = np.array([0.0, - self.rb_parameters["RBmax"] - self.rb_parameters["m"] - self.rb_parameters["Rin"]])
        else:
            raise AssertionError(center, " must be either '0', 'bearing or 'local'")
        return rot_center

class ZpToYp:
    def __init__(self, f, t, z_interval, npts=200):
        self.f = f
        self.t = t
        a, b = z_interval
        z_grid = np.linspace(a, b, npts)  # Z original

        # grilles
        self.zp_grid = np.cos(t) * z_grid - np.sin(t) * f(z_grid)
        self.yp_grid = np.sin(t) * z_grid + np.cos(t) * f(z_grid)

        # inversion zp -> yp
        self.yp_from_zp = interp1d(self.zp_grid, self.yp_grid, kind="cubic", fill_value="extrapolate")

    def __call__(self, zp_target):
        zp_target = np.asarray(zp_target)
        result = np.zeros_like(zp_target, dtype=float)  # par défaut 0
        # masque des valeurs valides
        mask = (zp_target >= self.zp_grid.min()) & (zp_target <= self.zp_grid.max())
        # on remplit seulement sur le masque
        result[mask] = self.yp_from_zp(zp_target[mask])

        return result

    def plot(self, **kwargs):
        """Trace la courbe réelle (zp(z), yp(z)) sans passer par l’inverse."""
        plt.plot(self.zp_grid, self.yp_grid, **kwargs)

class YpInterpolator:
    def __init__(self, f, t_values, z_interval, npts=2000):
        """
        f        : fonction f(z)
        t_values : liste/array des valeurs de t échantillonnées
        z_interval : (a,b) intervalle pour z
        npts     : nb de points par mapper
        """
        self.t_values = np.array(t_values)
        self.mappers = [ZpToYp(f, t, z_interval, npts=npts) for t in t_values]

    def __call__(self, t, zp):
        """
        Retourne yp(t, zp) par interpolation linéaire entre les deux mappers
        encadrant t.
        """
        # cas hors bornes → extrapolation
        if t <= self.t_values[0]:
            return self.mappers[0](zp)
        if t >= self.t_values[-1]:
            return self.mappers[-1](zp)

        # trouve l’index de l’intervalle (bisect_right donne le 1er t > t)
        j = bisect_right(self.t_values, t)
        i = j - 1

        t1, t2 = self.t_values[i], self.t_values[j]
        m1, m2 = self.mappers[i], self.mappers[j]

        # interpolation linéaire en t
        alpha = (t - t1) / (t2 - t1)
        yp1, yp2 = m1(zp), m2(zp)
        return (1 - alpha) * yp1 + alpha * yp2

class BearingSimulation:
    def __init__(self, roller, iring, oring, rext=None, rshaft=None):
        self.roller = roller  # shape 2
        self.iring = iring  # shape 3
        self.oring = oring  # shape 1
        self.parameters = dict()
        self.score_weight = {"disloc_score": 1.0, "block_score": 1.0, "thetay0": 1.0, "fbb_score": 1.0}

        # DISLOCATION
        self.x_iring, self.theta_iring = 0, 0
        self.pos_roller = []  # x, y, t
        self.d_storage = {"grad": [], "params": [], "energy": [], "pos_iring": []}
        self.acc_grid = 200

        # INNER ROTATION
        self.r_storage = {"thetay": [], "energy": []}

        self.b_istep = None

        # PARAMETERS PROPAGATION
        self.get_parameters(rext=rext, rshaft=rshaft)
        self.iring.rb_parameters = self.parameters
        self.oring.rb_parameters = self.parameters
        self.roller.rb_parameters = self.parameters

    def __str__(self):
        return "| rminmax:" + str([round(float(value), 2) for value in self.roller.rminmax()]) + " | " + str(self.parameters)
        # return "RB | m:" + str(self.roller.clearance) + " | " + str(len(self.storage["energy"])) + "dis. & " + str(len(self.r_storage["energy"])) + "rot. steps |"

    # ROLLER BREAKING METHODS
    def br_score(self, F_ext=10, z_uniform=True, E_star=3e9):
        """External radial force of 10N."""
        Nb = self.parameters["Nb"]
        Fmax = F_ext / (2 * Nb / np.pi)

        if z_uniform:  # If the slope is flat: can be approximated by Fmax/L
            Qmax = Fmax / L
            pomax = np.sqrt((Qmax * E_star) / (np.pi * self.parameters["RBmin"]))
        else:  # If the slope is not flat like spheres, explanation readerbook p.288
            # Get list of uy.ni
            uy = np.array([0, 1])
            rxp = self.roller.get_Rxp(offset=0)
            ks = list()
            for i in range(len(rxp[0]) - 1):
                pt1 = np.array([rxp[0][i], rxp[1][i]])
                pt2 = np.array([rxp[0][i+1], rxp[1][i+1]])
                v = pt2 - pt1
                ni = np.array([-v[1], v[0]]) / np.linalg.norm(np.array([-v[1], v[0]]))
                ki = np.dot(uy, ni)
                ks.append(ki)
            correct = 1 / 2*sum(ks)  # Readerbook p288

            wz = np.array([Fmax * correct * ki for ki in ks])
            rbz = np.array([r for z, r in list(zip(*self.roller.get_Rxp(offset=self.roller.rmid)))[:-1]])

            poz = np.sqrt((wz * E_star) / (np.pi * np.abs(rbz)))
            pomax = np.max(poz)

        return 1 / (1 + pomax)

    # DISLOCATION METHODS
    def d_reset(self, real=True, res=0.1):
        self.x_iring, self.theta_iring = 0, 0
        self.pos_roller = []  # x, y, t
        self.d_storage["pos_iring"] = []
        self.d_storage["grad"] = []
        self.d_storage["params"] = []
        self.d_storage["energy"] = []

        # Compute SDF of each part
        self.roller.cache_sdf_grid(resolution=res, real=real)
        self.iring.cache_sdf_grid(resolution=res, real=real)
        self.oring.cache_sdf_grid(resolution=res, real=real)

    def d_step(self, x_iring, theta_iring, method="Powell", silent=False):
        if len(self.pos_roller) == 0:
            params = np.array([0.0, 0.0, 0.0])
        else:
            params = np.array(self.pos_roller[-1])

        if silent:
            print("NEW STEP: (previous pos_roller: ", params, ")")
            print("Iring parameters: ", x_iring, theta_iring)

        # Grille de calcul pour oring (fixée une fois pour toutes)
        if not hasattr(self, 'sdf_oring'):
            xs = np.linspace(-2 * self.roller.mid_width, 2 * self.roller.mid_width, self.acc_grid)
            ys = np.linspace(-10.0, 10.0, self.acc_grid)
            self.X, self.Y = np.meshgrid(xs, ys)
            xmin_oring, ymin_oring, xmax_oring, ymax_oring = self.oring.BB()
            mask = (self.X >= xmin_oring) & (self.X <= xmax_oring) & (self.Y >= ymin_oring) & (self.Y <= ymax_oring)
            self.X_oring, self.Y_oring = self.X[mask], self.Y[mask]
            self.sdf_oring = self.oring.SDF(self.X_oring, self.Y_oring)
        else:
            xmin_oring, ymin_oring, xmax_oring, ymax_oring = self.oring.BB()

        # Positionne l’inner ring
        self.iring.set_transform([x_iring, 0], theta_iring, center="bearing")
        self.d_storage["pos_iring"].append([[x_iring, 0], theta_iring])
        xmin_iring, ymin_iring, xmax_iring, ymax_iring = self.iring.BB()
        mask = (self.X >= xmin_iring) & (self.X <= xmax_iring) & (self.Y >= ymin_iring) & (self.Y <= ymax_iring)
        X_iring, Y_iring = self.X[mask], self.Y[mask]
        sdf_iring = self.iring.SDF(X_iring, Y_iring)

        # Contrainte sur les bornes
        bounds = [(min(xmin_iring, xmin_oring), max(xmax_iring, xmax_oring)),
                  (min(ymin_iring, ymin_oring), max(ymax_iring, ymax_oring)),
                  (-np.pi, np.pi)]

        # Énergie à minimiser
        def energy(pose):
            # Clip dans les bornes
            x = np.clip(pose[0], bounds[0][0], bounds[0][1])
            y = np.clip(pose[1], bounds[1][0], bounds[1][1])
            theta = np.clip(pose[2], bounds[2][0], bounds[2][1])

            self.roller.set_transform([x, y], theta)
            sdf_roller_iring = self.roller.SDF(X_iring, Y_iring)
            sdf_roller_oring = self.roller.SDF(self.X_oring, self.Y_oring)
            sdf_inter_iring = np.maximum(sdf_roller_iring, sdf_iring)
            sdf_inter_oring = np.maximum(sdf_roller_oring, self.sdf_oring)
            E = -min(np.min(sdf_inter_iring), np.min(sdf_inter_oring))
            # print("     energy eval: pose ", pose, "| E=", E)
            return E


        result = minimize(
            energy,
            params,
            method=method,
            bounds=bounds if method != "Powell" else None,  # Powell ne gère pas les bornes directement
            options={"maxiter": 50, "disp": silent}
        )

        best_params = result.x
        best_energy = result.fun

        self.roller.set_transform([best_params[0], best_params[1]], best_params[2])
        self.pos_roller.append(tuple(best_params))

        self.d_storage["energy"].append([best_energy])
        self.d_storage["params"].append([best_params])
        self.d_storage["grad"].append([None])  # Pas de gradient ici

        if silent:
            print(f"Optimisation termidée - énergie: {best_energy:.4f} - params: {best_params}")

    def d_step2(self, x_iring, theta_iring, method="Powell", silent=False, real=True):
        if len(self.pos_roller) == 0:
            params = np.array([0.0, 0.0, 0.0])
        else:
            params = np.array(self.pos_roller[-1])

        if silent:
            print("NEW STEP: (previous pos_roller: ", params, ")")
            print("Iring parameters: ", x_iring, theta_iring)

        # Grille de calcul pour oring (fixée une fois pour toutes)
        xs = np.linspace(-2 * self.roller.mid_width, 2 * self.roller.mid_width, self.acc_grid)
        ys = np.linspace(-10.0, 10.0, self.acc_grid)
        X, Y = np.meshgrid(xs, ys)

        # Calcul SDF oring
        xmin_oring, ymin_oring, xmax_oring, ymax_oring = self.oring.BB()
        mask = (X >= xmin_oring) & (X <= xmax_oring) & (Y >= ymin_oring) & (Y <= ymax_oring)
        X_oring, Y_oring = X[mask], Y[mask]
        sdf_oring = self.oring.SDF2(X_oring, Y_oring, real=real)

        # Relocate iring
        self.iring.set_transform([x_iring, 0], theta_iring, center="bearing")
        self.d_storage["pos_iring"].append([[x_iring, 0], theta_iring])

        # Calcul SDF iring
        xmin_iring, ymin_iring, xmax_iring, ymax_iring = self.iring.BB()
        mask = (X >= xmin_iring) & (X <= xmax_iring) & (Y >= ymin_iring) & (Y <= ymax_iring)
        X_iring, Y_iring = X[mask], Y[mask]
        sdf_iring = self.iring.SDF2(X_iring, Y_iring, real=real)

        # Contrainte sur les bornes
        bounds = [(min(xmin_iring, xmin_oring), max(xmax_iring, xmax_oring)),
                  (min(ymin_iring, ymin_oring), max(ymax_iring, ymax_oring)),
                  (-np.pi, np.pi)]

        # Énergie à minimiser
        def energy(pose):
            # Clip in bounds
            x = np.clip(pose[0], bounds[0][0], bounds[0][1])
            y = np.clip(pose[1], bounds[1][0], bounds[1][1])
            theta = np.clip(pose[2], bounds[2][0], bounds[2][1])

            # Relocate roller
            self.roller.set_transform([x, y], theta)
            if len(X_iring) != 0:
                sdf_roller_iring = self.roller.SDF2(X_iring, Y_iring, real=real)
                sdf_inter_iring = np.maximum(sdf_roller_iring, sdf_iring)
            else:
                sdf_inter_iring = np.array([1.0])

            if len(X_oring) != 0:
                sdf_roller_oring = self.roller.SDF2(X_oring, Y_oring, real=real)
                sdf_inter_oring = np.maximum(sdf_roller_oring, sdf_oring)
            else:
                sdf_inter_oring = np.array([1.0])

            E = -min(np.min(sdf_inter_iring), np.min(sdf_inter_oring))
            # print("     energy eval: pose ", pose, "| E=", E)
            return E


        result = minimize(
            energy,
            params,
            method=method,
            bounds=bounds if method != "Powell" else None,  # Powell ne gère pas les bornes directement
            options={"maxiter": 50, "disp": silent}
        )

        best_params = result.x
        best_energy = result.fun

        self.roller.set_transform([best_params[0], best_params[1]], best_params[2])
        self.pos_roller.append(tuple(best_params))

        self.d_storage["energy"].append([best_energy])
        self.d_storage["params"].append([best_params])
        self.d_storage["grad"].append([None])  # Pas de gradient ici

        if silent:
            print(f"Optimisation termidée - énergie: {best_energy:.4f} - params: {best_params}")

    def d_render(self, show_mimization=False, show_polygon=True, step=-1, ax=None, show=True, clearance_factor=1.0):
        self.roller.clearance *= clearance_factor
        poses = np.array(self.d_storage["params"][step])
        grads = np.array(self.d_storage["grad"][step])
        energies = np.array(self.d_storage["energy"][step])

        if show_mimization:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs = axs.ravel()
            if len(poses) > 0:
                # --- Courbe 1 : Trajectoire dans l'espace (x, y) ---
                axs[0].plot(poses[:, 0], label="x")
                axs[0].plot(poses[:, 1], label="y")
                axs[0].plot(poses[:, 2], label="θ")
                axs[0].set_title("Évolution des paramètres de pose")
                axs[0].set_xlabel("Itération")
                axs[0].set_ylabel("Valeur")
                axs[0].legend()
                axs[0].grid()

                # --- Courbe 2 : Valeurs des gradients ---
                axs[1].plot(grads[:, 0], label="∂E/∂x")
                axs[1].plot(grads[:, 1], label="∂E/∂y")
                axs[1].plot(grads[:, 2], label="∂E/∂θ")
                axs[1].set_title("Évolution du gradient")
                axs[1].set_xlabel("Itération")
                axs[1].set_ylabel("Valeur du gradient")
                axs[1].legend()
                axs[1].grid()

                # --- Courbe 3 : Norme du gradient ---
                grad_norm = np.linalg.norm(grads, axis=1)
                axs[2].plot(grad_norm, label="‖grad‖")
                axs[2].set_title("Norme du gradient")
                axs[2].set_xlabel("Itération")
                axs[2].set_ylabel("‖∇E‖")
                axs[2].legend()
                axs[2].grid()

                # --- Courbe 4 : Énergie ---
                axs[3].plot(energies, label="Énergie", linestyle='--')
                axs[3].set_title("Energy")
                axs[3].set_xlabel("Itération")
                axs[3].set_ylabel("E(pose)")
                axs[3].legend()
                axs[3].grid()
                plt.tight_layout()
                plt.show()
            else:
                axs[0].text(0.5, 0.5, "No iteration")
                axs[1].text(0.5, 0.5, "No iteration")
                axs[2].text(0.5, 0.5, "No iteration")
                axs[3].text(0.5, 0.5, "Energy: " + str(energies[-1]), size="large")

        if show_polygon:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))

            best_pose_index = np.argmin(energies)
            params = self.d_storage["params"][step][best_pose_index]
            xr, yr, tr = params
            pos_roller = [[xr, yr], tr]
            pos_oring = [[0, 0], 0]
            pos_iring = self.d_storage["pos_iring"][step]
            for shape, pos in zip([self.iring, self.roller, self.oring], [pos_iring, pos_roller, pos_oring]):
                if shape in [self.iring, self.oring]:
                    center = "bearing"
                else:
                    center = "0"
                shape.set_transform(*pos, center=center)
                poly = shape.polygon()
                x, y = poly.exterior.xy
                ax.fill(x, y, color=shape.color, alpha=0.7, label=str(shape))
            ax.set_aspect('equal')
            ax.set_axis_off()
            # ax.legend()

            if show:
                plt.show()

            self.roller.clearance /= clearance_factor

    def d_sdf(self, x_iring, y_iring, theta_iring, x_roller, y_roller, theta_roller, bounds=[(-10, 10), (-10, 10)]):
        """Figure SAFintersection in the paper."""
        (xmin, xmax), (ymin, ymax) = bounds
        # Grille de points
        xs = np.linspace(xmin, xmax, 200)
        ys = np.linspace(ymin, ymax, 200)
        X, Y = np.meshgrid(xs, ys)

        sdf_oring = self.oring.SDF(X, Y, real=False)
        xmin_oring, ymin_oring, xmax_oring, ymax_oring = self.oring.BB()

        self.roller.set_transform((x_roller, y_roller), theta_roller, center="0")
        sdf_roller = self.roller.SDF(X, Y, real=False)

        self.iring.set_transform((x_iring, y_iring), theta_iring, center="bearing")
        sdf_iring = self.iring.SDF(X, Y, real=False)
        xmin_iring, ymin_iring, xmax_iring, ymax_iring = self.iring.BB()

        sdf_inter_iring = np.maximum(sdf_roller, sdf_iring)
        sdf_inter_oring = np.maximum(sdf_roller, sdf_oring)
        E = -np.minimum(sdf_inter_iring, sdf_inter_oring)
        # E = -min(np.min(sdf_inter_iring), np.min(sdf_inter_oring))

        def plot_func(values):
            # Création de la figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

            polyoring = self.oring.polygon()
            x, y = polyoring.exterior.xy
            ax.plot(x, y, color="black", lw=2.0)

            polyroller = self.roller.polygon()
            x, y = polyroller.exterior.xy
            ax.plot(x, y, color="black", lw=2.0)

            polyiring = self.iring.polygon()
            x, y = polyiring.exterior.xy
            ax.plot(x, y, color="black", lw=2.0)

            # Remplissage coloré avec contours noirs
            contourf = ax.contourf(X, Y, values, levels=15, cmap="jet")  # jet est cool
            contour = ax.contour(X, Y, values, levels=15, colors='k', linewidths=0.8)
            # ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

            # Barre de couleur verticale
            cbar = plt.colorbar(contourf, ax=ax)

            plt.tight_layout()
            ax.set_aspect('equal')
            plt.show()

        plot_func(E)

    def d_energies(self):
        plt.plot([-p[0][0] for p in self.d_storage["pos_iring"]],
                 [min(ens) for ens in self.d_storage["energy"]],
                 color='k', lw=2.5)
        plt.show()

    def d_score(self, N=5, max_iring_dist=5.0, clearance_factor=2.0, method="Powell", silent=False, real=True, res=0.05):
        self.d_reset(real=real, res=res)
        self.roller.clearance *= clearance_factor
        for x_iring in np.linspace(0, -max_iring_dist, N):
            self.d_step2(x_iring, 0.0, method=method, silent=silent, real=real)
            # self.d_step(x_iring, 0.0, method=method, silent=silent)
        disloc_score = 1 / (10 * max(np.max(self.d_storage["energy"]), 0) + 1.0)
        self.roller.clearance /= clearance_factor
        return disloc_score

    def d_score_rot(self, N=5, max_iring_angle=np.pi/4, clearance_factor=2.0, method="Powell", silent=False):
        self.d_reset()
        self.roller.clearance *= clearance_factor
        for theta_iring in np.linspace(0, max_iring_angle, N):
            self.d_step(0.0, theta_iring, method=method, silent=silent)
        disloc_score = 1 / (10 * max(np.max(self.d_storage["energy"]), 0) + 1.0)
        self.roller.clearance /= clearance_factor
        return disloc_score

    # INNER ROTATION METHODS
    def r_reset(self):
        self.r_storage["energy"] = []
        self.r_storage["thetay"] = []
        self.r_storage["thetay0"] = []
        self.r_storage["equlibrium_metric"] = None
        self.r_storage["dist_btwn_2rollers"] = []
        self.thetay_dist = []

        self.roller.set_transform(np.array([0, 0]), 0.0)
        self.iring.set_transform(np.array([0, 0]), 0.0)

    def r_step(self, theta_roller, silent=True):
        if not silent:
            print("NEW STEP: (theta_roller: ", theta_roller, ")")

        rmin, rmax = self.roller.rminmax()
        xs = np.linspace(-self.roller.mid_width, self.roller.mid_width, self.acc_grid)
        ys = np.linspace(-rmax, 0.0, self.acc_grid)
        zs = np.linspace(-10.0, 10.0, self.acc_grid)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        xmin_iring, ymin_iring, xmax_iring, ymax_iring = self.iring.BB(dim=2)
        zmin_iring, zmax_iring = -self.roller.mid_width, self.roller.mid_width

        xmin_roller, ymin_roller, zmin_roller, xmax_roller, ymax_roller, zmax_roller = self.roller.BB(dim=3, z="rev")
        xmin_inter, xmax_inter = max(xmin_roller, xmin_iring), min(xmax_roller, xmax_iring)
        ymin_inter, ymax_inter = max(ymin_roller, ymin_iring), min(ymax_roller, ymax_iring)
        zmin_inter, zmax_inter = max(zmin_roller, zmin_iring), min(zmax_roller, zmax_iring)
        mask = ((X >= xmin_inter) & (X <= xmax_inter)
                & (Y >= ymin_inter) & (Y <= ymax_inter)
                & (Z >= zmin_inter) & (Z <= zmax_inter))
        X_inter, Y_inter, Z_inter = X[mask], Y[mask], Z[mask]
        sdf_roller = self.roller.SDF3D(X_inter, Y_inter, Z_inter, theta=theta_roller)
        sdf_iring = self.iring.SDF3D(X_inter, Y_inter, Z_inter)

        # Calcul de l'énergie
        sdf_inter_iring = np.maximum(sdf_roller, sdf_iring)
        if len(sdf_inter_iring) == 0:
            E = 0.0
        else:
            E = -np.min(sdf_inter_iring)

        self.r_storage["energy"].append(E)
        self.r_storage["thetay"].append(theta_roller)
        if not silent:
            print("Energy found: ", E)

        return E

    def r_score(self, n_samples=5, tol=1e-2, max_iter=20, clearance_factor=2.0, silent=True):
        self.roller.clearance *= clearance_factor  # To take into account worse case

        # Reset values
        self.r_reset()

        # 1. Recherche grossière
        print("Coarse search") if not silent else None
        thetas = np.linspace(0, np.pi / 2, n_samples)
        prev_theta = thetas[0]

        collision_theta = None
        for t in thetas[1:]:
            val = self.r_step(t, silent=True)
            print("r_E: ", val, " | t: ", t) if not silent else None
            if val >= 0:  # Détection du changement de signe
                theta_low, theta_high = prev_theta, t
                collision_theta = (theta_low, theta_high)
                break
            prev_theta = t

        # 2. Si rien trouvé -> pas de collision
        if collision_theta is None:
            print("Collision not detected") if not silent else None
            self.r_storage["thetay0"] = np.pi / 2
            self.roller.clearance /= 2
            return np.pi / 2

        # 3. Dichotomie pour affiner
        print("Binary search...") if not silent else None
        theta_low, theta_high = collision_theta
        for _ in range(max_iter):
            theta_mid = 0.5 * (theta_low + theta_high)
            val_mid = self.r_step(theta_mid, silent=True)
            print("r_E: ", val_mid, " | t: ", theta_mid) if not silent else None
            if val_mid < 0:
                theta_low = theta_mid
            else:
                theta_high = theta_mid
            if abs(theta_high - theta_low) < tol:
                break

        self.r_storage["thetay0"] = theta_high
        self.roller.clearance /= clearance_factor
        return theta_high  # premier thetay où col_value >= 0

    def r_render(self):
        thetas = np.array(self.r_storage["thetay"])
        energies = np.array(self.r_storage["energy"])

        # Tri des points
        order = np.argsort(thetas)
        thetas_sorted = thetas[order]
        energies_sorted = energies[order]

        # Récupérer thetay0
        thetay0 = self.r_storage["thetay0"]

        plt.figure()
        plt.plot(thetas_sorted, energies_sorted, label="Énergie")

        # Ajout du point rouge transparent
        plt.scatter([thetay0], [0], color="red", s=200, alpha=0.6, zorder=5, label="thetay0")

        # Titre avec thetay0
        plt.title(f"Énergie vs thetay (thetay0 = {thetay0:.3f} rad)")

        plt.xlabel("thetay [rad]")
        plt.ylabel("energy")
        plt.legend()
        plt.show()

    # BLOCKING METHODS
    def b_init(self):
        if "thetay0" in self.r_storage and self.r_storage["thetay0"] != np.pi / 2:
            max_theta = self.r_storage["thetay0"]
        else:
            max_theta = np.pi / 4  # To replace with r

        def R(z):
            return -self.roller.get_Rxf(z, offset=self.roller.rmid)

        rmin, rmax = self.roller.rminmax()
        zmax = np.sqrt(self.roller.mid_width**2 + rmax**2)
        z_interval = (-self.roller.mid_width, self.roller.mid_width)
        t_values = np.linspace(0, max_theta, 200)  # discrétisation en t
        b_interpolator = YpInterpolator(R, t_values, z_interval)
        z_grid = np.linspace(-zmax, zmax, 200)

        def dist(t1, t2=None, eps=None, cart=False):
            if t2 is None:
                t1, t2, eps = t1
            if cart:  # Cartesian coordinate tx, ty, phi
                tx, ty, phi = t1, t2, eps
                eps = np.sqrt(tx**2 + ty**2)
                t1 = np.arctan2(ty, tx)
                t2 = phi - t1
            if -t1 < 0:  # mapper(-t, z) = mapper(t, -z) and t MUST BE positive
                d1 = b_interpolator(t1, -(z_grid + eps))
            else:
                d1 = b_interpolator(-t1, +(z_grid + eps))
            if t2 < 0:
                d2 = b_interpolator(-t2, -z_grid)
            else:
                d2 = b_interpolator(t2, z_grid)
            vals = d1 + d2
            return np.max(vals)

        self.b_istep = dist

    def b_hess(self):
        self.b_init()

        # ---- num grad et hess (centrées)
        def numerical_grad(F, x0, eps=1e-4):
            x0 = np.asarray(x0, float)
            grad = np.zeros_like(x0)
            for i in range(len(x0)):
                dx = np.zeros_like(x0)
                dx[i] = eps
                x0pos = x0 + dx
                x0neg = x0 - dx
                grad[i] = (F(*x0pos) - F(*x0neg)) / (2 * eps)
            return grad

        def numerical_hessian(F, x0, eps=1e-4):
            x0 = np.asarray(x0, float)
            n = len(x0)
            H = np.zeros((n, n))
            f0 = F(x0)
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1
                for j in range(i, n):
                    ej = np.zeros(n)
                    ej[j] = 1
                    if i == j:
                        H[i, i] = (F(x0 + eps * ei) - 2 * f0 + F(x0 - eps * ei)) / (eps ** 2)
                    else:
                        H[i, j] = (F(x0 + eps * (ei + ej)) - F(x0 + eps * ei) - F(x0 + eps * ej) + f0) / (eps ** 2)
                        H[j, i] = H[i, j]
            return H

        # point d'intérêt
        x0 = np.array([0.0, 0.0])

        grad0 = numerical_grad(self.b_istep, x0)
        H0 = numerical_hessian(self.b_istep, x0)

        print("grad at 0:", grad0)
        print("Hessian at 0:\n", H0)

        # spectre
        vals, vecs = eigh(H0)  # valeurs propres rèelles (symmetric)
        print("eigenvalues:", vals)
        print("eigenvectors:\n", vecs)

        # test directionnel: v^T H v
        for k in range(len(vals)):
            v = vecs[:, k]
            val_dir = v @ H0 @ v
            print(f"direction {k}, eigenvalue {vals[k]:.6g}, v^T H v = {val_dir:.6g}")

        # verdict heuristique pour gradient flow x' = -grad F(x)
        if np.all(vals > 1e-8):
            print("H positive definite -> local minimum -> stable (for gradient flow).")
        elif np.all(vals < -1e-8):
            print("H negative definite -> local maximum -> unstable (for gradient flow).")
        elif vals[0] * vals[-1] < -1e-12:
            print("H indefinite -> saddle -> unstable (there is at least one unstable direction).")
        else:
            print("Degenerate / nearly singular Hessian -> need higher-order analysis / simulation.")

    def b_hess2(self):
        self.b_init()

        def numerical_hessian(F, x0, eps=1e-4):
            x0 = np.asarray(x0, float)
            n = len(x0)
            H = np.zeros((n, n))
            f0 = F(x0)
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1
                for j in range(i, n):
                    ej = np.zeros(n)
                    ej[j] = 1
                    if i == j:
                        H[i, i] = (F(x0 + eps * ei) - 2 * f0 + F(x0 - eps * ei)) / (eps ** 2)
                    else:
                        H[i, j] = (F(x0 + eps * (ei + ej)) - F(x0 + eps * ei) - F(x0 + eps * ej) + f0) / (eps ** 2)
                        H[j, i] = H[i, j]
            return H

        def unstable_directions_from_hessian(H, tol_eig=1e-12):
            """
            H : 2x2 symmetric Hessian matrix (numpy array)
            retourne :
              - eigvals : valeurs propres (array)
              - eigvecs : vecteurs propres (colonnes) correspondants
              - unstable : list of dicts { 'lambda': val, 'v': vecteur_unitaire, 'vHv': v^T H v }
            """
            # valeurs propres (ascendantes) et vecteurs (colonnes)
            eigvals, eigvecs = eigh(H)
            unstable = []
            for k, lam in enumerate(eigvals):
                if lam < -tol_eig:  # négatif => direction instable pour gradient flow
                    v = eigvecs[:, k]
                    v = v / norm(v)
                    vHv = float(v @ H @ v)  # devrait être lam
                    unstable.append({'lambda': float(lam), 'v': v, 'vHv': vHv})
            return eigvals, eigvecs, unstable

        x0 = np.array([0.0, 0.0, 0.0])
        H0 = numerical_hessian(self.b_istep, x0)
        # print("Hessian at 0:\n", H0)

        eigvals, eigvecs, unstables = unstable_directions_from_hessian(H0)
        # print("eigvals:", eigvals)
        print("unstable directions:", unstables)

    def b_stepflex(self, theta1, theta2, silent=True, tol=1e-3, max_iter=20):
        """Compute minimum x distance for given rotations theta1, theta2, without collision, using binary search."""

        if not silent:
            print(f"NEW STEP: (theta1={theta1}, theta2={theta2})")

        rmin, rmax = self.roller.rminmax()
        dmin, dmax = 2 * rmin, 2 * np.sqrt(rmax**2 + self.roller.mid_width**2) + 0.1

        # Grille de calcul
        xs = np.linspace(-self.roller.mid_width, self.roller.mid_width, self.acc_grid)
        ys = np.linspace(-rmax, 3 * rmax, self.acc_grid)
        X, Y = np.meshgrid(xs, ys)

        # Backup position
        pos_backup = self.roller.position.copy()

        # Roller 1
        self.roller.set_transform(pos_backup, theta1, center="0")
        sdf_roller1 = self.roller.SDF(X, Y)

        # Préparation du binaire sur roller 2
        lo, hi = dmin, dmax
        best_val, best_dist = None, None

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)

            # Roller 2
            self.roller.set_transform(pos_backup + np.array([0.0, mid]), theta2, center=np.array([0.0, mid]))
            sdf_roller2 = self.roller.SDF(X, Y)

            # Remettre à l’état initial (important pour ne pas accumuler les translations)
            self.roller.position = pos_backup

            # Intersection
            sdf_inter = np.maximum(sdf_roller1, sdf_roller2)
            value = np.min(sdf_inter)

            if not silent:
                print(f"dist={mid:.6f}, value={value:.6f}")

            # Test convergence
            if abs(value) < tol:
                best_val, best_dist = value, mid
                break

            # Ajustement bornes
            if value > 0:  # pas encore de contact → il faut descendre plus bas
                hi = mid
            else:  # collision → il faut monter
                lo = mid

            best_val, best_dist = value, mid

        return best_dist, best_val

    def b_functiontraj(self, N=50, alpha=np.pi/4, dofig=False, silent=True, eps=0.15, ax=None, show=False):
        if "thetay0" in self.r_storage and self.r_storage["thetay0"] != np.pi /2:
            max_theta = self.r_storage["thetay0"]
        else:
            max_theta = np.pi / 4  # To replace with r

        traj = [np.cos(alpha), np.sin(alpha)]

        theta_step = max_theta / N
        theta = 0.0
        ds, ts = [], []
        dist, _ = self.b_stepflex(theta, theta, silent=silent)
        dist0 = dist
        maxdist = dist0
        maxthetas = [0, 0]

        for k in range(N):
            ds.append(dist)
            ts.append(theta)
            theta += theta_step
            dist, _ = self.b_stepflex(traj[0] * theta, traj[1] * theta, silent=silent)

            if dist > max(dist0 + eps, maxdist):  # New maximum to update
                maxdist = dist
                maxthetas = [traj[0] * theta, traj[1] * theta]
            if dist < dist0 - eps:  # Found a new local minima
                break
                # pass

        length, depth = math.sqrt(maxthetas[0]**2 + maxthetas[1]**2), maxdist - dist0
        self.r_storage["equilibrium_metric"] = 10 * (0.5 * length + depth)

        ds = np.array(ds)

        self.r_storage["dist_btwn_2rollers"] = ds
        self.r_storage["thetay_dist"] = ts

        if dofig:
            if ax is None:
                fig, ax = plt.subplots()
                ax.set_xlabel("γ(t) [rad]")
                ax.set_ylabel("d(γ(t))")
                # Ligne horizontale au niveau de dist0
                ax.axhline(dist0, color="red", linestyle="--", label="dist0")
                # Bande tolérance ± eps
                ax.axhline(dist0 + eps, color="orange", linestyle=":", label="+eps")
                ax.axhline(dist0 - eps, color="orange", linestyle=":", label="-eps")

            idx_max = np.argmax(ds)
            theta_max = ts[idx_max]
            maxdist = ds[idx_max]

            ax.plot(ts, ds, label="s(θ)", color="black")

            # Segment vertical
            ax.vlines(
                x=theta_max, ymin=dist0, ymax=maxdist,
                color="green", linestyle="-", linewidth=2,
                label="maxdist - dist0"
            )

            # Point rouge au maximum
            ax.scatter([theta_max], [maxdist], color="green", s=60, zorder=5, label="maximum")

            # ax1.legend()
            # fig.tight_layout()

            if show:
                plt.show()
            return ax

        return self.r_storage["equilibrium_metric"]

    def b_functionflex(self, N=20, show=False, axis=(0, 1)):
        if "thetay0" in self.r_storage and self.r_storage["thetay0"] != np.pi / 2:
            max_theta = self.r_storage["thetay0"]
        else:
            max_theta = np.pi / 4  # To replace with r

        # TODO: calculer bien le zclearance
        zclearance = self.roller.clearance

        thetas1 = np.linspace(-max_theta, max_theta, N)
        thetas2 = np.linspace(-max_theta, max_theta, N)
        epsilons = np.linspace(-2*zclearance, 2*zclearance, N)
        params = [thetas1, thetas2, epsilons]
        T1, T2 = np.meshgrid(params[axis[0]], params[axis[1]])

        # Matrice pour stocker les distances
        D = np.zeros_like(T1)

        # Calcul de la distance pour chaque couple (t1, t2)
        for i in range(N):
            for j in range(N):
                # Initialise le vecteur paramétrique à 0.0
                param = [0.0, 0.0, 0.0]
                # Assigne les deux valeurs variables aux bons indices
                param[axis[0]] = T1[i, j]
                param[axis[1]] = T2[i, j]
                # Calcule la distance
                D[i, j] = self.b_istep(param, cart=True)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Surface 3D
            surf = ax.plot_surface(T1, T2, D, cmap="jet", edgecolor="none")

            names = [r"$\theta_1$", r"$\theta_2$", r"$\epsilon$"]
            ax.set_xlabel(names[axis[0]])
            ax.set_ylabel(names[axis[1]])
            ax.set_zlabel("distance")

            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.show()

        return T1, T2, D

    def b_renderflex(self, theta1, theta2, eps=0.0, dist=None, save=False):
        if dist is None:
            # dist, _ = self.b_stepflex(theta1, theta2)
            dist = self.b_istep(theta1, theta2, eps)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Backup position
        pos_backup = self.roller.position.copy()

        self.roller.set_transform(np.array([eps, 0]), theta1, center="0")
        poly1 = self.roller.polygon()
        x, y = poly1.exterior.xy
        ax.plot(x, y, color="black")
        ax.fill(x, y, color=self.roller.color, alpha=0.7, label=self.roller.__str__())
        ax.scatter(eps, 0, s=200, color="red")
        l = np.sin(theta1) * self.roller.mid_width
        ax.plot([-self.roller.mid_width + eps, self.roller.mid_width + eps], [- l, l], '--', color="lightgrey")
        ax.plot([-self.roller.mid_width + eps, self.roller.mid_width + eps], [0, 0], '--', color="lightgrey")

        self.roller.set_transform(np.array([0, dist]), theta2, center="0")
        poly2 = self.roller.polygon()
        x, y = poly2.exterior.xy
        ax.plot(x, y, color="black")
        ax.fill(x, y, color=self.roller.color, alpha=0.7, label=self.roller.__str__())
        ax.scatter(0, dist, s=200, color="red")
        l = np.sin(theta2) * self.roller.mid_width
        ax.plot([-self.roller.mid_width, self.roller.mid_width], [dist - l, dist + l], '--', color="lightgrey")
        ax.plot([-self.roller.mid_width, self.roller.mid_width], [dist, dist], '--', color="lightgrey")

        ax.plot([0, 0], [0, dist], color="black", lw=2.0)

        self.roller.position = pos_backup

        ax.set_aspect('equal')
        ax.set_axis_off()

        if save:
            filename = r"D:\LUCAS\COURS\POSTDOC\Publications\SPM 2026\images\blocking_render"
            plt.savefig(filename + '\\(' + str(round(theta1, 2)) + ','+ str(round(theta2, 2)) +').png', transparent=True)
        plt.show()

    def b_sdf(self, theta1, theta2, dist=None):
        """Only for debug purpose"""
        rmin, rmax = self.roller.rminmax()
        dmin, dmax = 2 * rmin, 2 * np.sqrt(rmax ** 2 + self.roller.mid_width ** 2) + 0.1

        # Grille de calcul
        xs = np.linspace(-self.roller.mid_width, self.roller.mid_width, self.acc_grid)
        ys = np.linspace(-rmax, 3 * rmax, self.acc_grid)
        X, Y = np.meshgrid(xs, ys)

        # Backup position
        pos_backup = self.roller.position.copy()

        # Roller 1
        self.roller.set_transform(pos_backup, theta1, center="0")
        sdf_roller1 = self.roller.SDF(X, Y)

        # Roller 2
        self.roller.rotation = theta2
        self.roller.position = pos_backup + np.array([0.0, dist])
        self.roller.set_transform(pos_backup + np.array([0.0, dist]), theta2, center=np.array([0.0, dist]))
        sdf_roller2 = self.roller.SDF(X, Y)

        # Remettre à l’état initial (important pour ne pas accumuler les translations)
        self.roller.position = pos_backup

        # Intersection
        sdf_inter = np.maximum(sdf_roller1, sdf_roller2)
        # E = -np.min(sdf_inter)

        def plot_func(values):
            # Création de la figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set(xlim=(-self.roller.mid_width, self.roller.mid_width), ylim=(-rmax, 3 * rmax))

            # Remplissage coloré avec contours noirs
            contourf = ax.contourf(X, Y, values, levels=15, cmap="jet")  # jet est cool
            contour = ax.contour(X, Y, values, levels=15, colors='k', linewidths=0.8)
            # ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

            # Barre de couleur verticale
            cbar = plt.colorbar(contourf, ax=ax)

            plt.tight_layout()
            ax.set_aspect('equal')
            plt.show()

        plot_func(sdf_inter)

    def b_score(self, ntraj=4, n_samples=10, eps=0.1, silent=True):
        if "thetay0" in self.r_storage and self.r_storage["thetay0"] != np.pi / 2:
            max_theta = self.r_storage["thetay0"]
        else:
            max_theta = np.pi / 4  # To replace with r

        theta_step = max_theta / n_samples
        theta = 0.0
        dist0, _ = self.b_stepflex(theta, theta, silent=True)
        alpha_traj, deltaE_traj = [], []

        if not silent:
            traj_debug = []

        for itraj in range(ntraj):
            theta = 0.0
            alpha = np.pi * (itraj + 1) / ntraj
            traj = [np.cos(alpha), np.sin(alpha)]
            maxdist = dist0

            if not silent:
                traj_debug.append([])

            while theta < max_theta:
                theta += theta_step
                dist, _ = self.b_stepflex(traj[0] * theta, traj[1] * theta)

                if not silent:
                    traj_debug[itraj].append([theta, dist])

                if dist > max(dist0 + eps, maxdist):  # New maximum to update
                    maxdist = dist
                if dist < dist0 - eps:  # Found a new local minima
                    break

            alpha_traj.append(alpha)
            deltaE_traj.append(maxdist - dist0)

        if not silent:
            e = min(deltaE_traj)
            a = alpha_traj[deltaE_traj.index(e)]
            print("Minimum deltaE found:", e, " for trajectory ", (np.cos(a), np.sin(a)))

        return min(deltaE_traj)

    def b_fastscore(self, n_samples=10, silent=True):
        "Compute global minimum, and if exists, get max of direct trajectory."
        if "thetay0" in self.r_storage and self.r_storage["thetay0"] != np.pi / 2:
            max_theta = self.r_storage["thetay0"]
        else:
            max_theta = np.pi / 4  # To replace with r

        dist0, _ = self.b_stepflex(0.0, 0.0, silent=True)
        mindist, ts = dist0, (0.0, 0.0)
        traj = None

        # Precompute of 3 domain bound:
        for t1, t2 in [(0.0, max_theta), (max_theta, max_theta), (max_theta, 0.0)]:
            dist, _ = self.b_stepflex(t1, t2, silent=True)
            if dist < mindist:
                mindist = dist
                traj = (t1, t2)

        if traj is None:
            return 0.0

        maxdist = dist0
        for i in range(n_samples):
            dist, _ = self.b_stepflex((traj[0] * i) / n_samples, (traj[1] * i) / n_samples)
            if dist > maxdist: maxdist = dist

        return maxdist - dist0

    # FRICTION
    def friction_length(self, eps=0.1):
        """Approximation of friction by number of point between rmax -eps and rmax in Rxp."""
        npts = np.sum((self.roller.Rxp[1] < - self.roller.rmax + eps))
        dx = self.roller.mid_width / len(self.roller.Rxp[0])
        return npts * dx

    def get_parameters(self, Nb=None, rext=None, rshaft=None):
        m = self.roller.clearance
        if hasattr(self.roller, "Rxp"):
            Rb = self.roller.rminmax()[1]  # equal to rmax
        else:
            return

        if Nb is not None:
            Rin = ((Rb + (m/2)) / np.sin(np.pi / Nb)) - m - Rb
            print("Rin constraint not checked, Rin=", Rin)
        else:
            # Assume Rin = 12.0
            Rin = R_SHAFT + T_IN_MIN
            # Compute Nb, round it to superior to get an int
            Nb = int(np.ceil(np.pi / np.arcsin((m / 2 + Rb) / (Rin + m + Rb))))
            # Recompute corrected Rin
            Rin = ((m / 2 + Rb) / np.sin(np.pi / Nb)) - m - Rb

        self.parameters["Rout"] = round(float(Rin + 2 * m + 2 * Rb), 3)  # Internal max radius of outer ring
        self.parameters["Nb"] = Nb
        self.parameters["Rin"] = round(float(Rin), 3)
        self.parameters["m"] = m
        self.parameters["RBmin"], self.parameters["RBmax"] = self.roller.rminmax()

        if rext is not None:
            self.parameters["Rext"] = rext
        if rshaft is not None:
            self.parameters["Rshaft"] = rshaft

        return self.parameters

    # OPTIMISATION
    def constraints(self):
        """
        Vérifie les contraintes sur la spline et renvoie :
        - 0 si tout est OK
        - une liste de pénalités (valeurs > 0) sinon
        """

        rout_constr = 1.0 if self.parameters["Rout"] > R_EXT - T_OUT_MIN else 0.0

        rin_constr = 1.0 if self.parameters["Rin"] < R_SHAFT + T_IN_MIN else 0.0  # 11.5 = 6.5 + 5

        x, y = self.roller.get_Rxp()
        dy = np.diff(y)
        dx = np.diff(x)
        angles = np.abs(np.degrees(np.arctan(dy / dx)))

        max_idx = np.argmax(angles)
        max_angle = angles[max_idx]

        self.parameters["max_angle"] = round(float(max_angle), 3)

        maxslope_contr = 1.0 if max_angle > ALPHA_LIM else 0.0

        # rbmin_constr = 1.0 if np.max(self.roller.Rxp[1]) > 0 else 0.0

        return [rout_constr, rin_constr, maxslope_contr]  # , rbmin_constr]

    def score(self):
        """SCORES TO MINIMIZE (must be between 0 and 1)"""
        # Compute dislocation score
        self.acc_grid = 200
        disloc_score = self.d_score(N=10, clearance_factor=1.0, real=True, res=0.05)
        disloc_score *= self.score_weight["disloc_score"]

        # Compute inner rotation score
        self.acc_grid = 50
        self.r_score(n_samples=5, tol=1e-2, max_iter=20, clearance_factor=2.0)
        # Compute thetay0, the angle of inner rotation until collision
        if self.r_storage["thetay0"] == 0.0:
            thetay0 = 1.0
        else:
            thetay0 = (2 * self.r_storage["thetay0"]) / np.pi
        thetay0 *= self.score_weight["thetay0"]

        # Compute block_score
        self.acc_grid = 200
        eq_metric = self.b_functiontraj(N=20, alpha=np.pi/4, show=False)
        block_score = 1 / (eq_metric + 1.0)
        block_score *= self.score_weight["block_score"]

        # Compute friction score
        # fric_score = self.friction_length(eps=0.1)  # Doesn play a great role in practice

        # Compute additive_margin
        # margin_score = self.parameters["m"] * self.parameters["Nb"]

        # Compute FBB score ~Rmin to maximize
        # fbb_score = self.br_score(z_uniform=True)
        fbb_score = 1 / ((self.parameters["RBmin"] * self.parameters["Nb"]) - 16.5)
        fbb_score *= self.score_weight["fbb_score"]

        return [disloc_score, block_score, thetay0, fbb_score]  # , margin_score]


def showSDF(bearing, real=False, cmap='bwr', partname="roller"):
    """
    Article figure SDF/SAF.
    :param bearing:
    :param real:
    :param cmap:
    :return:
    """
    xmin, xmax, ymin, ymax = -10, 10, -10, 10
    # Grille de points
    xs = np.linspace(xmin, xmax, 500)
    ys = np.linspace(ymin, ymax, 500)
    X, Y = np.meshgrid(xs, ys)

    part = {"roller": bearing.roller, "iring": bearing.iring, "oring": bearing.oring}
    sdf_roller = part[partname].SDF2(X, Y, real=real)
    x_untilmid = np.linspace(0, bearing.roller.mid_width, 100)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    # Définir les niveaux comme dans ton exemple
    # max_level = max(abs(np.min(sdf_roller)), abs(np.max(sdf_roller)))
    max_level = L #  7.5
    levels = np.linspace(-max_level, max_level, 15)

    # Remplissage coloré avec contours noirs
    contourf = ax.contourf(X, Y, sdf_roller, levels=levels, cmap=cmap)  # jet est cool
    contour = ax.contour(X, Y, sdf_roller, levels=levels, colors='k', linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # Iso-ligne en 0 avec un gros trait noir
    y_curve = -bearing.roller.get_Rxf(x_untilmid, offset=bearing.roller.rmid)
    ax.plot(x_untilmid, y_curve, color='k', lw=2.5)
    ax.plot([bearing.roller.mid_width, bearing.roller.mid_width],
            [y_curve[-1], 0], color='k', lw=2.5)

    # Barre de couleur verticale
    cbar = plt.colorbar(contourf, ax=ax)
    # cbar.set_label('SDF function')

    plt.tight_layout()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.grid(True)
    plt.show()
    return fig, ax

# Fonction de recherche binaire
def find_first_positive_index(X, increasing=True):
    low, high = 0, len(X)
    while low < high:
        mid = (low + high) // 2
        if (X[mid] > 0 if increasing else X[mid] < 0):
            high = mid
        else:
            low = mid + 1
    return low

def make_spline_func(y_ctrl, xmax):
    n = len(y_ctrl)
    x_ctrl = np.linspace(0, xmax, n + 1, endpoint=True)
    y_ctrl = np.insert(y_ctrl, 0, np.array(0.0))

    # Ajoute un point très proche de 0 avec la même valeur (pour forcer y'(0) = 0)
    x_ctrl = np.insert(x_ctrl, 1, 1e-8)
    y_ctrl = np.insert(y_ctrl, 1, 0.0)

    spline = PchipInterpolator(x_ctrl, y_ctrl, extrapolate=True)
    return spline

class NumpyEncoder(json.JSONEncoder):
    """Convertit les objets numpy en types JSON standard"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def compute_trajectory(system):
    # Génère une trajectoire (x, theta) pour la forme 3
    t0 = time.time()
    n_steps = 30
    x_vals = np.linspace(0, -15, n_steps)
    theta_vals = np.zeros([n_steps])
    trajectory = list(zip(x_vals, theta_vals))
    for i, (x3, theta3) in enumerate(trajectory):
        system.step(x3, theta3, method="Powell", silent=False)
        # system.render(show_polygon=True)
    print("Time computation: ", time.time() - t0)
    system.save_storage(filename="../storage_clearance.json")

def animate_construction(system, fps=30, save='collision_path'):
    # Initialiser le graphique
    print("Create animation ...")
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(-25, 8)
        ax.set_ylim(-8, 8)

        system.render(step=frame, ax=ax)

    ani = FuncAnimation(fig, update, frames=len(system.storage["energy"]), interval=50, repeat=True)
    ani.save('D:\\LUCAS\\COURS\\POSTDOC\\Code\\data\\' + save + '.gif', writer='imagemagick', fps=fps)
    return ani

def bsp(Nb=7):
    filepath = "D:\\LUCAS\\COURS\\POSTDOC\\Experiment\\Bearings\\rollerbearing\\bspbearing_reverse.FCStd"
    import FreeCAD
    import rollerbearing_extruder as rbe
    doc = FreeCAD.open(filepath)
    rol = Roller()
    sk = doc.Sketch001
    params = [{"Nb": 7, "Rb": 7.3, "Rin&m": 9.8705},
              {"Nb": 8, "Rb": 6.6, "Rin&m": 11.0386},
              {"Nb": 9, "Rb": 5.9, "Rin&m": 11.789},
              {"Nb": 10, "Rb": 5.5, "Rin&m": 12.7837},
              {"Nb": 11, "Rb": 5.2, "Rin&m": 13.7896},
              {"Nb": 12, "Rb": 4.8, "Rin&m": 14.32533},
              {"Nb": 13, "Rb": 4.6, "Rin&m": 15.24826}]
    Rb = next(p["Rb"] for p in params if p["Nb"] == Nb)
    Rin = next(p["Rin&m"] - 0.3 for p in params if p["Nb"] == Nb)
    sk.setDatum(11, Rb)  # Rb constraint, 7.3 for Nb=7
    sk.setDatum(13, Rin)  # Rin + m constraint, 9.8705, Info reperdue lors du offset.

    rol.importSketch(sk)
    oring = OuterRing(rol)
    iring = InnerRing(rol)
    rb = BearingSimulation(rol, iring, oring)
    rb.get_parameters(Nb=Nb)  # force recompute with Nb

    extruder = rbe.Extruder(rb)
    extruder.make()
    extruder.mesh(export=True, name="BSPN" + str(Nb))

def plot_b_scores(rb, n_samples_list=[5, 10, 20], n_traj_list=[4, 5, 6, 7, 8, 9, 10]):
    """
    Trace le score en fonction de n_samples pour différentes valeurs de n_traj.

    Parameters
    ----------
    b_score : function
        Fonction de scoring: b_score(ntraj, n_samples) -> float
    n_samples_list : list of int
        Valeurs de n_samples (x-axis)
    n_traj_list : list of int
        Valeurs de n_traj (une courbe par valeur)
    """
    rb.r_score(n_samples=5)
    plt.figure(figsize=(7, 5))

    for ntraj in n_traj_list:
        scores = [rb.b_score(ntraj=ntraj, n_samples=ns) for ns in n_samples_list]
        plt.plot(n_samples_list, scores, marker="o", label=f"n_traj={ntraj}")

    plt.xlabel("n_samples")
    plt.ylabel("score")
    plt.title("Score en fonction de n_samples et n_traj")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Baselines functions
def baselineC_func():
    """Baseline orange with few contact"""
    def r_func(x):
        x = np.asarray(x)
        x = np.abs(x)
        xs = np.array([0.0, 3.0, 4.22, 7.5, 7.5 + 0.01])
        ys = np.array([2.45, 6.7, 6.7, 4.08, 0.0])
        return np.interp(x, xs, -ys)
    return 0, r_func

# Celle là de coté
def baseline2_func():
    rmid = 2.13
    x_ctrl = np.array([0.0, 0.01, 5.53, 7.5])
    y_ctrl = np.array([rmid, rmid, 6.7, 5.55])

    spline = PchipInterpolator(x_ctrl, -y_ctrl, extrapolate=True)
    return 0.0, spline

def baselineA_func():
    """Baseline orange with few contact"""
    def r_func(x):
        x = np.asarray(x)
        x = np.abs(x)
        xs = np.array([0.0, 2.5, 5, 7.5, 7.5 + 0.01])
        ys = np.array([6.7, 6.7, 3.88, 3.88, 0.0])
        return np.interp(x, xs, -ys)
    return 0, r_func

def baselineB_func():
    """Baseline orange with few contact"""
    def r_func(x):
        x = np.asarray(x)
        x = np.abs(x)
        xs = np.array([0.0, 1.66, 4.54, 7.5, 7.5 + 0.01])
        ys = np.array([4.46, 4.46, 6.7, 6.7, 0.0])
        return np.interp(x, xs, -ys)
    return 0, r_func

def baselineSphere_func():
    """Baseline orange with few contact"""
    def r_func(x, radius=6.8):
        x = np.asarray(x)
        abs_x = np.abs(x)
        condition = abs_x <= 1
        result_inside = np.sqrt(radius - abs_x * abs_x)
        result_outside = 0.0
        return np.where(condition, result_inside, result_outside)

    return 0, r_func

def test_bearings(N=1, n_ctrl=5):
    """Function which test unstable direction of a lot of bearings"""
    for k in range(N):
        RB_MAX0 = (np.tan(70 * np.pi / 180) * (L / (n_ctrl * 2))) + RB_MIN  # = 6.62 but set to 8 before
        RB_MAX = min([(R_EXT - (T_OUT_MIN + T_IN_MIN + R_SHAFT + 2 * CLEARANCE)) / 2, RB_MAX0])
        x = np.random.uniform(RB_MIN, RB_MAX, (n_ctrl,))
        rb = utils.make_rollerbearing(x, rext=R_EXT, rshaft=R_SHAFT, L=L)
        print(rb.score(), rb.constraints())
        # b.b_hess2()

import numpy as np
import matplotlib.pyplot as plt

def test_rotation(d):
    def f(z):
        z = np.asarray(z)
        return np.where((-5 < z) & (z < 5), (z * z) / 10, 0.0)

    # paramètres (exemple)
    x, y, theta = 0, d, 0  # pose initiale de R1 dans R0
    theta1, eps, theta2 = 0.1, 1.0, -0.2
    u = np.linspace(-6, 6, num=200)
    v = f(u)  # points exprimés dans R1

    # transformations
    phi = theta1 + theta + theta2
    tx = np.cos(theta1) * (x + eps * np.cos(theta)) - np.sin(theta1) * (y + eps * np.sin(theta))
    ty = np.sin(theta1) * (x + eps * np.cos(theta)) + np.cos(theta1) * (y + eps * np.sin(theta))

    X = np.cos(phi) * u - np.sin(phi) * v + tx
    Y = np.sin(phi) * u + np.cos(phi) * v + ty

    # --- Visualisation ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(X, Y, label="Courbe transformée (dans R0)")
    ax.scatter(tx, ty, color="red", label="Origine R1 (dans R0)")
    ax.plot(u, v, label="Courbe originale (dans R0)")
    ax.scatter(0, 0, color="black", label="Origine R0")

    # Grille et proportions
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (R0)")
    ax.set_ylabel("Y (R0)")

    # Axes du repère R0
    ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='black', width=0.005)
    ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='black', width=0.005)
    ax.text(1.1, 0, "x₀", fontsize=12)
    ax.text(0, 1.1, "y₀", fontsize=12)

    # Axes du repère R1 (dessinés à partir de son origine dans R0)
    ax.quiver(tx, ty, np.cos(phi), np.sin(phi), angles='xy', scale_units='xy', scale=1, color='red', width=0.005)
    ax.quiver(tx, ty, -np.sin(phi), np.cos(phi), angles='xy', scale_units='xy', scale=1, color='red', width=0.005)
    ax.text(tx + np.cos(phi)*1.1, ty + np.sin(phi)*1.1, "x₁", color="red", fontsize=12)
    ax.text(tx - np.sin(phi)*1.1, ty + np.cos(phi)*1.1, "y₁", color="red", fontsize=12)

    ax.legend()
    plt.title(f"Transformation du repère R1 (d={d})")
    plt.show()

def test_sdftime(roller, acc_grid=200):
    xs = np.linspace(-2 * roller.mid_width, 2 * roller.mid_width, acc_grid)
    ys = np.linspace(-10.0, 10.0, acc_grid)
    X, Y = np.meshgrid(xs, ys)

    t0 = time.time()
    roller.cache_sdf_grid(resolution=0.1, real=True)
    print("Init time realSDF: ", time.time() - t0)

    t0 = time.time()
    for k in range(50):
        sdf_roller_iring = roller.SDF2(X, Y, real=True)
    print("Average time access realSDF: ", (time.time() - t0) / 50)

    t0 = time.time()
    roller.cache_sdf_grid(resolution=0.1, real=False)
    print("Init time SAF: ", time.time() - t0)

    t0 = time.time()
    for k in range(50):
        sdf_roller_iring = roller.SDF2(X, Y, real=False)
    print("Average time access SAF: ", (time.time() - t0) / 50)

def test_scoretime(rb):
    t0 = time.time()
    for k in range(20):
        print(str(k), end="")
        sc = rb.score()
    print()
    print("Average score time: ", (time.time() - t0) / 20)




if __name__ == "__main__":
    # filepath = "D:\\LUCAS\\COURS\\POSTDOC\\Experiment\\Bearings\\rollerbearing\\bestbearing_parametric45deg.FCStd"
    # filepath = "D:\\LUCAS\\COURS\\POSTDOC\\Experiment\\Bearings\\rollerbearing\\bspbearing_reverse.FCStd"
    # filepath = "D:\\LUCAS\\COURS\\POSTDOC\\Experiment\\Bearings\\rollerbearing\\spherebearing_reverse.FCStd"
    filepath = "D:\\LUCAS\\COURS\\POSTDOC\\Experiment\\Bearings\\rollerbearing\\bspbr_Nb=8.FCStd"
    # import FreeCAD
    # doc = FreeCAD.open(filepath)
    # rol = Roller()
    # rol.rmid, rfunc = baselineSphere_func()
    # rol.set_Rxf(rfunc)
    # rol.render()
    # rol.importSketch(doc.Sketch001)

    path = "../data/optim_results/optim_cylbearing5.json"
    # rb = utils.getbyrank(path, 1, w=[1, 1, 1, 0.1])
    # rb.roller.render()

    test_bearings(N=10)

    # oring = OuterRing(rol)
    # iring = InnerRing(rol)
    # rb = BearingSimulation(rol, iring, oring, rext=30.0, rshaft=6.5)
    # print(rb.br_score(z_uniform=True))
    # print(rb.br_score(z_uniform=False))
    # test_scoretime(rb)
    scs = rb.score()
    print("Sbreak: ", round(scs[3],3))
    print("Sdicloc: ", round(scs[0], 3))
    print("Sblock1: ", round(scs[2], 3))
    print("Sblock2: ", round(scs[1], 3))
    # rb = utils.make_rollerbearing(np.array([4.5, 5, 5.5, 5.5, 5, 1.5, 5, 5]), rext=30.0, rshaft=6.5)
    # rb.d_score_rot(N=30, max_iring_angle=np.pi / 4, clearance_factor=2.0, method="Powell", silent=False)  # method="L-BFGS-B"
    # rb.d_step(0.0, 0.2, silent=True, method="L-BFGS-B")
    # rb.d_sdf(0, 0, 0.2, 0, 0, 0)
    # rb.d_render(step=5)
    # rb.d_energies()
    # rb.d_render(show=True)
    # rb.acc_grid = 200
    # rb.d_score(N=5, clearance_factor=2.0)
#
    # print(rb.r_score(n_samples=5))
    # print(rb.b_hess2())
    # rb.roller.render()
    # rb.b_functionflex(show=True, axis=(0, 1))
    # rb.b_functionflex(show=True, axis=(0, 2))
    # rb.b_functionflex(show=True, axis=(1, 2))
    # rb.b_init()
    # rb.b_istep(0, 0.1)
    # rb.b_functionflex()
    # ax = rb.b_functiontraj(N=20, alpha=0        , dofig=True, silent=True, show=False, ax=None)
    # ax = rb.b_functiontraj(N=20, alpha=np.pi / 4, dofig=True, silent=True, show=False, ax=ax)
    # ax = rb.b_functiontraj(N=20, alpha=np.pi / 2, dofig=True, silent=True, show=True, ax=ax)

    # rb.roller.render()
    # rb.b_score(ntraj=4, n_samples=50, eps=0.1, silent=False)
    # rb.b_fastscore(n_samples=10)
    # plot_b_scores(rb)
    # dist, _ = rb.b_stepflex(0.2, 0.0)
    # rb.b_sdf(0.2, 0.0, dist=dist)
    # rb.r_render()
    # rb.b_function()

    # rb.roller.render(show=True)
    # rb.b_functiontraj(N=50, traj=(1, 1), show=False, silent=True)
    # rol.render()
    # rol.set_Rxf(np.array([0.0, 0.5]))
    # rol.set_Rxf(np.array([-0.3, -1.0, -0.5, -0.6]))
    # rol.render()

    # system = BearingSimulation(rol, iring, oring)
    # print(system.score())
    # print(system.constraints())
    # print(system.friction_length())
    # system.acc_grid = 50
    # for thetay in np.linspace(0, np.pi / 2, 5):
    #     system.r_step(thetay, silent=True)
    # # system.step(-5.0, 0.0, lr=0.05, n_iter=2000, silent=False)
    # # # system.render()
    # showSDF(rb, real=False, cmap="jet")
    # showSDF(rb, real=True, cmap="jet")
    # plt.plot(system.r_storage["energy"])

    # rol.render()
    # compute_trajectory(system)
    #system.load_storage(filename="storage.json")
    #system.energies()


