import FreeCAD as App
import Part, Draft
from BOPTools import BOPFeatures
import MeshPart, Mesh
import roller_design as RD
import utils
import numpy as np


class Extruder:
    def __init__(self, rollerbearing: RD.BearingSimulation):
        self.rb = rollerbearing
        self.doc = App.newDocument("RevolutionExample")
        self.filename = "../data/experiments/bearing.FCStd"
        self.export_path = "../data/experiments/rollerbearing-stl"
        self.y_offset = self._offset_roller()
        self.rshaft = RD.R_SHAFT
        self.rext = RD.R_EXT
        self.iring, self.oring, self.rollers = None, None, None
        self.meshes = list()

        self.bputils = BOPFeatures.BOPFeatures(self.doc)

    def _offset_roller(self):
        return self.rb.parameters["Rin"] + self.rb.roller.rminmax()[1] + self.rb.roller.clearance

    def _roller_sketch(self):
        """Y_offset should be set to Rin+m+Rbmax."""
        self.roller_sketch = self.doc.addObject('Sketcher::SketchObject', 'RollerSketch')
        self.roller_sketch.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(0, 0, 0, 1))

        x1, y1 = self.rb.roller.get_Rxp(offset=self.rb.roller.rmid)
        x1, y1 = x1[1:], y1[1:]
        x6, y6 = list(reversed(-x1)), list(reversed(y1))  # Tronçon en bas à gauche
        xrevleft, yrevleft = np.array([self.rb.roller.mid_width]), np.array([0.0])
        xrevright, yrevright = np.array([-self.rb.roller.mid_width]), np.array([0.0])

        xf = np.concatenate((x6, x1, xrevleft, xrevright))
        yf = np.concatenate((y6, y1, yrevleft, yrevright))

        points = [App.Vector(x, y + self.y_offset, 0) for x, y in zip(xf, yf)]
        points.append(points[0])

        # Ajout des segments reliant les points
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            self.roller_sketch.addGeometry(Part.LineSegment(p1, p2), False)

        self.roller_axis = (App.Vector(1, 0, 0), App.Vector(0, self.y_offset, 0))

    def _iring_sketch(self):
        self.iring_sketch = self.doc.addObject('Sketcher::SketchObject', 'IringSketch')
        self.iring_sketch.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(0, 0, 0, 1))

        x1, y1 = self.rb.roller.get_Rxp(offset=self.rb.roller.rmid + self.rb.roller.clearance)  # Tronçon en bas à droite
        x1, y1 = x1[1:], y1[1:]
        neg_length_shaft = self.y_offset - self.rshaft
        xrevleft, yrevleft = np.array([-self.rb.roller.mid_width]), np.array([-neg_length_shaft])
        xrevright, yrevright = np.array([self.rb.roller.mid_width]), np.array([-neg_length_shaft])
        x6, y6 = list(reversed(-x1)), list(reversed(y1))  # Tronçon en bas à gauche

        xf = np.concatenate((x6, x1, xrevright, xrevleft))
        yf = np.concatenate((y6, y1, yrevright, yrevleft))

        points = [App.Vector(x, y + self.y_offset, 0) for x, y in zip(xf, yf)]
        points.append(points[0])

        # Ajout des segments reliant les points
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            self.iring_sketch.addGeometry(Part.LineSegment(p1, p2), False)

        self.iring_axis = (App.Vector(1, 0, 0), App.Vector(0, 0, 0))

    def _oring_sketch(self):
        self.oring_sketch = self.doc.addObject('Sketcher::SketchObject', 'OringSketch')
        self.oring_sketch.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(0, 0, 0, 1))

        x1, y1 = self.rb.roller.get_Rxp(offset=self.rb.roller.rmid + self.rb.roller.clearance)  # Tronçon en bas à droite
        x1, y1 = x1[1:], y1[1:]
        x3, y3 = list(reversed(x1)), list(reversed(-y1))  # Tronçon haut droite (de droite à gauche)
        x4, y4 = -x1, -y1  # Tronçon haut gauche
        x5, y5 = [-self.rb.roller.mid_width, self.rb.roller.mid_width], [self.rext - self.y_offset] * 2

        xf = np.concatenate((x3, x4, x5))
        yf = np.concatenate((y3, y4, y5))

        points = [App.Vector(x, y + self.y_offset, 0) for x, y in zip(xf, yf)]
        points.append(points[0])

        # Ajout des segments reliant les points
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            self.oring_sketch.addGeometry(Part.LineSegment(p1, p2), False)

        self.oring_axis = (App.Vector(1, 0, 0), App.Vector(0, 0, 0))

    def _revolve(self, obj: str):
        assert obj in ["roller", "iring", "oring"]
        sketch = getattr(self, obj + "_sketch")
        axis, base = getattr(self, obj + "_axis")
        revolve = self.doc.addObject("Part::Revolution", obj + "Rev")
        revolve.Source = sketch
        revolve.Axis = axis
        revolve.Base = base
        revolve.Angle = 360.0  # Révolution complète
        revolve.Solid = True
        setattr(self, obj + "_rev", revolve)

    def _duplicate_roller(self):
        Nb = self.rb.parameters["Nb"]
        rollers_list = list()
        rollers_list.append(self.roller_rev)
        for k in range(Nb - 1):
            angle_deg = (k + 1) * 360.0 / Nb
            pl = App.Placement(App.Vector(0, 0, 0), App.Rotation(App.Vector(1, 0, 0), angle_deg))
            clone = Draft.make_clone(self.roller_rev)
            clone.Placement = pl
            rollers_list.append(clone)

        self.rollers = self.doc.addObject("Part::Compound","Compound")
        self.rollers.Links = rollers_list
        self.rollers.Label = "Rollers"

    def _oring_slot(self, slots=True):
        if slots:
            box1 = self.doc.addObject("Part::Box","Oslot1")
            box1.Length, box1.Width, box1.Height = "3mm", str(RD.L) + "mm", "1mm"
            box1.Placement = App.Placement(App.Vector(RD.L / 2, RD.R_EXT - 0.5, -0.5), App.Rotation(90, 0, 0))

            box2 = self.doc.addObject("Part::Box", "Oslot2")
            box2.Length, box2.Width, box2.Height = "3mm", str(RD.L) + "mm", "1mm"
            box2.Placement = App.Placement(App.Vector(RD.L / 2, -RD.R_EXT - 2.5, -0.5), App.Rotation(90, 0, 0))

            self.oring = self.bputils.make_multi_fuse([self.oring_rev.Name, "Oslot1", "Oslot2"])

        else:
            self.oring = self.oring_rev

        self.oring.Label = "OuterRing"
        self.doc.recompute()

    def _iring_slot(self, slots=True):
        if slots:
            box = self.doc.addObject("Part::Box", "Islot")
            length = RD.L + 1
            box.Length, box.Width, box.Height = str(length) + "mm", "2mm", "3mm"
            box.Placement = App.Placement(App.Vector(-length, -1.0, RD.R_SHAFT - 1), App.Rotation(0, 0, 0))

            self.iring = self.bputils.make_cut([self.iring_rev.Name, "Islot"])

        else:
            self.iring = self.iring_rev

        self.iring.Label = "InnerRing"
        self.doc.recompute()

    def _save(self):
        self.doc.saveAs(self.filename)

    def make(self, slots=True):
        print("FreeCAD bearing making...")
        self._roller_sketch()
        self._iring_sketch()
        self._oring_sketch()
        self._revolve("roller")
        self._revolve("iring")
        self._revolve("oring")
        self._iring_slot(slots=slots)
        self._oring_slot(slots=slots)
        self._duplicate_roller()
        self._save()
        print("✅ FreeCAD object constructed.")

    def mesh(self, export=True, name="bearing"):
        for obj in [self.oring, self.iring, self.rollers]:
            mesh = self.doc.addObject("Mesh::Feature", "Mesh" + obj.Label)
            mesh.Mesh = MeshPart.meshFromShape(Shape=obj.Shape,
                                               LinearDeflection=0.2,
                                               AngularDeflection=0.0872665,
                                               Relative=False)
            mesh.Label = "Mesh" + obj.Label
            self.meshes.append(mesh)
            self.doc.recompute()

        self._save()

        if export:
            path = self.export_path + "\\" + name + ".stl"
            Mesh.export(self.meshes, path)
            print("✅ Bearing exported to ", path)


if __name__ == '__main__':
    path = "../data/optim_results/optim_cylbearing9.json"
    # rb = utils.select_best(path)
    rb = utils.getbyrank(path, 2)
    rb.roller.render(show=True)
    extruder = Extruder(rb)
    extruder.make(slots=True)
    extruder.mesh(export=True, name="9#2_team1")
