# roller-bearing-optimisation for SPM 2026 paper
Work done for scientific paper on 3D-printer bearing design optimisation.

## Installation
Assuming that conda already installed.

```
# ENVIRONEMENT INSTALLATION
conda create -n env-name rollerbearingenv python=3.11
conda activate rollerbearingenv
conda install conda-forge::freecad
pip install scipy, plotly, pymoo, dash, pandas

# PROJECT INSTALLATION
git clone https://github.com/luvrgz/roller-bearing-optimisation.git
```
If error No module named 'FreeCAD', the Library/bin and Library/Mod paths have to be added in the environnement variables.
If error FreeCADGui process not found (when importing Draft module), comment line 39 import FreeCADGui as Gui in Mod/Draft/draftutils/ini_draft_statusbar.py.

## Usage
### External constraints
Define dimensional constraints in roller_design.py: 
```
L = 15 --> Bearing width
CLEARANCE = 0.3  --> Clearance between the parts in the bearing
R_EXT = 30  --> External radius of the outer ring
R_SHAFT = 6.5  --> Internal radius of the inner ring
T_OUT_MIN = 5  --> Minimum thickness of the outer ring
T_IN_MIN = 3  --> Minimum thickness of the inner ring
ALPHA_LIM = 60  # (deg)  --> Maximal printing angle
RB_MIN = 2.5   --> Minimum radius of the roller
```

### Bearing simulation
Objectives functions computed in BearingSimulation.scores(), and contraints functions in BearingSimulation.constraints().

### Design optimisation
To run the optimisation:
```
optim_roller_bearing.py, stored in the path
```

See optimisation results:
optim_roller_bearing.py dashboard(path) to see the generated shapes.

### Generate STL
Run rollerbearing_extruder.py

## Experiments
The experimental data are stored in ... (explain the generated data)

Get lifetime parameters from scores: utils.get_kb ...