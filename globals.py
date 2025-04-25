pfp = None
pfdyn = None
pfplot = None
mtb = None

from powfacpy.base.active_project import ActiveProject
from powfacpy.applications.dynamic_simulation import DynamicSimulation
from powfacpy.pf_class_protocols import PFApp
from powfacpy.applications.plots import Plots
from typing import Optional, Tuple
import sys
if getattr(sys, "gettrace", None) is not None:
    sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2025\\Python\\3.12")
import powerfactory as pf  # type: ignore

def connectPF(project_name) -> Tuple[PFApp, int]:
    """
    Connects to the powerfactory application and returns the application and project.
    """
    global pfp
    global pfdyn
    global pfplot
    global mtb
    app: Optional[pf.Application] = pf.GetApplicationExt()
    if not app:
        raise RuntimeError("No connection to powerfactory application")
    app.ClearOutputWindow()
    app.PrintInfo(
        f"Powerfactory application connected externally. Executable: {sys.executable}"
    )
    app.PrintInfo(f"Imported powerfactory module from {pf.__file__}")

    version: str = pf.__version__
    pfVersion = 2000 + int(version.split(".")[0])
    app.PrintInfo(f"Powerfactory version registred: {pfVersion}")

    app.ActivateProject(project_name)

    pfp = ActiveProject(app)
    pfdyn = DynamicSimulation(app)
    pfplot = Plots(app)
    mtb = pfp.get_unique_obj("MTB.ElmNet", include_subfolders=True)

    return app, pfVersion