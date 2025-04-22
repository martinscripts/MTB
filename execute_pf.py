"""
Executes the Powerplant model testbench in Powerfactory.
"""

from __future__ import annotations
import globals

DEBUG = True

import os

# Ensure right working directory
executePath = os.path.abspath(__file__)
executeFolder = os.path.dirname(executePath)
os.chdir(executeFolder)

from configparser import ConfigParser

from powfacpy.base.active_project import ActiveProject
from powfacpy.applications.study_cases import StudyCases
from powfacpy import pf_class_protocols as pfclasses


class Config:
    def __init__(self) -> None:
        self.cp = ConfigParser(allow_no_value=True)
        self.cp.read("config.ini")
        self.parsedConf = self.cp["config"]
        self.sheetPath = str(self.parsedConf["Casesheet path"])
        self.pythonPath = str(self.parsedConf["Python path"])
        self.parallel = bool(self.parsedConf["Parallel"])
        self.exportPath = str(self.parsedConf["Export folder"])
        self.QDSLcopyGrid = str(self.parsedConf["QDSL copy grid"])
        self.project_name = str(self.parsedConf["PF project name"])


class PowerfactorySettings:
    def __init__(self) -> None:
        self.cp = ConfigParser(allow_no_value=True)
        self.cp.read("config.ini")
        self.parsedPfSettings = self.cp["powerfactorySettings"]

        self.only_setup = int(self.parsedPfSettings["Only_setup"])
        self.post_run_backup = bool(self.parsedPfSettings["Post_run_backup"])
        self.sub_conf_str = str(self.parsedPfSettings["sub_conf_str"])

        self.pref_sub_attrib = str(self.parsedPfSettings["Pref_sub_attrib"])
        self.pref_sub_scale = float(self.parsedPfSettings["Pref_sub_scale"])
        self.qref_q_sub_attrib = str(self.parsedPfSettings["Qref_q_sub_attrib"])
        self.qref_q_sub_scale = float(self.parsedPfSettings["Qref_q_sub_scale"])
        self.qref_qu_sub_attrib = str(self.parsedPfSettings["Qref_qu_sub_attri"])
        self.qref_qu_sub_scale = float(self.parsedPfSettings["Qref_qu_sub_scale"])
        self.qref_pf_sub_attrib = str(self.parsedPfSettings["Qref_pf_sub_attri"])
        self.qref_pf_sub_scale = float(self.parsedPfSettings["Qref_pf_sub_scale"])
        self.pref_sub = str(self.parsedPfSettings["Pref_sub"])
        self.qref_q_sub = str(self.parsedPfSettings["Qref_q_sub"])
        self.qref_qu_sub = str(self.parsedPfSettings["Qref_qu_sub"])
        self.qref_pf_sub = str(self.parsedPfSettings["Qref_pf_sub"])

        self.custom1_sub_attrib = str(self.parsedPfSettings["Custom1_sub_attri"])
        self.custom1_sub_scale = float(self.parsedPfSettings["Custom1_sub_scale"])
        self.custom2_sub_attrib = str(self.parsedPfSettings["Custom2_sub_attri"])
        self.custom2_sub_scale = float(self.parsedPfSettings["Custom2_sub_scale"])
        self.custom3_sub_attrib = str(self.parsedPfSettings["Custom3_sub_attri"])
        self.custom3_sub_scale = float(self.parsedPfSettings["Custom3_sub_scale"])
        self.custom1_sub = str(self.parsedPfSettings["Custom1_sub"])
        self.custom2_sub = str(self.parsedPfSettings["Custom2_sub"])
        self.custom3_sub = str(self.parsedPfSettings["Custom3_sub"])

        self.meas_obj = {}
        for i in range(1, 101):
            meas_obj = str(self.parsedPfSettings[f"Meas_obj_{i}"])
            if meas_obj == "":
                continue
            pf_obj_name, alias, signals = meas_obj.split("|")
            assert len(signals) > 0
            assert not alias == ""
            assert not pf_obj_name == ""
            self.meas_obj[pf_obj_name] = (alias, signals.split(","))


config = Config()
pfSettings = PowerfactorySettings()
import sys

sys.path.append(config.pythonPath)

from typing import Optional, Tuple, List, Union

if getattr(sys, "gettrace", None) is not None:
    sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2025\\Python\\3.12")
import powerfactory as pf  # type: ignore

import re
import time
from datetime import datetime
import case_setup as cs
import sim_interface as si


def script_GetExtObj(script: pf.ComPython, name: str) -> Optional[pf.DataObject]:
    """
    Get script external object.
    """
    retVal: List[Union[int, pf.DataObject, None]] = script.GetExternalObject(name)
    assert isinstance(retVal[1], (pf.DataObject, type(None)))
    return retVal[1]


def script_GetStr(script: pf.ComPython, name: str) -> Optional[str]:
    """
    Get script string parameter.
    """
    retVal: List[Union[int, str]] = script.GetInputParameterString(name)
    if retVal[0] == 0:
        assert isinstance(retVal[1], str)
        return retVal[1]
    else:
        return None


def script_GetDouble(script: pf.ComPython, name: str) -> Optional[float]:
    """
    Get script double parameter.
    """
    retVal: List[Union[int, float]] = script.GetInputParameterDouble(name)
    if retVal[0] == 0:
        assert isinstance(retVal[1], float)
        return retVal[1]
    else:
        return None


def script_GetInt(script: pf.ComPython, name: str) -> Optional[int]:
    """
    Get script integer parameter.
    """
    retVal: List[Union[int, int]] = script.GetInputParameterInt(name)
    if retVal[0] == 0:
        assert isinstance(retVal[1], int)
        return retVal[1]
    else:
        return None


def resetProjectUnits(project: pfclasses.IntPrj) -> None:
    """
    Resets the project units to the default units.
    """
    SetPrj = project.SearchObject("Settings.SetFold")
    if SetPrj:
        SetPrj.Delete()

    project.Deactivate()
    project.Activate()


def setupResFiles(pfSettings):
    """
    Setup the result files for the studycase.
    """
    elmRes: pfclasses.ElmRes = globals.pfp.app.GetFromStudyCase("ElmRes")  # type: ignore
    assert elmRes is not None

    measurementBlock = globals.pfp.get_unique_obj(
        "measurements.ElmDsl", include_subfolders=True, parent_folder=globals.mtb
    )
    globals.pfp.add_results_variable(
        measurementBlock,
        [
            "s:Ia_pu",
            "s:Ib_pu",
            "s:Ic_pu",
            "s:Vab_pu",
            "s:Vag_pu",
            "s:Vbc_pu",
            "s:Vbg_pu",
            "s:Vca_pu",
            "s:Vcg_pu",
            "s:f_hz",
            "s:neg_Id_pu",
            "s:neg_Imag_pu",
            "s:neg_Iq_pu",
            "s:neg_Vmag_pu",
            "s:pos_Id_pu",
            "s:pos_Imag_pu",
            "s:pos_Iq_pu",
            "s:pos_Vmag_pu",
            "s:ppoc_pu",
            "s:qpoc_pu",
        ],
    )

    signals = [
        "mtb_s_pref_pu.ElmDsl",
        "mtb_s_qref.ElmDsl",
        "mtb_s_qref_q_pu.ElmDsl",
        "mtb_s_qref_qu_pu.ElmDsl",
        "mtb_s_qref_pf.ElmDsl",
        "mtb_s_qref_3.ElmDsl",
        "mtb_s_qref_4.ElmDsl",
        "mtb_s_qref_5.ElmDsl",
        "mtb_s_qref_6.ElmDsl",
        "mtb_s_1.ElmDsl",
        "mtb_s_2.ElmDsl",
        "mtb_s_3.ElmDsl",
        "mtb_s_4.ElmDsl",
        "mtb_s_5.ElmDsl",
        "mtb_s_6.ElmDsl",
        "mtb_s_7.ElmDsl",
        "mtb_s_8.ElmDsl",
        "mtb_s_9.ElmDsl",
        "mtb_s_10.ElmDsl",
    ]

    for signal in signals:
        signalObj = globals.pfp.get_unique_obj(
            signal, include_subfolders=True, parent_folder=globals.mtb
        )
        globals.pfp.add_results_variable(signalObj, "s:yo")

    # Include optional measurement objects and set alias
    for meas_obj_name, alias_and_signals in pfSettings.meas_obj.items():
        alias, signals = alias_and_signals
        meas_obj = globals.pfp.get_unique_obj(
            meas_obj_name, include_subfolders=True, parent_folder=globals.mtb
        )
        globals.pfp.add_results_variable(meas_obj, signals)
        meas_obj.for_name = alias


def setupExport(app: pf.Application, filename: str):
    """
    Setup the export component for the studycase.
    """
    comRes: pf.ComRes = globals.pfp.app.GetFromStudyCase("ComRes")  # type: ignore
    elmRes: pf.ElmRes = globals.pfp.app.GetFromStudyCase("ElmRes")  # type: ignore
    assert comRes is not None
    assert elmRes is not None

    csvFileName = f"{filename}.csv"
    comRes.SetAttribute("pResult", elmRes)
    comRes.SetAttribute("iopt_exp", 6)
    comRes.SetAttribute("iopt_sep", 0)
    comRes.SetAttribute("ciopt_head", 1)
    comRes.SetAttribute("iopt_locn", 4)
    comRes.SetAttribute("dec_Sep", ",")
    comRes.SetAttribute("col_Sep", ";")
    comRes.SetAttribute("f_name", csvFileName)


def setupPlots(app: pf.Application, root: pf.DataObject):
    """
    Setup the plots for the studycase.
    """
    globals.pfp.app.Show()
    measurementBlock = root.SearchObject("measurements.ElmDsl")
    assert measurementBlock is not None

    board: pf.SetDesktop = globals.pfp.app.GetFromStudyCase("SetDesktop")  # type: ignore
    assert board is not None

    plots: List[pf.GrpPage] = board.GetContents("*.GrpPage", 1)  # type: ignore

    for p in plots:
        p.RemovePage()

    # Create pages
    plotPage: pf.GrpPage = board.GetPage("Plot", 1, "GrpPage")  # type: ignore
    assert plotPage is not None

    # PQ plot
    pqPlot: pf.PltLinebarplot = plotPage.GetOrInsertPlot("PQ", 1)  # type: ignore
    assert pqPlot is not None
    pqPlotDS: pf.PltDataseries = pqPlot.GetDataSeries()  # type: ignore
    assert pqPlotDS is not None
    pqPlotDS.AddCurve(measurementBlock, "s:ppoc_pu")
    pqPlotDS.AddCurve(measurementBlock, "s:qpoc_pu")
    pqPlot.DoAutoScale()

    # U plot
    uPlot: pf.PltLinebarplot = plotPage.GetOrInsertPlot("U", 1)  # type: ignore
    assert uPlot is not None
    uPlotDS: pf.PltDataseries = uPlot.GetDataSeries()  # type: ignore
    assert uPlotDS is not None
    uPlotDS.AddCurve(measurementBlock, "s:pos_Vmag_pu")
    uPlotDS.AddCurve(measurementBlock, "s:neg_Vmag_pu")
    uPlot.DoAutoScale()

    # I plot
    iPlot: pf.PltLinebarplot = plotPage.GetOrInsertPlot("I", 1)  # type: ignore
    assert iPlot is not None
    iPlotDS: pf.PltDataseries = iPlot.GetDataSeries()  # type: ignore
    assert iPlotDS is not None
    iPlotDS.AddCurve(measurementBlock, "s:pos_Id_pu")
    iPlotDS.AddCurve(measurementBlock, "s:pos_Iq_pu")
    iPlotDS.AddCurve(measurementBlock, "s:neg_Id_pu")
    iPlotDS.AddCurve(measurementBlock, "s:neg_Iq_pu")
    iPlot.DoAutoScale()

    # F plot
    fPlot: pf.PltLinebarplot = plotPage.GetOrInsertPlot("F", 1)  # type: ignore
    assert fPlot is not None
    fPlotDS: pf.PltDataseries = fPlot.GetDataSeries()  # type: ignore
    assert fPlotDS is not None
    fPlotDS.AddCurve(measurementBlock, "s:f_hz")
    fPlot.DoAutoScale()

    globals.pfp.app.WriteChangesToDb()
    globals.pfp.app.Hide()


def addCustomSubscribers(pfSettings, channels: List[si.Channel]) -> None:
    """
    Add custom subscribers to the channels. For example, references applied as parameter events directly to control blocks.
    """

    def get_channel_by_name(name: str) -> si.Channel:
        for ch in channels:
            if ch.name == name:
                return ch
        raise RuntimeError(f"Channel {name} not found.")

    custConfStr = pfSettings.sub_conf_str
    assert isinstance(custConfStr, str)

    def convert_to_conf_str(param: str, signal: str) -> str:
        param = param.lower()
        obj_name = getattr(pfSettings, f"{param}_sub")
        if obj_name == "":
            return ""
        sub_obj = globals.pfp.get_unique_obj(
            obj_name,
            include_subfolders=True,
        )
        sub_attrib = getattr(pfSettings, f"{param}_sub_attrib")
        assert isinstance(sub_attrib, str)
        if sub_attrib == "":
            return ""
        sub_scale = getattr(pfSettings, f"{param}_sub_scale")
        assert isinstance(sub_scale, float)
        sub_signal = get_channel_by_name(signal)
        assert isinstance(sub_signal, si.Signal)
        return f"\\{sub_obj.GetFullName()}:{sub_attrib}={signal}:S~{sub_scale} * x"

    pref_conf = convert_to_conf_str("Pref", "mtb_s_pref_pu")
    qref1_conf = convert_to_conf_str("Qref_q", "mtb_s_qref_q_pu")
    qref2_conf = convert_to_conf_str("Qref_qu", "mtb_s_qref_qu_pu")
    qref3_conf = convert_to_conf_str("Qref_pf", "mtb_s_qref_pf")
    custom1_conf = convert_to_conf_str("Custom1", "mtb_s_1")
    custom2_conf = convert_to_conf_str("Custom2", "mtb_s_2")
    custom3_conf = convert_to_conf_str("Custom3", "mtb_s_3")

    configurations = custConfStr.split("|") + [
        pref_conf,
        qref1_conf,
        qref2_conf,
        qref3_conf,
        custom1_conf,
        custom2_conf,
        custom3_conf,
    ]

    confFilterStr = r"^([^:*?=\",~|\n\r]+):((?:\w:)?\w+(?::\d+)?)=(\w+):(S|s|S0|s0|R|r|T|t|C|c)~(.*)"
    confFilter = re.compile(confFilterStr)

    for configuration in configurations:
        confFilterMatch = confFilter.match(configuration)
        if confFilterMatch is not None:
            obj = confFilterMatch.group(1)
            attrib = confFilterMatch.group(2)
            sub = confFilterMatch.group(3)
            typ = confFilterMatch.group(4)
            lamb = confFilterMatch.group(5)

            chnl = get_channel_by_name(sub)
            if isinstance(chnl, si.Signal):
                if typ.lower() == "s" or typ.lower() == "c":
                    chnl.add_pf_sub_s(obj, attrib, lambda _, x, l=lamb: eval(l))
                elif typ.lower() == "s0":
                    chnl.add_pf_sub_s0(
                        obj, attrib, lambda _, x, l=lamb: eval(l)
                    )  # Not exactly safe
                elif typ.lower() == "r":
                    chnl.add_pf_sub_r(obj, attrib, lambda _, x, l=lamb: eval(l))
                elif typ.lower() == "t":
                    chnl.add_pf_sub_t(obj, attrib, lambda _, x, l=lamb: eval(l))
            elif (
                isinstance(chnl, si.Constant)
                or isinstance(chnl, si.PfObjRefer)
                or isinstance(chnl, si.String)
            ):
                chnl.add_pf_sub(obj, attrib)


def main():
    # Connect to Powerfactory
    app, pfVersion = globals.connectPF(config.project_name)

    project = globals.pfp.get_active_project()

    # Check if any studycase is active
    currentStudyCase = globals.pfp.app.GetActiveStudyCase()
    if currentStudyCase is None:
        raise RuntimeError("Please activate a studycase.")

    studyTime: int = currentStudyCase.iStudyTime

    # Get and check for active grids
    activeGrids: List[pf.ElmNet] = globals.pfp.get_obj(
        "*.ElmNet",
        parent_folder=globals.pfp.network_data_folder,
        condition=lambda x: x.IsCalcRelevant(),
        error_if_non_existent=True,
    )

    # Make project backup
    globals.pfp.create_project_version(
        f'PRE_MTB_{datetime.now().strftime(r"%d%m%Y%H%M%S")}'
    )

    resetProjectUnits(project)
    currentStudyCase.Consolidate()

    # Create task automation
    taskAuto: pf.ComTasks = globals.pfp.create_in_folder("taskauto.ComTasks", globals.pfp.study_cases_folder)  # type: ignore
    taskAuto.iEnableParal = int(config.parallel)
    taskAuto.parMethod = 0
    taskAuto.parallelSetting.procTimeOut = 3600

    # Read and setup cases from sheet
    plantSettings, channels, cases, maxRank = cs.setup(casesheetPath=config.sheetPath)

    # Add user channel subscribers
    addCustomSubscribers(pfSettings, channels)

    # Create export folder if it does not exist
    if not os.path.exists(config.exportPath):
        os.makedirs(config.exportPath)

    # Find initializer script object
    initScript = globals.pfp.get_unique_obj(
        "initializer_script.ComDpl",
        include_subfolders=True,
        parent_folder=globals.mtb,
    )

    # List of created studycases for later activation
    studycases: List[pf.IntCase] = []

    currentStudyCase.Deactivate()

    # Filter cases if Only_setup > 0
    assert isinstance(pfSettings.only_setup, int)

    if pfSettings.only_setup > 0:
        cases = list(filter(lambda x: x.rank == pfSettings.only_setup, cases))

    globals.pfp.app.EchoOff()
    for case in cases:
        if case.RMS:
            # Set-up studycase, variation and balance
            caseName = f"{str(case.rank).zfill(len(str(maxRank)))}_{case.Name}".replace(
                ".", ""
            )
            exportName = os.path.join(
                os.path.abspath(config.exportPath),
                f"{plantSettings.Projectname}_{case.rank}",
            )
            newStudycase = globals.pfp.create_in_folder(
                f"{caseName}.IntCase", globals.pfp.study_cases_folder
            )
            assert newStudycase is not None
            studycases.append(newStudycase)
            newStudycase.Activate()
            newStudycase.SetStudyTime(studyTime)

            # Activate the relevant networks
            for g in activeGrids:
                g.Activate()

            newVar = globals.pfp.create_in_folder(
                f"{caseName}.IntScheme", globals.pfp.variations_folder
            )
            assert newVar is not None
            newStage = globals.pfp.create_in_folder(f"{caseName}.IntSstage", newVar)
            assert newStage is not None
            newStage.SetAttribute("e:tAcTime", studyTime)
            newVar.Activate()
            newStage.Activate()

            si.apply_to_powerfactory(channels, case.rank)

            initScript.Execute()

            ### WORKAROUND FOR QDSL FAILING WHEN IN MTB-GRID ###
            # TODO: REMOVE WHEN FIXED
            if config.QDSLcopyGrid != "":
                qdslInitializer = globals.pfp.get_unique_obj(
                    "initializer_qdsl.ElmQdsl",
                    include_subfolders=True,
                    parent_folder=globals.mtb,
                )
                for g in activeGrids:
                    gridName = g.GetFullName()
                    assert isinstance(gridName, str)
                    if gridName.lower().endswith(
                        f"{config.QDSLcopyGrid.lower()}.elmnet"
                    ):
                        g.AddCopy(qdslInitializer)  # type: ignore

                qdslInitializer.SetAttribute("outserv", 1)
            ### END WORKAROUND ###

            inc = globals.pfp.app.GetFromStudyCase("ComInc")
            assert inc is not None
            sim = globals.pfp.app.GetFromStudyCase("ComSim")
            assert sim is not None
            comRes: pf.ComRes = globals.pfp.app.GetFromStudyCase("ComRes")  # type: ignore
            assert comRes is not None

            taskAuto.AppendStudyCase(newStudycase)
            taskAuto.AppendCommand(inc, -1)
            taskAuto.AppendCommand(sim, -1)
            taskAuto.AppendCommand(comRes, -1)
            setupResFiles(pfSettings)
            globals.pfp.app.WriteChangesToDb()
            setupExport(app, exportName)
            globals.pfp.app.WriteChangesToDb()
            newStudycase.Deactivate()
            globals.pfp.app.WriteChangesToDb()

    globals.pfp.app.EchoOn()

    if pfSettings.only_setup == 0:
        taskAuto.Execute()

    if pfVersion >= 2024:
        for studycase in studycases:
            studycase.Activate()
            setupPlots(app, root)
            globals.pfp.app.WriteChangesToDb()
            studycase.Deactivate()
            globals.pfp.app.WriteChangesToDb()
    else:
        globals.pfp.app.PrintWarn(
            "Plot setup not supported for PowerFactory versions older than 2024."
        )

    # Create post run backup
    postBackup = script_GetInt(thisScript, "Post_run_backup")
    assert isinstance(postBackup, int)
    if postBackup > 0:
        project.CreateVersion(f'POST_MTB_{datetime.now().strftime(r"%d%m%Y%H%M%S")}')


if __name__ == "__main__":
    main()
    print()
