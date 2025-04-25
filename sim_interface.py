"""
This module contains classes and functions for interfacing with Powerfactory.
Powerfactory is interfaced through powfacpy.
"""

from __future__ import annotations
import globals
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple, Optional, Callable
from math import isnan
from warnings import warn
from os.path import join, split, splitext, exists, abspath
from os import mkdir
import pandas as pd
from powfacpy.base.active_project import ActiveProject
from powfacpy import PFGeneral
import powerfactory

try:
    import powerfactory as pf  # type: ignore

    print(f"Imported powerfactory module from {pf.__file__}")
except ImportError:
    warn("sim_interface.py: Powerfactory module not found.")


MEAS_FILE_FOLDER: str = "MTB_files"  # constant

pf_time_offset: float = 0.0


class Waveform(ABC):
    @property
    @abstractmethod
    def s0(self) -> float: ...

    @abstractmethod
    def add(self, t: float, s: float, r: float = 0.0) -> None: ...


class Piecewise(Waveform):
    """
    Piecewise defined waveform. At every defined point in time the waveform is set to "s" and continues with gradient "r".
    Only used in the signal type channel.
    """

    def __init__(self, s0: float) -> None:
        self.__t__: List[float] = [0.0]
        self.__s__: List[float] = [s0]
        self.__r__: List[float] = [0.0]

    def add(self, t: float, s: float, r: float = 0.0) -> None:
        if isnan(t):
            raise ValueError("t must be a float")

        assert len(self.__t__) == len(self.__s__) == len(self.__r__)
        assert len(self.__t__) > 0
        assert self.__t__[0] == 0.0

        if t < 0.0:
            if isnan(s):
                raise ValueError("Initial value of piecewise must be a float")
            self.__s__[0] = s
            return

        i = len(self.__t__) - 1

        while True:
            if t >= self.__t__[i]:
                if t > self.__t__[i] or t == 0.0:
                    newIndex = i + 1
                else:
                    newIndex = i

                if t > self.__t__[i]:
                    donorIndex = i
                else:
                    donorIndex = max(i - 1, 0)

                if isnan(s):
                    dt = t - self.__t__[donorIndex]
                    s = self.__s__[donorIndex] + self.__r__[donorIndex] * dt

                if isnan(r):
                    r = self.__r__[donorIndex]

                self.__t__.insert(newIndex, t)
                self.__s__.insert(newIndex, s)
                self.__r__.insert(newIndex, r)
                break
            i -= 1

    def t_pf(self, minLength: int = 0) -> List[float]:
        return self.__tf__(minLength, pf_time_offset)

    def __tf__(self, minLength: int = 0, offset: float = 0.0) -> List[float]:
        _t = [0.0] + [t + offset for t in self.__t__[1:]]
        if len(_t) >= minLength:
            return _t
        else:
            return _t + (minLength - len(_t)) * [0.0]

    def s(self, minLength: int = 0) -> List[float]:
        if len(self.__s__) >= minLength:
            return self.__s__
        else:
            return self.__s__ + (minLength - len(self.__s__)) * [0.0]

    def r(self, minLength: int = 0) -> List[float]:
        if len(self.__r__) >= minLength:
            return self.__r__
        else:
            return self.__r__ + (minLength - len(self.__r__)) * [0.0]

    @property
    def len(self):
        return len(self.__t__)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return (
                self.__t__ == other.__t__
                and self.__s__ == other.__s__
                and self.__r__ == other.__r__
            )
        else:
            return False

    @property
    def s0(self) -> float:
        return self.__s__[0]


class Recorded(Waveform):
    """
    Waveform defined in specified column in file. Time must be first column (column = 0). Supports powerfactory ElmFile format.
    Only used in signal type channel.
    """

    def __init__(self, path: str, column: int, scale: float = 1.0) -> None:

        self.__path__: str = path
        self.__column__: int = column
        self.__pf__: bool = True
        self.__scale__: float = scale
        self.__pf_path__: Optional[str] = None
        self.__pfLen__: float = 0.0
        self.__s0__: float = 0.0
        self.__loadFile__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.__pf_path__ == other.__pf_path__
        else:
            return False

    def __loadFile__(self) -> None:
        if not self.__pf__:
            return None

        _, pathFilename = split(self.__path__)
        pathName, pathExtension = splitext(pathFilename)

        reader = open(self.__path__, "r")

        if pathExtension.lower() == ".meas" or pathExtension.lower() == ".out":
            lineBuffer = reader.readlines()
            data: List[List[float]] = []

            def parseLine(
                line: str, linenr: int, column: int, file: str
            ) -> List[float]:
                floatBuffer: str = ""
                line += "\n"
                colNr: int = -1
                time: float = 0.0

                for c in line:
                    if not c in [",", " ", "\t", "\n"]:
                        floatBuffer += c
                    else:
                        if len(floatBuffer) > 0:
                            colNr += 1
                            try:
                                if colNr == 0:
                                    time = float(floatBuffer)
                                elif colNr == column:
                                    return [time, float(floatBuffer)]
                            except ValueError:
                                raise RuntimeError(
                                    f'Could not parse line nr: {linenr} in "{file}". Value "{floatBuffer}" not understandable as float. Exiting.'
                                )
                            floatBuffer = ""

                raise RuntimeError(
                    f'Could not parse line nr: {linenr} in "{file}". Column {column} not found.'
                )

            i = 2
            for line in lineBuffer[1:]:
                data.append(parseLine(line, i, self.__column__, self.__path__))
                i += 1

            df: pd.DataFrame = pd.DataFrame(data)

        elif pathExtension.lower() == ".csv":
            df: pd.DataFrame = pd.read_csv(self.__path__, sep=";", decimal=".", header=None, skiprows=1)  # type: ignore
        else:
            raise RuntimeError(f"Unknown filetype of: {self.__path__}.")

        # Data is loaded
        df = df.set_index(0)  # type: ignore
        df.sort_index(ascending=True, inplace=True)  # type: ignore
        df = df * self.__scale__
        self.__s0__ = float(df.iloc[0, 0])  # type: ignore
        time = df.index  # type: ignore

        if not exists(MEAS_FILE_FOLDER):
            mkdir(MEAS_FILE_FOLDER)

        if self.__pf__:
            df.index = df.index + pf_time_offset  # type: ignore
            df.rename(index={df.index[0]: time[0]}, inplace=True)  # type: ignore

            recFilePath = join(
                MEAS_FILE_FOLDER,
                f"{pathName}_{self.__column__}_{self.__scale__}_{pf_time_offset}.meas",
            )
            measData: str = df.to_csv(None, sep=" ", header=False, index_label=False).replace("\r\n", "\n")  # type: ignore
            measData = "1\n" + measData
            f = open(recFilePath, "w")
            f.write(measData)
            f.close()
            self.__pf_path__ = recFilePath
            self.__pfLen__ = df.index[-1]  # type: ignore

    @property
    def pf_len(self):
        if self.__pf_path__ is None:
            warn(
                f"Recorded waveform (source: {self.__path__}) pfLen call with pfPath set to None. Returning 0.0."
            )
        return self.__pfLen__

    @property
    def s0(self) -> float:
        return self.__s0__

    @property
    def pf_path(self):
        if self.__pf_path__ is None:
            raise RuntimeError("pfPath not set.")
        return self.__pf_path__

    def add(self, t: float, s: float, r: float = 0) -> None:
        warn(
            f"Recorded waveform (source: {self.__path__}) .add method called. Ignoring."
        )


class Channel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...


class PfApplyable(ABC):
    @abstractmethod
    def apply_to_pf(self, rank: int) -> None: ...

    def parse_target_string(self, target: str):
        """If target string starts with $studycase\\$, the object will be taken from active study case.
        Otherwise, the object is found in the model test bench grid folder.

        Args:
            target (str): Target string

        Returns:
            PFGeneral: Target object
        """
        if target.startswith("$studycase$\\"):
            classname = target.split("\\")[-1]
            return globals.pfp.app.GetFromStudyCase(classname)
        else:
            return globals.pfp.get_unique_obj(
                target, include_subfolders=True, parent_folder=globals.mtb
            )

    def parse_attribute_value_type(
        self, target: PFGeneral, attribute: str, value: Union[int, str, float]
    ):
        """Returns value of attribute of target in the correct data type.

        Args:
            target (PFGeneral): Target PowerFactory object
            attribute (str): Attribute name
            value (Union[int, str, float]): Attribute value

        Raises:
            KeyError: If attribute not available for target.
            TypeError: If value type other than str is given to be converted into PFGeneral.
            RuntimeError: If attribute type not know.

        Returns:
            Union[int, str, float, PFGeneral]: Value in correct type.
        """
        attribute_types = powerfactory.DataObject.AttributeType
        this_attribute_type = target.GetAttributeType(attribute)

        if this_attribute_type == attribute_types.INVALID:
            raise KeyError(
                f"Attribute {attribute} not found in object {target.GetFullName(0)}"
            )

        if (
            this_attribute_type == attribute_types.OBJECT
            or this_attribute_type == attribute_types.OBJECT_VEC
        ):
            if not isinstance(value, str):
                raise TypeError(
                    "Attribute is of type OBJECT or OBJECT_VEC. Value must be a string containing the path to the set object."
                )
            return globals.pfp.get_unique_obj(
                value, parent_folder=globals.mtb, include_subfolders=True
            )

        elif (
            this_attribute_type == attribute_types.STRING
            or this_attribute_type == attribute_types.STRING_VEC
        ):
            return str(value)

        elif (
            this_attribute_type == attribute_types.DOUBLE
            or this_attribute_type == attribute_types.DOUBLE_MAT
            or this_attribute_type == attribute_types.DOUBLE_VEC
        ):
            return float(value)

        elif (
            this_attribute_type == attribute_types.INTEGER
            or this_attribute_type == attribute_types.INTEGER_VEC
            or this_attribute_type == attribute_types.INTEGER64
            or this_attribute_type == attribute_types.INTEGER64_VEC
        ):
            return int(value)

        else:
            raise RuntimeError(
                f"Attribute {attribute} of type {this_attribute_type} not supported."
            )


# CHANNEL TYPES
class Constant(Channel, PfApplyable):
    """
    Constant (irrespective of rank and time) value passed to Powerfactory.
    """

    def __init__(
        self,
        name: str,
        value: Union[float, int, bool],
    ) -> None:
        self.__name__ = name
        self.__value__: float = float(value)
        self.__pf_subs__: List[Tuple[str, str]] = []
        self.__signal_template__: str = f"""subroutine {name}_const(y)
    implicit none
    real, intent(out) :: y

    y = {value}

end subroutine {name}_const"""

    @property
    def value(self) -> Union[float, int]:
        return self.__value__

    @property
    def name(self) -> str:
        return self.__name__

    def add_pf_sub(self, target: str, attribute: str) -> None:
        """Add PowerFactory subscription to Constant.

        Subscribed target attributes will be assigned Constant.value.

        Args:
            target (Union[PFGeneral, str]): PF object with the attribute to be defined by Constant.
            attribute (str): Attribute of target to be defined by Constant.
        """
        if not (target, attribute) in self.__pf_subs__:
            self.__pf_subs__.append((target, attribute))

    @property
    def pf_subs(self):
        """Attributes that are subscribed (i.e. whose value is defined by) to Constant."""

        return self.__pf_subs__

    def apply_to_pf(self, rank: int) -> None:
        """Write Constant.value to all subscibed targets/attributes.

        Args:
            rank (int): Rank of the test case.

        Returns: None
        """

        for target, attrib in self.__pf_subs__:
            target = self.parse_target_string(target)
            attrib_value = self.parse_attribute_value_type(target, attrib, self.value)
            globals.pfp.set_attr(target, {attrib: attrib_value})


class Signal(Channel, PfApplyable):
    """
    Dynamic value both in respect to time and rank (= test case) passed to Powerfactory.
    Each rank can either contain a piecewise defined waveform or a recorded waveform.
    """

    def __init__(self, name: str) -> None:
        self.__name__: str = name
        self.__waveforms__: Dict[int, Waveform] = dict()
        self.__pf_subs_value__: List[
            Tuple[PFGeneral, str, Optional[Callable[[Signal, float], float]]]
        ] = []
        self.__pf_subs_value0__: List[
            Tuple[PFGeneral, str, Optional[Callable[[Signal, float], float]]]
        ] = []
        self.__pf_subs_ramp__: List[
            Tuple[PFGeneral, str, Optional[Callable[[Signal, float], float]]]
        ] = []
        self.__pf_subs_mode__: List[
            Tuple[PFGeneral, str, Optional[Callable[[Signal, float], float]]]
        ] = []
        self.__pfp__: Optional[ActiveProject] = globals.pfp
        self.__elmfile__: Optional[PFGeneral] = None

    @property
    def name(self):
        return self.__name__

    @property
    def elmfile(self):
        return self.__elmfile__

    def set_elmfile(self, file: str) -> None:
        self.__elmfile__ = file

    def __setitem__(self, rank: int, wave: Union[Waveform, float, int]) -> None:
        if isinstance(wave, float) or isinstance(wave, int):
            wave = Piecewise(float(wave))
        self.__waveforms__[rank] = wave

    def __getitem__(self, rank: int) -> Waveform:
        return self.__waveforms__[rank]

    def __array_size__(self) -> int:
        max_length = -1
        for rank in self.ranks:
            wf = self[rank]
            if isinstance(wf, Piecewise):
                max_length = max(wf.len, max_length)

        return max_length

    @property
    def ranks(self):
        return self.__waveforms__.keys()

    def add_pf_sub_value(
        self,
        target: str,
        attribute: str,
        func: Optional[Callable[[Signal, float], float]] = None,
    ):
        if not (target, attribute, func) in self.__pf_subs_value__:
            self.__pf_subs_value__.append((target, attribute, func))

    def add_pf_sub_value0(
        self,
        target: str,
        attribute: str,
        func: Optional[Callable[[Signal, float], float]] = None,
    ):
        """Add PowerFactory subscription to ElmFile Signal.

        Subscribed target attributes will be assigned the waveform of Signal.
        It originates from a PowerFactory ElmFile object.

        Args:
            target (str): PF object with the attribute to be defined by Signal.
            attribute (str): Attribute of the target to be definded by Signal.
            func (Optional[Callable[[Signal, float], float]], optional): _description_. Defaults to None.
        """
        if not (target, attribute, func) in self.__pf_subs_value0__:
            self.__pf_subs_value0__.append((target, attribute, func))

    def add_pf_sub_ramp(
        self,
        target: str,
        attribute: str,
        func: Optional[Callable[[Signal, float], float]] = None,
    ):
        if not (target, attribute, func) in self.__pf_subs_ramp__:
            self.__pf_subs_ramp__.append((target, attribute, func))

    def add_pf_sub_mode(
        self,
        target: str,
        attribute: str,
        func: Optional[Callable[[Signal, float], float]] = None,
    ):
        if not (target, attribute, func) in self.__pf_subs_mode__:
            self.__pf_subs_mode__.append((target, attribute, func))

    def apply_to_pf(self, rank: int) -> None:
        _ = globals.pfp.app.GetFromStudyCase("ComInc").Execute()

        wf = self.__waveforms__[rank]
        print(self.name, ':')
        if isinstance(wf, Piecewise): # signal realized by events
            for target, attrib, func in self.__pf_subs_value__:
                target = self.parse_target_string(target)
                for event_number in range(wf.len):
                    event_time = wf.t_pf(0)[event_number]
                    if event_time == 0.0:
                        continue
                    event_value = wf.s(0)[event_number]
                    globals.pfdyn.create_dyn_sim_event(
                        f"{self.name}_value.EvtParam",
                        {
                            "p_target": target,
                            "time": event_time,
                            "variable": attrib,
                            "value": str(func(self, event_value) if func else event_value),
                        },
                        overwrite=False,
                    )

            for target, attrib, func in self.__pf_subs_ramp__:
                target = self.parse_target_string(target)
                for event_number in range(wf.len):
                    event_time = wf.t_pf(0)[event_number]
                    if event_time == 0.0:
                        continue
                    event_ramp = wf.r(0)[event_number]
                    globals.pfdyn.create_dyn_sim_event(
                        f"{self.name}_ramp.EvtParam",
                        {
                            "p_target": target,
                            "time": wf.t_pf(0)[event_number],
                            "variable": attrib,
                            "value": str(func(self, event_ramp) if func else event_ramp),
                        },
                        overwrite=False,
                    )

            if self.elmfile: # deactivate elmfile
                globals.pfp.set_attr(self.elmfile, {"e:outserv": 1, "e:f_name": ""})

        elif isinstance(wf, Recorded): # signal realized by elmfile
            if self.elmfile: # activate elmfile
                globals.pfp.set_attr(
                    self.elmfile, {"e:outserv": 0, "e:f_name": abspath(wf.pf_path)}
                )

        for target, attrib, func in self.__pf_subs_value0__:
            target = self.parse_target_string(target)
            attrib_value = func(self, wf.s0) if func else wf.s0
            attrib_value = self.parse_attribute_value_type(target, attrib, attrib_value)
            globals.pfp.set_attr(target, {attrib: attrib_value})
            print(f"    {target.loc_name}|{attrib} = {attrib_value}")

        for target, attrib, func in self.__pf_subs_mode__:
            target = self.parse_target_string(target)
            typ = 0.0 if isinstance(wf, Piecewise) else 1.0 if isinstance(wf, Recorded) else None
            assert typ is not None, "wf must be an instance of Piecewise or Recorded."
            attrib_value = func(self, typ) if func else typ
            attrib_value = self.parse_attribute_value_type(target, attrib, attrib_value)
            globals.pfp.set_attr(target, {attrib: attrib_value})
            print(f"    {target.loc_name}|{attrib} = {attrib_value}")


class String(Channel, PfApplyable):
    """
    String value, only dynamic in respect to rank, passed to Powerfactory.
    """

    def __init__(self, name: str) -> None:
        self.__name__: str = name
        self.__strings__: Dict[int, str] = dict()
        self.__pf_subs__: List[Tuple[str, str]] = []

    @property
    def name(self):
        return self.__name__

    def __getitem__(self, rank: int) -> str:
        return self.__strings__[rank]

    def __setitem__(self, rank: int, string: str) -> None:
        self.addRank(rank, string)

    def addRank(self, rank: int, string: str) -> None:
        self.__strings__[rank] = string

    @property
    def ranks(self):
        return self.__strings__.keys()

    def add_pf_sub(self, target: str, attribute: str):
        if not (target, attribute) in self.__pf_subs__:
            self.__pf_subs__.append((target, attribute))

    @property
    def pf_subs(self):
        return self.__pf_subs__

    def apply_to_pf(self, rank: int) -> None:
        if globals.pfp is None:
            warn(f"Powerfactory interface not set on string: {self.name}. Ignoring.")
            return None

        for target, attribute in self.__pf_subs__:
            target = self.parse_target_string(target)
            attrib_value = self.parse_attribute_value_type(
                target, attribute, self.__strings__[rank]
            )
            globals.pfp.set_attr(target, {attribute: attrib_value})


class PfObjRefer(String):
    """
    Powerfactory object reference dynamic in respect to rank. Reference defined as path relative to rootobject passed to .applyToPF function.
    """

    def apply_to_pf(self, rank: int) -> None:
        if globals.pfp is None:
            warn(
                f"Powerfactory interface not set on PfObjRefer: {self.name}. Ignoring."
            )
            return None

        if self.__strings__[rank] == "$nochange$":
            return None

        for target, attribute in self.__pf_subs__:
            target = self.parse_target_string(target)
            attrib_value = self.parse_attribute_value_type(
                target, attribute, self.__strings__[rank]
            )
            globals.pfp.set_attr(target, {attribute: attrib_value})


def apply_to_powerfactory(channels: List[Channel], rank: int):
    """
    Apply all channel setups in list to Powerfactory.
    """
    for channel in channels:
        if not isinstance(channel, PfApplyable):
            continue
        channel.apply_to_pf(rank)
        # globals.pfp.app.PrintPlain(f"EXECUTING:{channel.name}")#
        # globals.pfdyn.initialize_and_run_sim()#