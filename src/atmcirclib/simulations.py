#!/usr/bin/env python
"""Classes to define individual simulation setups."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import pkgutil
import sys
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from copy import copy
from importlib import import_module
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import cast
from typing import Dict
from typing import Optional
from typing import overload
from typing import Protocol
from typing import SupportsIndex
from typing import Tuple
from typing import TypeVar
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import pandas as pd

# First-party
from atmcirclib.typing import PathLike_T

# Generic type aliases

# Time step notations
DDHH = Tuple[int, int]
DDHHMM = Tuple[int, int, int]
YYYYMMDDHH = Tuple[int, int, int, int]

# Raw time interval(s)
IntervalLike_T = Union[pd.Interval, Tuple[Union[DDHH, DDHHMM], Union[DDHH, DDHHMM]]]
IntervalsLike_T = Sequence[IntervalLike_T]
IntervalSLike_T = Union[IntervalLike_T, IntervalsLike_T]

# Raw output stream type(s)
OutputStreamTypeLike_T = Union["OutputStreamType", str]
OutputStreamTypesLike_T = Sequence[OutputStreamTypeLike_T]
OutputStreamTypeSLike_T = Union[OutputStreamTypeLike_T, OutputStreamTypesLike_T]

# Raw output streams dict
OutputStreamsDictLike_T = Dict[OutputStreamTypeSLike_T, IntervalSLike_T]
OutputStreamsLike_T = Union["OutputStreams", OutputStreamsDictLike_T]


class NamedObj(Protocol):
    """Protocol for objects with ``name`` property."""

    @property
    def name(self) -> str:
        """Define the name of the object."""
        ...


NamedObj_T = TypeVar("NamedObj_T", bound=NamedObj)


class NamedObjList(list[NamedObj_T]):
    """Extend ``UserList`` to get items by name in addition to index."""

    def find(self, name: str) -> NamedObj_T:
        """Find output stream type by name."""
        for type_ in self:
            if type_.name == name:
                return type_
        else:
            raise ValueError(f"no {type(self).__name__} object named '{name}'")

    @overload
    def __getitem__(self, idx_or_name: SupportsIndex) -> NamedObj_T:
        ...

    @overload
    def __getitem__(self, idx_or_name: slice) -> NamedObjList[NamedObj_T]:
        ...

    @overload
    def __getitem__(self, idx_or_name: str) -> NamedObj_T:
        ...

    def __getitem__(
        self, idx_or_name: Union[SupportsIndex, slice, str]
    ) -> Union[NamedObj_T, NamedObjList[NamedObj_T]]:
        """Get an item by index or name."""
        if isinstance(idx_or_name, str):
            name: str = idx_or_name
            return self.find(name)
        elif isinstance(idx_or_name, slice):
            slice_ = idx_or_name
            objs: list[NamedObj_T] = super().__getitem__(slice_)
            return type(self)(objs)
        else:
            idx: SupportsIndex = idx_or_name
            obj: NamedObj_T = super().__getitem__(idx)
            return obj

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{type(self).__name__}([\n " + "\n ".join(map(str, self)) + "\n])"


@dc.dataclass(unsafe_hash=True)
class OutputStreamType:
    """Type of simulation output stream.

    Properties:
        name: Name of the stream.

        file_tmpl: File template incl. path relative to simulation directory;
            lead time is encoded with '{YYYY}' for year, '{MM}' for months,
            '{DD}' for days, '{hh}' for hours, '{mm}' for minutes and '{ss}' for
            seconds; example: 'output/m2d/lfff{DD}{hh}{mm}{ss}.nc'.

        freq: Output frequency.

        color: Color of stream in plots.

        removed_files (optional): Files have been removed from disk.

    """

    name: str
    file_tmpl: str
    freq: pd.Timedelta
    color: str
    removed_files: bool = False

    def format_file(
        self, step: pd.Timestamp, start: Optional[pd.Timestamp] = None
    ) -> str:
        """Format the output file name at a given step.

        If a ``start`` time step is passed, the time step is relative to that.

        """
        if start is not None:
            if step < start:
                raise ValueError(f"step must not precede start: {step} < {start}")
            delta = step - start
            days = int(np.floor(delta.total_seconds() / 60 / 60 / 24))
            hours = int(np.floor(delta.total_seconds() / 60 / 60)) % 24
            minutes = int(np.floor(delta.total_seconds() / 60)) % 60
            seconds = delta.components.seconds
            return self.file_tmpl.format(
                DD=f"{days:02}",
                hh=f"{hours:02}",
                mm=f"{minutes:02}",
                ss=f"{seconds:02}",
            )
        return self.file_tmpl.format(
            YYYY=f"{step.year:04}",
            MM=f"{step.month:02}",
            DD=f"{step.day:02}",
            hh=f"{step.hour:02}",
            mm=f"{step.minute:02}",
            ss=f"{step.second:02}",
        )


class OutputStreamTypes(NamedObjList[OutputStreamType]):
    """Collection of output stream types."""

    def get_names(self) -> list[str]:
        """Get the names of all output stream types."""
        return [stream_type.name for stream_type in self]


class EmptyOutputStreamError(Exception):
    """Output stream does not contain any output intervals."""


@dc.dataclass(unsafe_hash=True)
class OutputStream:
    """Output stream of a given type with output intervals."""

    stream_type: OutputStreamType
    intervals: list[pd.Interval]

    def __post_init__(self) -> None:
        """Finalize initialization."""
        self._run: Optional[SimulationRun] = None
        self.check_intervals()

    def get_run(self) -> SimulationRun:
        """Get the associated simulation run, if there is one."""
        if self._run is None:
            raise Exception("output stream is not associated with a simulation run")
        return self._run

    def set_run(self, run: SimulationRun) -> None:
        """Associate the output stream with a simulation run."""
        self._run = run

    def get_start(self) -> pd.Timestamp:
        """Get earliest start across all output intervals."""
        try:
            return min(interval.left for interval in self.intervals)
        except ValueError as e:
            raise EmptyOutputStreamError(self.stream_type) from e

    def get_end(self) -> pd.Timestamp:
        """Get latest end across all output intervals."""
        try:
            return max(interval.right for interval in self.intervals)
        except ValueError as e:
            raise EmptyOutputStreamError(self.stream_type) from e

    def get_steps(self) -> pd.DatetimeIndex:
        """Get all output time steps in the output intervals."""
        steps = pd.DatetimeIndex([])
        for interval in self.intervals:
            steps = steps.append(
                pd.date_range(interval.left, interval.right, freq=self.stream_type.freq)
            )
        return steps

    def get_all_steps(self) -> pd.DatetimeIndex:
        """Get all output time steps without gaps, regardless of intervals."""
        return pd.date_range(
            self.get_start(), self.get_end(), freq=self.stream_type.freq
        )

    def get_full_path(self, step: Optional[pd.Timestamp] = None) -> Path:
        """Get the full path to the stream's output directory."""
        if step is None:
            step = self.get_start()
        return Path(
            self.get_run().get_full_path() / self.stream_type.format_file(step)
        ).parent

    def format_file_path(
        self,
        step: pd.Timestamp,
        start: Optional[pd.Timestamp] = None,
        *,
        check_exists: bool = True,
    ) -> Path:
        """Format the full output file path at a given time step."""
        path = self.get_run().get_full_path() / self.stream_type.format_file(
            step, start
        )
        if check_exists and not path.exists():
            raise FileNotFoundError(path)
        return path

    def collect_files(
        self,
        condition: Optional[Callable[[Path], bool]] = None,
        *,
        exist: Optional[bool] = None,
    ) -> list[Path]:
        """Collect all output files, optionally meeting a condition.

        Args:
            condition (optional): Arbitrary condition applied to each file path.

            exist (optional): If true/false, only collect files that do/don't
                exist on disk.

        """
        if self.stream_type.removed_files:
            return []
        try:
            steps = self.get_steps()
        except EmptyOutputStreamError:
            return []
        start = self.get_run().get_start()
        paths: list[Path] = []
        for step in steps:
            path = self.format_file_path(step, start=start, check_exists=False)
            if condition is not None and not condition(path):
                continue
            if exist is not None:
                if (exist and not path.exists()) or (not exist and path.exists()):
                    continue
            paths.append(path)
        return paths

    def check_intervals(self, steps: Optional[pd.DatetimeIndex] = None) -> None:
        """Check that the bounds of all intervals coincide with time steps."""
        if len(self.intervals) == 0:
            return
        if steps is None:
            steps = self.get_all_steps()
        for interval in self.intervals:
            if interval.left not in steps:
                raise Exception(
                    f"inconsistent output interval {interval}: start '{interval.left}'"
                    f" not among time steps: " + ", ".join(steps.format())
                )
            if interval.right not in steps:
                raise Exception(
                    f"inconsistent output interval {interval}: end '{interval.right}'"
                    f" not among time steps: " + ", ".join(steps.format())
                )

    @classmethod
    def create(
        cls,
        stream_type: OutputStreamTypeLike_T,
        intervals: IntervalSLike_T,
        *,
        output_stream_types: Optional[OutputStreamTypes] = None,
        start: Optional[pd.Timestamp] = None,
    ) -> OutputStream:
        """Create an instance of ``OutputStream`` from raw input."""
        if not isinstance(stream_type, OutputStreamType):
            if output_stream_types is None:
                raise ValueError(
                    "must pass output_stream_types to convert stream_type from type"
                    f" {type(stream_type).__name__} to OutputStreamType"
                )
            stream_type = output_stream_types[stream_type]
        if intervals is None:
            intervals = []
        else:
            try:
                interval = init_interval(intervals, start)
            except ValueError:
                intervals = [
                    init_interval(raw_interval, start) for raw_interval in intervals
                ]
            else:
                intervals = [interval]
            intervals = [copy(interval) for interval in intervals]
        return cls(stream_type=stream_type, intervals=intervals)


class OutputStreams(list[OutputStream]):
    """Collection of output streams."""

    def get_stream_types(self) -> OutputStreamTypes:
        """Get the types of all output streams."""
        return OutputStreamTypes([stream.stream_type for stream in self])

    def get_start(self) -> pd.Timestamp:
        """Get earliest start across output streams."""
        try:
            return min(stream.get_start() for stream in self if stream.intervals)
        except ValueError as e:
            raise EmptyOutputStreamError(f"all streams in {self}") from e

    def get_end(self) -> pd.Timestamp:
        """Get latest end across output streams."""
        try:
            return max(stream.get_end() for stream in self if stream.intervals)
        except ValueError as e:
            raise EmptyOutputStreamError(f"all streams in {self}") from e

    def get_all_steps(self) -> pd.DatetimeIndex:
        """Get all time steps; only works if all streams are of the same type."""
        stream_types = set(self.get_stream_types())
        if len(stream_types) > 1:
            raise Exception(
                "can only get steps if all streams are of the same type"
                f"; found {len(stream_types)} types: "
                + ", ".join([stream_type.name for stream_type in stream_types])
            )
        stream_type = next(iter(stream_types))
        all_steps = pd.date_range(
            self.get_start(), self.get_end(), freq=stream_type.freq
        )
        for stream in self:
            stream.check_intervals(all_steps)
        return all_steps

    def count_per_step(self) -> tuple[npt.NDArray[np.int_], pd.DatetimeIndex]:
        """Count the number of streams overing each step."""
        steps = self.get_all_steps()
        counts = np.zeros(steps.size, np.int32)
        for stream in self:
            for interval in stream.intervals:
                i0 = steps.get_loc(interval.left)
                i1 = steps.get_loc(interval.right)
                counts[i0 : i1 + 1] += 1
        return counts, steps

    def set_run(self, run: SimulationRun) -> None:
        """Associate all output streams with a simulation run."""
        for stream in self:
            stream.set_run(run)

    @classmethod
    def create(
        cls,
        output: OutputStreamsLike_T,
        *,
        output_stream_types: Optional[OutputStreamTypes] = None,
        start: Optional[pd.Timestamp] = None,
    ) -> OutputStreams:
        """Create an instance of ``OutputStreams`` from raw input."""
        if isinstance(output, OutputStreams):
            return copy(output)
        streams = cls()
        stream_type_s: OutputStreamTypeSLike_T
        stream_type: OutputStreamTypeLike_T
        interval_s: IntervalSLike_T
        for stream_type_s, interval_s in output.items():
            if isinstance(stream_type_s, str) or not isinstance(
                stream_type_s, Collection
            ):
                stream_type_s = [stream_type_s]
            for stream_type in stream_type_s:
                try:
                    stream = OutputStream.create(
                        stream_type,
                        interval_s,
                        output_stream_types=output_stream_types,
                        start=start,
                    )
                except Exception as e:
                    if isinstance(stream_type, str):
                        name = stream_type
                        freq = "N/A"
                    else:
                        name = stream_type.name
                        freq = str(stream_type.freq).replace("0 days ", "") + " hours"
                    intervals = (
                        f"{interval_s}"
                        if isinstance(interval_s, list)
                        else f"[{interval_s}]"
                    )
                    raise Exception(
                        f"error creating '{name}' output stream (frequency: {freq})"
                        f"in interval(s) {intervals} (run start: {start})"
                    ) from e
                else:
                    streams.append(stream)
        return streams


@dc.dataclass(unsafe_hash=True)
class SimulationRunEndType:
    """Type of end of a simulation, like success or crash."""

    name: str
    color: str


class SimulationRunEndTypes(NamedObjList[SimulationRunEndType]):
    """A collection of simulation run end types."""


class SimulationRun:
    """An individual simulation run."""

    def __init__(
        self,
        simulation: Optional[Simulation] = None,
        *,
        path: Optional[PathLike_T] = None,
        rel_path: Optional[PathLike_T] = None,
        output: Optional[OutputStreamsLike_T] = None,
        end_rel: Optional[Union[pd.Timedelta, DDHH, DDHHMM]] = None,
        end_type: Optional[Union[SimulationRunEndType, str]] = None,
        simulation_run_end_types: Optional[SimulationRunEndTypes] = None,
    ) -> None:
        """Create an instance of ``SimulationRun``."""
        self._simulation: Optional[Simulation] = None
        if simulation is not None:
            self.link_simulation(simulation)
        self.abs_path: Optional[Path] = None if path is None else Path(path)
        self.rel_path: Optional[Path] = None if rel_path is None else Path(rel_path)
        try:
            streams = OutputStreams.create(output or {}, start=self.get_start())
        except Exception as e:
            path = self.abs_path or self.rel_path
            raise Exception(
                f"error creating output streams for run at path: {path}"
            ) from e
        self.output: OutputStreams = streams
        self.end_rel: pd.Timedelta = self._init_end_rel(
            end_rel, self.output, self.get_start()
        )
        self.end_type: SimulationRunEndType = self._init_end_type(
            end_type, simulation_run_end_types
        )

        self.output.set_run(self)
        self.label: str = self._init_label([self.abs_path, self.rel_path])
        self.end: pd.Timestamp = self.get_simulation().get_start() + self.end_rel
        self.write_start: Optional[pd.Timestamp]
        self.write_end: Optional[pd.Timestamp]
        self.write_duration: Optional[pd.Timedelta]
        if not self.output:
            self.write_start = None
            self.write_end = None
            self.write_duration = None
        else:
            self.write_start = min([i.left for s in self.output for i in s.intervals])
            self.write_end = max([i.right for s in self.output for i in s.intervals])
            self.write_duration = self.write_end - self.write_start

    def get_simulation(self) -> Simulation:
        """Get simulation if one is linked, otherwise raise an exception."""
        if self._simulation is None:
            raise Exception(f"run not linked to simulation: run={self}")
        return self._simulation

    def link_simulation(self, sim: Simulation, register: bool = True) -> None:
        """Link a simulation to the run."""
        if self._simulation == sim:
            pass
        elif self._simulation is None:
            self._simulation = sim
        else:
            raise ValueError(
                "run already linked to different simulation"
                f"\nrun={self}\nrun._simulation={self._simulation}\n{sim=}"
            )
        if register:
            sim.register_run(self, link=False)

    def get_start(self) -> pd.Timestamp:
        """Get the start of the simulation run."""
        return self.get_simulation().get_start()

    def get_full_path(self) -> Path:
        """Get the absolute path to the simulation run."""
        path: Optional[Path] = None
        error = ""
        if self.abs_path is not None:
            path = self.abs_path
        elif self._simulation is not None:
            if self.rel_path is not None:
                path = self._simulation.path / self.rel_path
            else:
                error = "_rel_path is None"
        else:
            error = "_abs_path and simulation are both None"
        if path is None:
            raise Exception(
                "either initialize with absolute path; or initialize with relative path"
                f"and register simulation (error: {error})"
            )
        return path

        def init_path(
            path: Optional[PathLike_T], rel_path: Optional[PathLike_T]
        ) -> Path:
            if path is None and rel_path is None:
                raise ValueError("path and rel_path are both None")
            elif rel_path is not None:
                return Path(rel_path)
            elif path is not None:
                return Path(path)
            else:
                raise ValueError("one of path and rel_path must be None")

    def exists(self) -> bool:
        """Check if simulation run directory exists on disk."""
        return self.get_full_path().exists()

    def __repr__(self) -> str:
        """Return a string representation."""
        path = self.get_full_path()
        return f"{type(self).__name__}('{path}')"

    @staticmethod
    def _init_end_rel(
        raw_end_rel: Optional[Union[pd.Timedelta, DDHH, DDHHMM]],
        output: OutputStreams,
        start: pd.Timestamp,
    ) -> pd.Timedelta:
        """Initialize ``end_rel``."""
        if raw_end_rel is None:
            if not output:
                raw_end_rel = pd.Timedelta(0)
            else:
                last_output = max(
                    interval.right for stream in output for interval in stream.intervals
                )
                raw_end_rel = last_output - start
        return init_timedelta(raw_end_rel)

    @staticmethod
    def _init_end_type(
        raw_end_type: Optional[Union[SimulationRunEndType, str]],
        simulation_run_end_types: Optional[SimulationRunEndTypes],
    ) -> SimulationRunEndType:
        """Initialize ``end_type``."""
        if isinstance(raw_end_type, SimulationRunEndType):
            return raw_end_type
        if simulation_run_end_types is None:
            raise ValueError(
                "must pass simulation_run_end_types if end_type is not of type"
                " SimulationRunEndType"
            )
        return simulation_run_end_types[raw_end_type or "success"]

    @staticmethod
    def _init_label(paths: Collection[Optional[Path]]) -> str:
        label = ""
        for path in paths:
            if path is not None:
                if not label:
                    label = path.name
                elif label != path.name:
                    raise ValueError(
                        f"inconsistent labels: {label} != {path.name}; paths: "
                        + ", ".join(map(str, paths))
                    )
        if not label:
            raise ValueError(f"could not derive label from paths: {paths}")
        return label


class Simulation:
    """A simulation comprised of one or more simulation runs."""

    def __init__(
        self,
        path: PathLike_T,
        start: Union[pd.Timestamp, YYYYMMDDHH],
        runs: Optional[Sequence[SimulationRun]] = None,
    ) -> None:
        """Create an instance of ``Simulation``."""
        self._start: pd.Timestamp = init_timestamp(start)
        self._runs: list[SimulationRun] = []
        for run in runs or []:
            self.register_run(run)
        self.path: Path = Path(path)

    def get_runs(self) -> list[SimulationRun]:
        """Return registered simulation runs."""
        return list(self._runs)

    def register_run(
        self, run: SimulationRun, link: bool = True, registered_ok: bool = True
    ) -> None:
        """Register a simulation run object and link it to the simulation."""
        if run in self._runs:
            if not registered_ok:
                raise ValueError(
                    f"run already registered in simulation\nsimulation={self}\n{run=}\n"
                )
        else:
            self._runs.append(run)
        if link:
            run.link_simulation(self, register=False)

    def get_start(self) -> pd.Timestamp:
        """Get start of simulation."""
        return self._start

    def get_end(self) -> pd.Timestamp:
        """Get end of simulation, i.e., latest run end."""
        return max([run.end for run in self.get_runs()])

    def collect_output_stream_types(self) -> list[OutputStreamType]:
        """Collect output stream types across all runs."""
        return list(self.collect_output_streams_by_type().keys())

    def collect_output_streams_by_type(
        self,
    ) -> dict[OutputStreamType, OutputStreams]:
        """Collect output streams of all runs."""
        streams_by_type: dict[OutputStreamType, OutputStreams] = {}
        for run in self.get_runs():
            for stream in run.output:
                if stream.stream_type not in streams_by_type:
                    streams_by_type[stream.stream_type] = OutputStreams()
                streams_by_type[stream.stream_type].append(stream)
        return streams_by_type

    def find_redundant_output(
        self,
        check_exists: bool = True,
    ) -> dict[OutputStreamType, dict[pd.Timestamp, list[Path]]]:
        """Find redundant output files produced by runs of a simulation."""
        if not self.get_runs():
            return {}
        multi_steps: dict[OutputStreamType, dict[pd.Timestamp, list[Path]]] = {}
        for stream_type, streams in self.collect_output_streams_by_type().items():
            if stream_type.removed_files:
                continue
            try:
                counts, steps = streams.count_per_step()
            except EmptyOutputStreamError:
                continue
            idcs = np.where(steps[counts > 1])[0]
            if idcs.size == 0:
                continue
            multi_steps[stream_type] = {}
            for idx in idcs:
                step = steps[idx]
                multi_steps[stream_type][step] = []
                for stream in streams:
                    if any(step in interval for interval in stream.intervals):
                        file_path = stream.format_file_path(
                            step, start=self.get_start(), check_exists=check_exists
                        )
                        multi_steps[stream_type][step].append(file_path)
        return multi_steps

    def contains_run(self, *, path: Optional[PathLike_T] = None) -> bool:
        """Check if the simulation contains a given run."""
        if path is not None:
            path = Path(path).resolve()
            if not path.exists():
                raise ValueError(f"path doesn't exist: {path}")
        for run in self.get_runs():
            if path is not None:
                run_path = run.get_full_path().resolve()
                if not run_path.exists():
                    print(
                        "warning [Simulation.contains_run]: skipping run"
                        f": path doesn't exist: {run_path}",
                        file=sys.stderr,
                    )
                    continue
                if run_path.samefile(path):
                    return True
            else:
                # Make path implicitly mandatory in preparation for additional
                # criteria, at least one which would be mandatory
                raise ValueError("must pass path")
        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(path={self.path}, start={self.get_start()}, runs=["
            + ", ".join(map(str, self.get_runs()))
            + "])"
        )


class Simulations(list[Simulation]):
    """Multiple simulations."""

    def collect_runs(
        self,
        condition: Optional[Callable[[SimulationRun], bool]] = None,
        *,
        exist: Optional[bool] = None,
    ) -> list[SimulationRun]:
        """Collect all runs across all simulations, optionally conditionally.

        Args:
            condition (optional): Arbitrary condition applied to each run path.

            exist (optional): If true/false, only collect runs that do/don't
                exist on disk.

        """
        runs: list[SimulationRun] = []
        for sim in self:
            for run in sim.get_runs():
                if condition is not None and not condition(run):
                    continue
                if exist is not None and (
                    (exist and not run.exists()) or (not exist and run.exists())
                ):
                    continue
                runs.append(run)
        return runs

    def collect_output_files(
        self,
        *,
        streams: Optional[Collection[str]] = None,
        exist: Optional[bool] = None,
    ) -> list[list[list[Path]]]:
        """Collect output files for each simulation.

        Keyword Arguments:
            streams (optional): Collection of output streams to which collected
                output files must belong.

            exist (optional): If true/false, only collect output files that do/don't
                exist on disk.

        """
        paths_sims: list[list[list[Path]]] = []
        for sim in self:
            paths_sim = []
            for run in sim.get_runs():
                for stream in run.output:
                    if streams is not None and stream.stream_type.name not in streams:
                        continue
                    paths_stream = []
                    for path in OutputStream.collect_files(stream, exist=exist):
                        paths_stream.append(path)
                    if paths_stream:
                        paths_sim.append(paths_stream)
            if paths_sim:
                paths_sims.append(paths_sim)
        return paths_sims

    def collect_redundant_output_files(self) -> list[list[Path]]:
        """Collect paths of redundant output files for each simulation."""
        paths_sims: list[list[Path]] = []
        for sim in self:
            paths_sims.append([])
            for paths_by_step in sim.find_redundant_output(check_exists=False).values():
                for paths_sim in paths_by_step.values():
                    for path in paths_sim:
                        paths_sims[-1].append(path)
        return paths_sims

    def contains_run(self, *, path: Optional[PathLike_T] = None) -> bool:
        """Check if any simulation contains a given run."""
        for simulation in self:
            if simulation.contains_run(path=path):
                return True
        return False

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{type(self).__name__}([\n " + "\n ".join(map(str, self)) + "\n])"

    @classmethod
    def from_modules(
        self, mods: Modules, attr_name: str = "simulations"
    ) -> Simulations:
        """Create a new instance from modules."""
        sims = Simulations()
        for module in mods:
            sims.extend(getattr(module, attr_name))
        return sims


class SimulationScanner:
    """Scan the file system to find simulation runs."""

    def find_simulation_run_paths(self, root: Path) -> list[Path]:
        """Find paths to all simulation runs under ``root``."""
        return [Path(p).parent for p in root.rglob("INPUT_ORG")]


class Modules(list[ModuleType]):
    """A list of Python modules."""

    @classmethod
    def from_package(
        cls, pkg: ModuleType, cond: Optional[Callable[[ModuleInfo], bool]] = None
    ) -> Modules:
        """Collect all modules in a package that meet an optional condition."""
        pkg_path = pkg.__path__
        modules = cls()
        for module_info in pkgutil.walk_packages(pkg_path):
            if cond is None or cond(module_info):
                module = import_module(f"{pkg.__package__}.{module_info.name}")
                modules.append(module)
        return modules


def init_timestamp(val: Union[pd.Timestamp, YYYYMMDDHH]) -> pd.Timestamp:
    """Initialize a timestamp object."""
    if isinstance(val, pd.Timestamp):
        return val
    yyyy, mm, dd, hh = val
    return pd.Timestamp(yyyy, mm, dd, hh)


def init_timedelta(val: Union[pd.Timedelta, DDHH, DDHHMM]) -> pd.Timedelta:
    """Initialize a timedelta object."""
    if isinstance(val, pd.Timedelta):
        return val
    if not isinstance(val, Sequence) or not all(isinstance(i, int) for i in val):
        raise ValueError(f"cannot create Timedelta from {val}")
    try:
        dd, hh, mm = cast(DDHHMM, val)
    except ValueError:
        dd, hh = cast(DDHH, val)
        mm = 0
    try:
        return pd.Timedelta(days=dd, hours=hh, minutes=mm)
    except TypeError as e:
        raise ValueError(f"cannot create Timedelta from ({dd}, {hh}, {mm})") from e


def init_interval(
    val: IntervalLike_T,
    start: Optional[pd.Timestamp] = None,
) -> pd.Interval:
    """Initialize a time interval object."""
    if isinstance(val, pd.Interval):
        return copy(val)
    if start is None:
        raise ValueError("must pass start if val is not already an Interval object")
    left, right = map(init_timedelta, val)
    left += start
    right += start
    return pd.Interval(left, right, closed="both")
