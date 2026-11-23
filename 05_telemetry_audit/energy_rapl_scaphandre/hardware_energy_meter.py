from __future__ import annotations
import ctypes
from ctypes import wintypes
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))
import math_stats_core as msc

PDH_FMT_LARGE = 0x00000400
ERROR_SUCCESS = 0


class PDH_FMT_COUNTERVALUE_LARGE(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("largeValue", ctypes.c_longlong),
    ]


class RealHardwareEnergyMeter:
    """
    Zero-Estimation Hardware RAPL / Windows EMI Energy Profiler.
    Strictly reads physical silicon energy counters (PKG + DRAM only; excludes PP0/PP1/_total).
    Fails closed with RuntimeError if hardware counters are inaccessible.
    """

    def __init__(self) -> None:
        self.backend: str = ""
        self.kappa_mj_per_unit: float = 0.0
        self.max_counter: int = 0
        self.p_idle_mw: float = 0.0
        self._query_handle = wintypes.HANDLE()
        self._pkg_handle = wintypes.HANDLE()
        self._dram_handle = wintypes.HANDLE()
        self._has_dram = False
        self._linux_pkg_file: Path | None = None
        self._linux_dram_file: Path | None = None

        if sys.platform == "win32":
            self._init_windows_emi_pdh()
        elif sys.platform.startswith("linux"):
            self._init_linux_powercap()
        else:
            raise RuntimeError(f"Unsupported OS for bare-metal RAPL/EMI telemetry: {sys.platform}")

    def _init_windows_emi_pdh(self) -> None:
        self._pdh = ctypes.windll.pdh
        status = self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query_handle))
        if status != ERROR_SUCCESS:
            raise RuntimeError(f"PdhOpenQueryW failed with Win32 PDH code: 0x{status & 0xFFFFFFFF:08X}")

        pkg_path = r"\Energy Meter(rapl_package0_pkg)\Energy"
        status = self._pdh.PdhAddEnglishCounterW(
            self._query_handle, pkg_path, None, ctypes.byref(self._pkg_handle)
        )
        if status != ERROR_SUCCESS:
            self.close()
            raise RuntimeError(f"Hardware counter missing ({pkg_path}): 0x{status & 0xFFFFFFFF:08X}")

        dram_path = r"\Energy Meter(rapl_package0_dram)\Energy"
        st_dram = self._pdh.PdhAddEnglishCounterW(
            self._query_handle, dram_path, None, ctypes.byref(self._dram_handle)
        )
        self._has_dram = (st_dram == ERROR_SUCCESS)

        self.backend = "windows_emi_pdh_rapl"
        self.kappa_mj_per_unit = msc.KAPPA_WINDOWS_PWH_TO_MJ
        self.max_counter = msc.UINT64_MAX_COUNTER

        # Verify live non-zero silicon counter readout immediately (Fail-Closed check)
        raw_check = self.read_raw_counter()
        if raw_check <= 0:
            self.close()
            raise RuntimeError("Zero-Estimation Guard: Windows EMI RAPL counter returned <= 0.")

    def _init_linux_powercap(self) -> None:
        pkg = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
        max_f = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        if not (pkg.is_file() and os.access(pkg, os.R_OK) and max_f.is_file()):
            raise RuntimeError("Zero-Estimation Guard: Linux /sys/class/powercap/intel-rapl:0 is inaccessible.")
        self._linux_pkg_file = pkg
        self.max_counter = int(max_f.read_text(encoding="utf-8").strip())
        for sub in Path("/sys/class/powercap/intel-rapl:0").glob("intel-rapl:0:*"):
            name_f = sub / "name"
            if name_f.is_file() and name_f.read_text(encoding="utf-8").strip().lower() == "dram":
                dram_e = sub / "energy_uj"
                if dram_e.is_file() and os.access(dram_e, os.R_OK):
                    self._linux_dram_file = dram_e
                    self._has_dram = True
        self.backend = "linux_powercap_rapl"
        self.kappa_mj_per_unit = msc.KAPPA_LINUX_UJ_TO_MJ

    def read_raw_counter(self) -> int:
        if self.backend == "windows_emi_pdh_rapl":
            st = self._pdh.PdhCollectQueryData(self._query_handle)
            if st != ERROR_SUCCESS:
                raise RuntimeError(f"PdhCollectQueryData failed: 0x{st & 0xFFFFFFFF:08X}")
            pkg_val = PDH_FMT_COUNTERVALUE_LARGE()
            ctypes.oledll  # keep ctypes state warm
            st_pkg = self._pdh.PdhGetFormattedCounterValue(
                self._pkg_handle, PDH_FMT_LARGE, None, ctypes.byref(pkg_val)
            )
            if st_pkg != ERROR_SUCCESS or pkg_val.CStatus != ERROR_SUCCESS:
                raise RuntimeError(f"Failed reading rapl_package0_pkg: 0x{st_pkg & 0xFFFFFFFF:08X}")
            total_raw = int(pkg_val.largeValue)
            if self._has_dram:
                dram_val = PDH_FMT_COUNTERVALUE_LARGE()
                st_d = self._pdh.PdhGetFormattedCounterValue(
                    self._dram_handle, PDH_FMT_LARGE, None, ctypes.byref(dram_val)
                )
                if st_d == ERROR_SUCCESS and dram_val.CStatus == ERROR_SUCCESS and dram_val.largeValue >= 0:
                    total_raw += int(dram_val.largeValue)
            return total_raw

        assert self._linux_pkg_file is not None
        total_uj = int(self._linux_pkg_file.read_text(encoding="utf-8").strip())
        if self._has_dram and self._linux_dram_file is not None:
            total_uj += int(self._linux_dram_file.read_text(encoding="utf-8").strip())
        return total_uj

    def calibrate_idle_power(self, duration_sec: float = 3.0) -> float:
        if duration_sec < 0.5:
            raise ValueError("Idle calibration duration must be >= 0.5 seconds.")
        t0 = time.perf_counter()
        e0 = self.read_raw_counter()
        time.sleep(duration_sec)
        e1 = self.read_raw_counter()
        dt = time.perf_counter() - t0
        gross_mj = msc.compute_rapl_net_energy_mj(
            e_raw_t0=e0,
            e_raw_t1=e1,
            max_energy_counter=self.max_counter,
            p_idle_mw=0.0,
            delta_t_sec=dt,
            kappa_mj_per_unit=self.kappa_mj_per_unit,
        )
        self.p_idle_mw = float(gross_mj / dt)
        if not math.isfinite(self.p_idle_mw) or self.p_idle_mw <= 0.0:
            raise RuntimeError(f"Invalid calibrated idle power: {self.p_idle_mw} mW")
        return self.p_idle_mw

    def measure_window_net_mj(self, e_raw_t0: int, t0_perf: float) -> tuple[float, float, float]:
        e_raw_t1 = self.read_raw_counter()
        dt_sec = time.perf_counter() - t0_perf
        gross_mj = msc.compute_rapl_net_energy_mj(
            e_raw_t0=e_raw_t0,
            e_raw_t1=e_raw_t1,
            max_energy_counter=self.max_counter,
            p_idle_mw=0.0,
            delta_t_sec=dt_sec,
            kappa_mj_per_unit=self.kappa_mj_per_unit,
        )
        net_mj = msc.compute_rapl_net_energy_mj(
            e_raw_t0=e_raw_t0,
            e_raw_t1=e_raw_t1,
            max_energy_counter=self.max_counter,
            p_idle_mw=self.p_idle_mw,
            delta_t_sec=dt_sec,
            kappa_mj_per_unit=self.kappa_mj_per_unit,
        )
        return net_mj, gross_mj, dt_sec

    def close(self) -> None:
        if self.backend == "windows_emi_pdh_rapl" and self._query_handle:
            self._pdh.PdhCloseQuery(self._query_handle)
            self._query_handle = wintypes.HANDLE()