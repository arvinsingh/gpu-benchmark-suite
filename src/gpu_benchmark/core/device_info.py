"""
Device information utilities for GPU benchmarking.
"""

import platform
import subprocess
from typing import Dict, List, Optional

import pynvml
import torch


class DeviceInfo:
    """Collect and provide GPU device information."""

    def __init__(self):
        """Initialize device info collector."""
        self._cuda_available = torch.cuda.is_available()
        if self._cuda_available:
            pynvml.nvmlInit()

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get number of available GPUs."""
        if not self._cuda_available:
            return 0
        return torch.cuda.device_count()

    def get_device_info(self, device_id: int = 0) -> Dict[str, any]:
        """Get comprehensive device information.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary containing device information
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA is not available")

        if device_id >= self.device_count:
            raise ValueError(f"Invalid device ID {device_id}")

        device_props = torch.cuda.get_device_properties(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # basic device info
        info = {
            "device_id": device_id,
            "name": device_props.name,
            "compute_capability": f"{device_props.major}.{device_props.minor}",
            "total_memory": device_props.total_memory,
            "multi_processor_count": device_props.multi_processor_count,
            "max_threads_per_multiprocessor": device_props.max_threads_per_multi_processor,
            "warp_size": device_props.warp_size,
        }

        # memory info
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.update(
                {
                    "memory_total": memory_info.total,
                    "memory_free": memory_info.free,
                    "memory_used": memory_info.used,
                }
            )
        except pynvml.NVMLError:
            pass

        # clock speeds
        try:
            info.update(
                {
                    "memory_clock": pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    ),
                    "graphics_clock": pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    ),
                }
            )
        except pynvml.NVMLError:
            pass

        # temperature
        try:
            info["temperature"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            pass

        # power consumption
        try:
            info["power_draw"] = (
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            )  # Convert to watts
            info["power_limit"] = (
                pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            )
        except pynvml.NVMLError:
            pass

        return info

    def get_cuda_version(self) -> Optional[str]:
        """Get CUDA version."""
        if not self._cuda_available:
            return None
        return torch.version.cuda

    def get_cudnn_version(self) -> Optional[str]:
        """Get cuDNN version."""
        if not self._cuda_available:
            return None
        return str(torch.backends.cudnn.version())

    def get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }

        if self._cuda_available:
            info["cuda_version"] = self.get_cuda_version()
            info["cudnn_version"] = self.get_cudnn_version()

        return info

    def get_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split("\n")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def print_device_info(self, device_id: int = 0) -> None:
        """Print formatted device information.

        Args:
            device_id: GPU device ID
        """
        if not self._cuda_available:
            print("CUDA is not available")
            return

        info = self.get_device_info(device_id)
        system_info = self.get_system_info()
        driver_version = self.get_driver_version()

        print("=" * 60)
        print(f"GPU Device Information (Device {device_id})")
        print("=" * 60)

        print(f"Device Name: {info['name']}")
        print(f"Compute Capability: {info['compute_capability']}")
        print(f"Multi-processors: {info['multi_processor_count']}")
        print(f"Max threads per MP: {info['max_threads_per_multiprocessor']}")
        print(f"Warp size: {info['warp_size']}")

        print("\nMemory Information:")
        if "memory_total" in info:
            print(f"Total memory: {info['memory_total'] / 1024**3:.2f} GB")
            print(f"Free memory: {info['memory_free'] / 1024**3:.2f} GB")
            print(f"Used memory: {info['memory_used'] / 1024**3:.2f} GB")
        else:
            print(f"Total memory: {info['total_memory'] / 1024**3:.2f} GB")

        if "memory_clock" in info:
            print(f"\nClock Speeds:")
            print(f"Memory clock: {info['memory_clock']} MHz")
            print(f"Graphics clock: {info['graphics_clock']} MHz")

        if "temperature" in info:
            print(f"\nTemperature: {info['temperature']}°C")

        if "power_draw" in info:
            print(f"Power draw: {info['power_draw']:.1f}W / {info['power_limit']:.1f}W")

        print("\nSoftware Information:")
        print(f"Platform: {system_info['platform']}")
        print(f"Python: {system_info['python_version']}")
        print(f"PyTorch: {system_info['pytorch_version']}")
        if driver_version:
            print(f"NVIDIA Driver: {driver_version}")
        if system_info.get("cuda_version"):
            print(f"CUDA: {system_info['cuda_version']}")
        if system_info.get("cudnn_version"):
            print(f"cuDNN: {system_info['cudnn_version']}")

        print("=" * 60)

    def list_all_devices(self) -> None:
        """List information for all available devices."""
        if not self._cuda_available:
            print("CUDA is not available")
            return

        for i in range(self.device_count):
            self.print_device_info(i)
            if i < self.device_count - 1:
                print("\n")
