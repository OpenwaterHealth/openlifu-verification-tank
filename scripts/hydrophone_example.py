"""
Simple example showing how to use the Hydrophone class.

This script demonstrates basic usage of the Hydrophone class for:
1. Loading calibration data
2. Querying sensitivity at specific frequencies
3. Performing deconvolution of voltage signals
"""

import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path so we can import openlifu_verification
sys.path.append(str(Path(__file__).parent.parent))

from openlifu_verification import Hydrophone


def main():
    """Main example function."""
    print("=== Hydrophone Calibration Example ===\n")
    
    # 1. Load the hydrophone calibration. Passing a bare ID lets
    # Hydrophone locate the file under ``config/hydrophone_calibrations/``
    # in the current working directory (with a fallback to the copy
    # shipped in the repo).
    hydrophone = Hydrophone("2246")
    
    print(f"Loaded hydrophone: {hydrophone}")
    print()
    
    # 2. Display metadata
    metadata = hydrophone.get_metadata_summary()
    print("Hydrophone information:")
    for key, value in metadata.items():
        if key != 'frequency_range_mhz':
            print(f"  {key}: {value}")
    
    freq_range = metadata['frequency_range_mhz']
    print(f"  Calibrated frequency range: {freq_range['min']:.3f} - {freq_range['max']:.1f} MHz")
    print(f"  Number of calibration points: {freq_range['n_points']}")
    print()
    
    # 3. Query sensitivity at different frequencies
    print("Sensitivity values:")
    test_frequencies = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]  # MHz
    for freq_mhz in test_frequencies:
        freq_hz = freq_mhz * 1e6
        try:
            sensitivity = hydrophone.get_sensitivity_pa_per_v(freq_hz)
            print(f"  {freq_mhz:4.1f} MHz: {sensitivity:.2e} Pa/V")
        except Exception as e:
            print(f"  {freq_mhz:4.1f} MHz: Error - {e}")
    print()
    
    # 4. Example deconvolution
    print("Deconvolution example:")
    
    # Create a simple test signal (2 MHz sine wave)
    fs = 50e6  # 50 MHz sampling rate
    duration = 2e-6  # 2 microseconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Simulated pressure signal (1000 Pa amplitude at 2 MHz)
    freq_test = 2e6  # 2 MHz
    pressure_original = 1000 * np.sin(2 * np.pi * freq_test * t)
    
    # Convert to voltage using hydrophone sensitivity
    sensitivity = hydrophone.get_sensitivity_pa_per_v(freq_test)
    voltage_signal = pressure_original / sensitivity
    
    # Deconvolve to recover pressure
    sampling_interval = 1.0 / fs
    pressure_recovered = hydrophone.deconvolve_voltage_signal(voltage_signal, sampling_interval)
    
    # Calculate performance metrics
    rms_error = np.sqrt(np.mean((pressure_original - pressure_recovered)**2))
    rms_signal = np.sqrt(np.mean(pressure_original**2))
    relative_error = rms_error / rms_signal * 100
    
    print(f"  Test frequency: {freq_test/1e6:.1f} MHz")
    print(f"  Original pressure amplitude: {np.max(np.abs(pressure_original)):.1f} Pa")
    print(f"  Sensitivity at test frequency: {sensitivity:.2e} Pa/V")
    print(f"  Voltage amplitude: {np.max(np.abs(voltage_signal)):.2e} V")
    print(f"  Recovered pressure amplitude: {np.max(np.abs(pressure_recovered)):.1f} Pa")
    print(f"  RMS error: {rms_error:.1f} Pa ({relative_error:.2f}%)")
    print()
    
    # 5. Show calibration data sample
    print("Sample calibration data (first 5 rows):")
    print(hydrophone.calibration_data.head())
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()