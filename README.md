# Open-LIFU Verification Tank

A Python-based system for automated acoustic field verification and characterization of focused ultrasound transducers using the OpenLIFU platform.

## Overview

The Open-LIFU Verification Tank provides a comprehensive solution for measuring and characterizing focused ultrasound acoustic fields. It integrates multiple instruments and provides high-level automation for common verification tasks, including beam profiling, frequency response measurement, and focus optimization.

### Key Features

- **Multi-instrument coordination**: Seamless integration of transducer control, data acquisition, and power management
- **Automated measurements**: Built-in scanning, peak finding, and field mapping capabilities  
- **Calibrated pressure measurements**: Support for calibrated hydrophones with frequency-dependent sensitivity
- **High-speed data acquisition**: Integration with PicoScope oscilloscopes for precision timing
- **Flexible control**: Programmable power supply control with voltage settling detection

## System Components

### Hardware
- **OpenLIFU Transducer System**: Multi-element focused ultrasound transducer with beam steering
- **PicoScope 5000A Series**: High-speed USB oscilloscope for data acquisition
- **AIM TTi QPX600DP**: Dual-channel programmable power supply for drive voltage control
- **Calibrated Hydrophone**: Precision pressure sensor with tank positioning system

### Software
- **Python API**: Object-oriented interfaces for each system component
- **Automation Scripts**: Ready-to-use measurement scripts for common tasks
- **Documentation**: Comprehensive API documentation and examples

## Installation

### Prerequisites

1. **Hardware Setup**
   - Connect PicoScope via USB and install drivers from Pico Technology
   - Connect QPX600DP power supply via USB and install AIM TTi drivers
   - Set up Open-LIFU system according to the manufacturer's instructions
   - Install a hydrophone in the tank positioning system
   - Connect Channel A of the Picoscope to the Hydrophone Output, and channel B to the trigger output of the OpenLIFU system (optional)

2. **Python Environment**
   ```bash
   # Python 3.8+ recommended
   pip install numpy scipy pandas matplotlib
   pip install pyserial
   pip install picosdk
   pip install openlifu
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/OpenwaterHealth/OpenLIFU-verification-tank.git
   cd OpenLIFU-verification-tank
   pip install -e .
   ```

## Quick Start

### Basic Pressure Measurement
```python
from openlifu_verification import VerificationTank, Hydrophone
from pathlib import Path

# Load hydrophone calibration
hydrophone = Hydrophone(Path("hydrophone_calibrations/your_calibration.txt"))

# Measure pressure at focus
with VerificationTank(frequency=400) as tank:
    # Configure system
    tank.configure_lifu(
        frequency_kHz=400,
        voltage=50,
        duration_msec=1.0,
        interval_msec=100
    )
    
    # Set focus position  
    tank.set_focus(x=0, y=0, z=30)  # mm coordinates
    
    # Capture data
    data = tank.run_capture()
    
    # Convert to calibrated pressure
    sensitivity = hydrophone.get_sensitivity_pa_per_v(400e3)
    pressure_pa = data['A'] * sensitivity
    
    peak_pressure = np.max(pressure_pa)
    print(f"Peak pressure: {peak_pressure:.0f} Pa")
```

### Automated Field Mapping
```python
# 2D beam profile measurement
with VerificationTank() as tank:
    tank.configure_lifu(frequency_kHz=1000, voltage=75, duration_msec=0.5, interval_msec=50)
    
    # Define scan grid
    x_positions = np.linspace(-10, 10, 21)  # ±10 mm
    y_positions = np.linspace(-10, 10, 21)
    z_focus = 25
    
    # Automated scanning
    pressure_map = np.zeros((len(x_positions), len(y_positions)))
    
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            pressure_map[i, j] = tank.get_peak_voltage(x, y, z_focus)
    
    # Analyze results
    peak_location = np.unravel_index(np.argmax(pressure_map), pressure_map.shape)
    print(f"Peak at ({x_positions[peak_location[0]]:.1f}, {y_positions[peak_location[1]]:.1f}) mm")
```

## Example Scripts

The `scripts/` directory contains ready-to-use measurement scripts:

- **`single_pulse.py`**: Single-point pressure measurement
- **`scan_2d.py`**: 2D beam profile mapping  
- **`scan_frequency.py`**: Frequency response characterization
- **`scan_voltage.py`**: Drive voltage linearity measurement
- **`find_peak.py`**: Automated focus optimization
- **`plot_*.py`**: Visualization utilities for measurement data

## API Documentation

Comprehensive API documentation is available in the `docs/` directory:

- **[API Overview](docs/api.md)**: Complete system documentation and usage guide
- **[Hydrophone Class](docs/hydrophone_class.md)**: Calibrated pressure measurement interface
- **[Picoscope Class](docs/picoscope_class.md)**: High-speed data acquisition interface  
- **[QPX600DP Class](docs/qpx600dp_class.md)**: Programmable power supply interface
- **[VerificationTank Class](docs/verificationtank_class.md)**: High-level system coordination

## Configuration

### Transducer Setup
Transducer configurations are stored as JSON files in the `transducers/` directory:
```json
{
  "name": "OpenLIFU 1x180 EVT1",
  "frequency_hz": 1000000,
  "element_count": 180,
  "geometry": "...",
  "focus_distance_mm": 30.0
}
```

### Hydrophone Calibration
Place hydrophone calibration files in `hydrophone_calibrations/` directory. Supports standard formats from:
- Onda Corporation
- Precision Acoustics  
- Other manufacturers using similar text-based formats

## Safety and Best Practices

### Safety Guidelines
- Always use appropriate drive voltage limits for your transducer
- Implement proper interlock systems for high-power operation
- Monitor for cavitation and heating effects during extended operation
- Ensure proper grounding and electrical safety measures

### Measurement Best Practices
- Allow adequate settling time between measurements
- Use appropriate sampling rates for your frequency range
- Calibrate the positioning system regularly
- Account for temperature effects on sound speed and sensitivity
- Implement error handling for robust automated measurements

### System Maintenance
- Regularly verify instrument calibration
- Check mechanical positioning accuracy
- Monitor cable connections and signal integrity
- Keep software and drivers updated

## Contributing

We welcome contributions to improve the OpenLIFU Verification Tank system:

1. **Fork the repository** and create a feature branch
2. **Add tests** for new functionality in the `tests/` directory
3. **Update documentation** for API changes
4. **Submit a pull request** with a clear description of changes

### Development Setup
```bash
git clone https://github.com/OpenwaterHealth/OpenLIFU-verification-tank.git
cd OpenLIFU-verification-tank
pip install -e .[dev]  # Install with development dependencies
pytest tests/          # Run test suite
```

## Troubleshooting

### Common Issues

**Instrument Connection Problems**
- Verify USB connections and driver installations
- Check that no other software is using the instruments
- Ensure proper user permissions for device access

**Measurement Accuracy Issues**  
- Verify hydrophone calibration and positioning
- Check for proper acoustic coupling
- Ensure stable temperature conditions
- Validate trigger settings and signal levels

**Performance Problems**
- Optimize sampling parameters for your application
- Use appropriate resolution modes based on signal amplitude
- Consider measurement time vs. spatial resolution trade-offs
- Implement efficient scanning patterns

### Getting Help
- Check the [API documentation](docs/api.md) for detailed usage information
- Review example scripts in the `scripts/` directory
- Run the test suite to verify system functionality
- Submit issues on GitHub with detailed error descriptions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenLIFU development team for the core transducer control platform
- Pico Technology for PicoScope SDK and support
- AIM TTi for power supply control protocols
- Contributors to the open-source Python scientific computing ecosystem

## Citation

If you use this software in your research, please cite:

```bibtex
@software{openlifu_verification_tank,
  title = {OpenLIFU Verification Tank},
  author = {Openwater Health},
  url = {https://github.com/OpenwaterHealth/OpenLIFU-verification-tank},
  year = {2024}
}
```
