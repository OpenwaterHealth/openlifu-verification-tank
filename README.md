# Open-LIFU Verification Tank

## Disclaimer

CAUTION - The system described here has not been evaluated by the FDA and is not designed for the treatment or diagnosis of any disease. It is provided AS-IS, with no warranties. User assumes all liability and responsibility for identifying and mitigating risks associated with using this software.

A Python-based system for automated acoustic field verification and characterization of focused ultrasound transducers using the Open-LIFU platform.

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
- **Open-LIFU Transducer System**: Multi-element focused ultrasound transducer with beam steering
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
   - Connect Channel A of the Picoscope to the Hydrophone Output, and channel B to the trigger output of the Open-LIFU system (optional)

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

### Setup Data
The GUI and every script share one setup data folder (default `~/.openlifu_verification/setup_data`). Create it, add any template files missing from an existing one, or switch to another folder with:

```
openlifu-setup-data [FOLDER]        # or: python -m openlifu_verification.setup_data
```

The GUI and `VerificationTank` also copy in any missing template files on first use. The template ships inside the package (`openlifu_verification/setup_data_template/`); existing files are never overwritten:

```
testers/testers.json                       {"<tester ID>": {"Name": "<name>"}}
consoles/consoles.json                     {"WL114Q#####" or "WL134Q#####": {"HWID": "<hardware ID>"}}
hydrophone_calibrations/hydrophones.json   {"HYDRO####": {"Calibration": "<file in this folder>"}}
hydrophone_calibrations/<calibration files>
transducers/openlifu_1x.json, openlifu_2x.json   transducer pinmaps
hydrophone.json                            last hydrophone position + ID (used when --hydrophone is omitted)
last_setup.json                            last tester / console / hydrophone (GUI)
```

Bare hydrophone IDs (e.g. `--hydrophone 2246`) resolve against `hydrophone_calibrations/` in the setup data folder. Supported calibration formats come from Onda Corporation, Precision Acoustics and other manufacturers using similar text-based formats.

`scripts/run_report_gui.py` only accepts testers, consoles and hydrophones registered in the setup data folder (changeable in the launcher or with `--setup-data`; the folder picked in the launcher becomes the active one for the scripts too).

The tester and console lists start empty; run with `--add-items` to add a missing, well-formed ID from the launcher (this flag also shows the "…" buttons for changing the setup data and reports folders). To make this the default, pass a `--scan-config` file with a `gui` section, e.g. `config/gui/add_items.json` (`{"gui": {"add_items": true}}`); `--no-add-items` turns it back off. The `gui` section only holds launcher defaults, so a scan config that sets nothing else does not mark the run NON-STANDARD. `--use-last-setup` pre-fills the tester, console and hydrophone from the previous run. The frequency is set from the TXM S/N (`WLM24Q#####` = 155 kHz, `WLM44Q#####` = 400 kHz). A run halts if the connected console's hardware ID doesn't match the one registered for the console S/N. Hardware IDs are base58 of the full 12-byte UID; a console registered with the older 6-byte form is accepted with a warning to re-register it. `--no-validate` accepts any TXM S/N, tester and console (frequency picked by hand, no console HWID check); the hydrophone must still be registered.

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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
