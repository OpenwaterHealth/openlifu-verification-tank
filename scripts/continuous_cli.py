import logging
import time
from pathlib import Path
import numpy as np
from openlifu_verification import VerificationTank

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

VALID_MODES = ("single", "continuous", "sequence")

def main():
    # All configure_lifu parameters live here so `param=<value>` commands
    # can update them uniformly. num_modules is a VerificationTank ctor
    # arg (not a configure_lifu arg) so it's kept separate. Initial
    # values come from the shared VerificationTank defaults so that a
    # single source of truth drives all scripts; continuous_cli tweaks
    # only where its use case diverges (lower voltage, longer interval,
    # continuous trigger, 3 pulses / burst).
    num_modules = 1
    settings = {
        "frequency_kHz": VerificationTank.DEFAULT_FREQUENCY_KHZ,
        "voltage": 10.0,
        "duration_usec": (10 / VerificationTank.DEFAULT_FREQUENCY_KHZ) * 1000.0,
        "interval_msec": 50,
        "pulse_count": 3,
        "pulse_train_interval_msec": 0,
        "pulse_train_count": 1,
        "trigger_mode": "continuous",
    }
    focus = {"x": 0.0, "y": 0.0, "z": 50.0}

    def coerce(key, value):
        """Coerce a raw string value to the type of settings[key].

        Special-cases trigger_mode (validated against VALID_MODES).
        Raises ValueError on bad input.
        """
        if key == "trigger_mode":
            v = value.strip().lower()
            if v not in VALID_MODES:
                raise ValueError(
                    f"trigger_mode must be one of: {', '.join(VALID_MODES)}"
                )
            return v
        current = settings[key]
        # bool must be checked before int since bool is a subclass of int
        if isinstance(current, bool):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return type(current)(value)

    logger.info("Starting Continuous CLI Script...")
    try:
        with VerificationTank(frequency=settings["frequency_kHz"],
                              ext_power_supply=False,
                              use_picoscope=False,
                              num_modules=num_modules) as ver:

            def reconfigure():
                """Re-apply configure_lifu + set_focus from `settings`/`focus`."""
                ver.configure_lifu(**settings)
                ver.set_focus(focus["x"], focus["y"], focus["z"])

            reconfigure()

            print(
                "Commands:\n"
                "  start                             - start_sonication (in current mode)\n"
                "  stop                              - stop_sonication + HV off\n"
                "  single                            - switch to single mode, fire one pulse train, stop\n"
                "  <param>=<value>                   - update a configure_lifu setting, e.g. voltage=12,\n"
                "                                      trigger_mode=single, duration_usec=50,\n"
                "                                      frequency_kHz=400, pulse_count=1, ...\n"
                "  x=<mm> / y=<mm> / z=<mm>          - update one focus coordinate\n"
                "  x,y,z                             - set focus in mm (three comma-separated values)\n"
                "  von / voff                        - HV output on/off\n"
                "  show                              - print current settings + focus\n"
                "  exit                              - quit"
            )

            while True:
                command = input(f"[{settings['trigger_mode']}]:").strip()
                if not command:
                    continue
                if command == "exit":
                    break
                elif command == "start":
                    logger.info("Starting Trigger...")
                    #ver.enable_hv_output(wait=True)
                    ver.lifu.start_sonication(turn_hv_on=False, wait_for_settle=False, async_mode=False)
                elif command == "stop":
                    logger.info("Stopping Trigger...")
                    try:
                        ver.lifu.stop_sonication(turn_hv_off=False, wait_for_settle=False)
                    except Exception as e:
                        logger.warning("stop_sonication raised: %s", e)
                    ver.disable_hv_output()
                elif command == "single":
                    if settings["trigger_mode"] != "single":
                        logger.info("Switching to single trigger mode...")
                        settings["trigger_mode"] = "single"
                        reconfigure()
                    logger.info("Firing single pulse train...")
                    ver.enable_hv_output(wait=True)
                    ver.lifu.start_sonication(turn_hv_on=False, wait_for_settle=False, async_mode=False)
                    # In single mode the device fires one train and stops on
                    # its own. Give it time to complete before we call
                    # stop_sonication so we don't cut it short.
                    time.sleep(1.0)
                    try:
                        ver.lifu.stop_sonication(turn_hv_off=False, wait_for_settle=False)
                    except Exception as e:
                        logger.warning("stop_sonication after single raised: %s", e)
                elif command == "von":
                    ver.enable_hv_output(wait=True)
                elif command == "voff":
                    ver.disable_hv_output()
                elif command == "show":
                    print(f"  focus:    {focus}")
                    for k, v in settings.items():
                        print(f"  {k}: {v}")
                elif "=" in command:
                    key, _, raw_value = command.partition("=")
                    key = key.strip()
                    if key in settings:
                        try:
                            new_value = coerce(key, raw_value)
                        except ValueError as e:
                            print(f"Invalid value for {key}: {e}")
                            continue
                        settings[key] = new_value
                        logger.info("Reconfiguring %s=%s", key, new_value)
                        reconfigure()
                    elif key in focus:
                        try:
                            focus[key] = float(raw_value)
                        except ValueError:
                            print(f"Invalid value for {key}: {raw_value}")
                            continue
                        ver.set_focus(focus["x"], focus["y"], focus["z"])
                    else:
                        valid = ", ".join(list(settings.keys()) + list(focus.keys()))
                        print(f"Unknown parameter '{key}'. Valid: {valid}")
                else:
                    # Fallback: `x,y,z` triple to set focus in one shot.
                    try:
                        coords = [float(x) for x in command.split(",")]
                    except ValueError:
                        print("Invalid command")
                        continue
                    if len(coords) != 3:
                        print("Invalid coordinates. Please provide x, y, and z.")
                        continue
                    focus["x"], focus["y"], focus["z"] = coords
                    ver.set_focus(focus["x"], focus["y"], focus["z"])

            # Ensure trigger is stopped and HV is off before exiting
            try:
                ver.lifu.stop_sonication(turn_hv_off=False, wait_for_settle=False)
            except Exception as e:
                logger.warning("stop_sonication on exit raised: %s", e)
            ver.disable_hv_output()

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return # Exit gracefully

    logger.info("Finished Continuous CLI.")

if __name__ == "__main__":
    main()
