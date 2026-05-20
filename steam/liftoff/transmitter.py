import time

from evdev import UInput, ecodes, AbsInfo


# Standard RC channel mapping (AETR order):
#   Channel 1 (ABS_X):  Roll     (Aileron)
#   Channel 2 (ABS_Y):  Pitch    (Elevator)
#   Channel 3 (ABS_Z):  Throttle
#   Channel 4 (ABS_RX): Yaw      (Rudder)
#   Channels 5-8:       Aux switches
RC_CHANNEL_AXES = [
    ecodes.ABS_X,   # CH1: Roll
    ecodes.ABS_Y,   # CH2: Pitch
    ecodes.ABS_Z,   # CH3: Throttle
    ecodes.ABS_RX,  # CH4: Yaw
    ecodes.ABS_RY,  # CH5: Aux 1
    ecodes.ABS_RZ,  # CH6: Aux 2
]


def _create_transmitter(with_buttons=False):
    """Create a virtual RC transmitter via evdev UInput.

    Mimics a USB-connected FPV radio (e.g. FrSky, RadioMaster)
    presenting as a 6-channel joystick device.

    If `with_buttons` is True, declares 8 BTN_JOYSTICK keys so SDL2/Unity
    classify the device as a joystick rather than a generic input device.
    """
    # All channels use the same range as real RC transmitters
    channel_info = AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)

    cap = {
        ecodes.EV_ABS: [(axis, channel_info) for axis in RC_CHANNEL_AXES],
    }
    if with_buttons:
        cap[ecodes.EV_KEY] = [ecodes.BTN_JOYSTICK + i for i in range(8)]

    device = UInput(
        cap,
        name="FPV Transmitter",
        vendor=0x1209,   # pid.codes (open-source USB VID)
        product=0x4F50,  # custom PID
        version=0x0001,
    )
    time.sleep(1)  # Wait for device registration
    return device


class EvdevTransmitter:
    """Virtual RC transmitter that sends stick inputs via evdev UInput."""

    def __init__(self, with_buttons=False):
        self.device = _create_transmitter(with_buttons=with_buttons)
        self._pending = False

    def set_channel(self, channel_idx, value):
        """Set a single RC channel value (-32768 to 32767)."""
        self.device.write(ecodes.EV_ABS, RC_CHANNEL_AXES[channel_idx], int(value))
        self._pending = True

    def set_sticks(self, roll=0, pitch=0, throttle=0, yaw=0):
        """Set all 4 primary stick axes at once."""
        self.device.write(ecodes.EV_ABS, ecodes.ABS_X, int(roll))
        self.device.write(ecodes.EV_ABS, ecodes.ABS_Y, int(pitch))
        self.device.write(ecodes.EV_ABS, ecodes.ABS_Z, int(throttle))
        self.device.write(ecodes.EV_ABS, ecodes.ABS_RX, int(yaw))
        self._pending = True

    def update(self):
        """Flush pending events with a SYN_REPORT."""
        if self._pending:
            self.device.syn()
            self._pending = False

    def center_all(self):
        """Center all channels (neutral position)."""
        for axis in RC_CHANNEL_AXES:
            self.device.write(ecodes.EV_ABS, axis, 0)
        self.device.syn()
        self._pending = False

    def close(self):
        self.center_all()
        self.device.close()

