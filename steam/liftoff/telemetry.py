import socket
import struct
import numpy as np


class LiftoffTelemetry:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 9001))
        self.sock.settimeout(0.5)
        # Matches TelemetryConfiguration.json
        self.fmt = "<f fff fff ffff fff ffff Bffff"

    def capture_telemetry(self):
        tel = np.zeros(21, dtype=np.float32)
        try:
            data, _ = self.sock.recvfrom(1024)
            unpacked = struct.unpack(self.fmt, data[:struct.calcsize(self.fmt)])
            # Flattened vector: Pos(3), Vel(3), Att(4), Gyro(3), Inputs(4), Motors(1 byte ignored + 4)
            # We skip index 18 (motor count)
            tel = np.array(unpacked[1:18] + unpacked[19:23], dtype=np.float32)
        except Exception as e:
            print("[ERROR] Could not read telemetry data")
            print(e)
        finally:
            return tel

