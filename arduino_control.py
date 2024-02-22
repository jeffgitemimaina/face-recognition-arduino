import serial
import time

class ArduinoController:
    def __init__(self, port='COM4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
        time.sleep(2)  # Allow time for Arduino to initialize

    def open_door(self):
        # Send command to Arduino to open the door
        self.serial.write(b'1')  # Assuming '1' is the command to open the door
