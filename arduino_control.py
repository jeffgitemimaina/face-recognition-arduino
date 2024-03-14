import pyfirmata
import time

# Replace 'COM3' with the port your Arduino is connected to
board = pyfirmata.Arduino('COM9')

# Define the pin the servo is connected to
servo_pin = 9 # Change this to the pin your servo is connected to

# Initialize the servo
servo = board.get_pin('d:%s:s' % servo_pin)

# Function to move the servo to a specific angle
def move_servo(angle):
    servo.write(angle)
    print(f"Moved servo to {angle} degrees")

# Function to move the servo to 90 degrees and then back to its original position
def move_servo_to_90_and_back():
    original_position = servo.read() # Save the current position
    move_servo(90) # Move to 90 degrees
    time.sleep(0.5) # Wait for 0.5 seconds
    move_servo(original_position) # Move back to the original position
    print("Moved servo to 90 degrees and back")
