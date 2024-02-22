#include <Servo.h>

Servo motor;

void setup() {
  Serial.begin(9600);
  motor.attach(9); // Attach the motor to pin 9
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'O') { // Open the motor
      motor.write(180);
      delay(1000); // Wait for 1 second
      motor.write(0); // Stop the motor
    }
  }
}
