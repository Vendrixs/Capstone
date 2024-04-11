#include <Keypad.h>

const int ledPin = 10;       // Pin connected to an LED or any other output device
const int buttonPin = 11;    // Pin connected to the button
const int buzzerPin = 12;    // Pin connected to the buzzer
const int greenlight = A1; // Pin for green led light 
const int redlight = A2; // Pin for red led light

const byte ROWS = 4; 
const byte COLS = 4;
bool commandReceived = false;

char keys[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};

byte rowPins[ROWS] = {9,8,7,6};
byte colPins[COLS] = {5,4,3,2};

Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );

String enteredPassword = "";

const String correctPassword = "AB1";

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT); // Set the buzzer pin as an output
  pinMode(greenlight, OUTPUT);
  pinMode(redlight, OUTPUT);
  Serial.begin(115200);      // Set the baud rate to match the Python program
}


void loop() {
  char key = keypad.getKey();
  if (key){
    if (key == '*'){
      if (enteredPassword == correctPassword) {
        //what happen when correct password?
        successSound(); // Play success sound
        digitalWrite(ledPin, HIGH);
        green_light();
        delay(2000);
        digitalWrite(ledPin, LOW);
        delay(1000);
        logAttempt("Entry");
      } else {
        //what happen when wrong password?
        errorSound();
        red_light2();
        digitalWrite(ledPin, LOW);
      }
      enteredPassword = "";
    } else {
      enteredPassword += key;
    }
  }
  loop_button();


  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '1') {
      commandReceived = true;
      successSound(); // Play success sound
      digitalWrite(ledPin, HIGH);
      green_light();
      delay(2000);
      digitalWrite(ledPin, LOW);
      delay(500);
    } 
    if (command == '0' && commandReceived) {
      red_light3();
      commandReceived = false;
    }
    if (command == '3') {
      commandReceived = true;
      //what happen when wrong password?
        errorSound();
        red_light2();
        digitalWrite(ledPin, LOW);
    }
  }
}

void logAttempt(String entryType) {
  // Send a message over serial indicating the type of entry (Entry/Exit)
  Serial.print("Successful ");
  Serial.println(entryType);
}

void loop_button(){  // Check f the button is pressed
  if (digitalRead(buttonPin) == LOW) {
    digitalWrite(ledPin, HIGH);
    emergencySound();
    green_light();
    delay(5000);
    digitalWrite(ledPin, LOW);
    delay(1000);
    logAttempt("Exit");
  }
}


void successSound() {
  // Play a success sound on the buzzer (you can customize this)
  tone(buzzerPin, 2000, 300); // Example: 1000 Hz tone for 100 ms
  delay(300);
  noTone(buzzerPin);
}

void errorSound() {
  // Play an error sound on the buzzer (you can customize this)  
  tone(buzzerPin, 500, 200); // Example: 500 Hz tone for 200 ms
  delay(300);
  noTone(buzzerPin);
  tone(buzzerPin, 500, 200); // Example: 500 Hz tone for 200 ms
  delay(300);
  noTone(buzzerPin);
}

void errorSound2() {
  // Play an error sound on the buzzer (you can customize this)  
  tone(buzzerPin, 500, 200); // Example: 500 Hz tone for 200 ms
  delay(300);
  noTone(buzzerPin);
  tone(buzzerPin, 500, 200); // Example: 500 Hz tone for 200 ms
  delay(300);
  noTone(buzzerPin);
  delay(5000);
}

void emergencySound() {
  // Play the emergency sound on the buzzer (customize this to your preference)
  tone(buzzerPin, 2000, 300); // Example: 1500 Hz tone for 300 ms
  delay(300);
  noTone(buzzerPin);
}

void green_light(){
  analogWrite(greenlight, 255);
  delay(2000);
  analogWrite(greenlight, 000);
  delay(500);
}

void red_light2(){
  analogWrite(redlight, 255);
  delay(2000);
  analogWrite(redlight, 000);
}
void red_light3(){
  analogWrite(redlight, 255);
  delay(2000);
  analogWrite(redlight, 000);
  delay(1000);
}


