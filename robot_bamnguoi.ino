/* Motor2 is right motor , motor1 is left motor */

int RL_EN_M2 = 17; // băm xung
int R_PWM_M2 = 25;// điều khiển chiều quay
int L_PWM_M2 = 26;// điều khiển chiều quay
int RL_EN_M1 = 13; // băm xung
int R_PWM_M1 = 14;// điều khiển chiều quay
int L_PWM_M1 = 12;// điều khiển chiều quay
const int base = 70;
const int PWM_CHANNEL = 0;    // ESP32 has 16 channels which can generate 16 independent waveforms
const int PWM_FREQ = 1900;     // Recall that Arduino Uno is ~490 Hz. Official ESP32 example uses 5,000Hz
const int PWM_RESOLUTION = 8; // We'll use same resolution as Uno (8 bits, 0-255) but ESP32 can go up to 16 bits 

const int M1CHANNEL = 0U;
const int M2CHANNEL = 2U;
int leftPWM = 0;
int rightPWM = 0;
void parseAndSetPWM(String cmd) {
  // Ví dụ: "L50 R60"
  cmd.trim();
  int lIndex = cmd.indexOf('L');
  int rIndex = cmd.indexOf('R');

  if (lIndex != -1 && rIndex != -1) {
    int left = cmd.substring(lIndex + 1, rIndex).toInt();
    int right = cmd.substring(rIndex + 1).toInt();

    leftPWM = constrain(left, 0, 100);
    rightPWM = constrain(right, 0, 100);
  Serial.print("leftpwm: ");Serial.println(leftPWM);
  Serial.print("rightPWM: ");Serial.println(rightPWM);
    // Scale to 0-255
      digitalWrite(R_PWM_M2, 0);
  digitalWrite(L_PWM_M2, 1); 
  digitalWrite(R_PWM_M1, 0);
  digitalWrite(L_PWM_M1, 1); 
    ledcWrite(M1CHANNEL, map(leftPWM, 0, 100, 0, 105));
    ledcWrite(M2CHANNEL, map(rightPWM, 0, 100, 0, 100));
  }
}

void MoveBackward()
{
  digitalWrite(R_PWM_M2, 1);
  digitalWrite(L_PWM_M2, 0); 
  digitalWrite(R_PWM_M1, 1);
  digitalWrite(L_PWM_M1, 0); 
  ledcWrite(M1CHANNEL,base);
  ledcWrite(M2CHANNEL,base);
}
void TurnRight(int sp)
{
  digitalWrite(R_PWM_M2, 0);
  digitalWrite(L_PWM_M2, 1); 
  digitalWrite(R_PWM_M1, 0);
  digitalWrite(L_PWM_M1, 1); 
  ledcWrite(M1CHANNEL,base+sp);
  ledcWrite(M2CHANNEL,base-sp);  
}
void MoveForward()
{
  digitalWrite(R_PWM_M2, 0);
  digitalWrite(L_PWM_M2, 1); 
  digitalWrite(R_PWM_M1, 0);
  digitalWrite(L_PWM_M1, 1); 
  ledcWrite(M1CHANNEL,base-0);
  ledcWrite(M2CHANNEL,base-0);  
}
void TurnLeft(int sp)
{
  digitalWrite(R_PWM_M2, 0);
  digitalWrite(L_PWM_M2, 1); 
  digitalWrite(R_PWM_M1, 0);
  digitalWrite(L_PWM_M1, 1); 
  ledcWrite(M1CHANNEL,base-sp);
  ledcWrite(M2CHANNEL,base+sp); 
}
void Stop()
{
  ledcWrite(M2CHANNEL,0);
  ledcWrite(M1CHANNEL,0);
}
void setup() 
{ 
  Serial.begin(115200);
  pinMode(RL_EN_M2, OUTPUT);
  pinMode(R_PWM_M2, OUTPUT);
  pinMode(L_PWM_M2, OUTPUT);
  pinMode(RL_EN_M1, OUTPUT);
  pinMode(R_PWM_M1, OUTPUT);
  pinMode(L_PWM_M1, OUTPUT);

  ledcSetup(M1CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(M2CHANNEL, PWM_FREQ, PWM_RESOLUTION);

  ledcAttachPin(RL_EN_M2, M2CHANNEL);
  ledcAttachPin(RL_EN_M1, M1CHANNEL);
//  MoveBackward();
}
void loop() {
  // if (Serial.available()) {
  //   String cmd = Serial.readStringUntil('\n');
  //   parseAndSetPWM(cmd);
  // }

  if(Serial.available()>0)
  {
    String msg = Serial.readStringUntil('\n');
    Serial.print(msg);
    if(msg == "r")
    { 
      if(Serial.available() > 0)
      {
        String msg = Serial.readStringUntil('\n');
        Serial.print(msg);

      }
      TurnRight(msg.toInt());
      
    }
    if(msg == "l")
    {
      if(Serial.available() > 0)
      {
        String msg = Serial.readStringUntil('\n');
        Serial.print(msg);
      }
      TurnLeft(msg.toInt());
      Serial.print(msg);
    }
    if(msg == "s")
    {
      Stop();
      Serial.print(msg);
    }
    if(msg == "f")
    {
      MoveForward();
      Serial.print(msg);
    }
    if(msg == "b")
    {
      MoveBackward();
      Serial.print(msg);
    }
 
  }
  
}
