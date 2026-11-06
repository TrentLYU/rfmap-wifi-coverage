

volatile int _E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91;
volatile int _E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91;
volatile int _E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91;
volatile int _E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91;
volatile int _E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM;
volatile int _E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM;
volatile int _E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM;
volatile int _E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM;
volatile unsigned long _E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92;
volatile int _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81;
volatile int _E6_8C_89_E9_94_AE_E5_B0_B1_E7_BB_AA;
volatile unsigned long _E5_BC_80_E5_A7_8B_E6_97_B6_E5_88_BB;
volatile unsigned long _E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0;
volatile int _E4_B8_B2_E5_8F_A3_E5_91_BD_E4_BB_A4;

void setup(){
  _E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91 = 0;
  _E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91 = 0;
  _E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91 = 1;
  _E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91 = 1;
  _E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM = 120;
  _E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM = 105;
  _E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM = 120;
  _E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM = 135;
  _E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92 = 5200;
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 0;
  _E6_8C_89_E9_94_AE_E5_B0_B1_E7_BB_AA = 0;
  _E5_BC_80_E5_A7_8B_E6_97_B6_E5_88_BB = 0;
  _E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0 = 0;
  _E4_B8_B2_E5_8F_A3_E5_91_BD_E4_BB_A4 = 0;
  Serial.begin(9600);
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(A3, INPUT);
analogWrite(10, 0);
analogWrite(9, 0);
analogWrite(6, 0);
analogWrite(11, 0);
pinMode(13, OUTPUT);
pinMode(8, OUTPUT);
pinMode(12, OUTPUT);
pinMode(7, OUTPUT);
Serial.println("SHIFT_LEFT_5200_READY: KEY1 = shift left 5200ms; 0 = stop");
  pinMode(A3, INPUT);
  pinMode(13, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(7, OUTPUT);
}

void loop(){
  if ((analogRead(A3) > 100)) {
  _E6_8C_89_E9_94_AE_E5_B0_B1_E7_BB_AA = 1;

  } else {
  if (((_E6_8C_89_E9_94_AE_E5_B0_B1_E7_BB_AA == 1) && (analogRead(A3) < 50))) {
  delay(25);
  if ((analogRead(A3) < 50)) {
  _E6_8C_89_E9_94_AE_E5_B0_B1_E7_BB_AA = 0;
  if ((_E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 == 1)) {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 0;
  Serial.println("ABORT: stopped early; do not use this distance");

  } else {
  if ((_E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 == 0)) {
  if (((((((((((_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91 >= 0) && (_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91 <= 1)) && ((_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91 >= 0) && (_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91 <= 1))) && ((_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91 >= 0) && (_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91 <= 1))) && ((_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91 >= 0) && (_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91 <= 1))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92 > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92 <= 60000)))) {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  pinMode(13, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(7, OUTPUT);
  digitalWrite(13,_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91);
  digitalWrite(8,_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91);
  digitalWrite(12,_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91);
  digitalWrite(7,_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91);
  analogWrite(11, _E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM);
  analogWrite(9, _E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM);
  analogWrite(10, _E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM);
  analogWrite(6, _E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM);
  _E5_BC_80_E5_A7_8B_E6_97_B6_E5_88_BB = millis();
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 1;
  _E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0 = (_E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0 + 1);
  Serial.print("RUN trial=");
  Serial.print(_E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0);
  Serial.print(" fixed_ms=");
  Serial.println(_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92);

  } else {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  Serial.println("INVALID_SETTINGS: check direction/PWM/fixed_ms");

  }

  }

  }

  }

  }

  }
  if (Serial.available() > 0) {
  _E4_B8_B2_E5_8F_A3_E5_91_BD_E4_BB_A4 = Serial.read();
  if ((_E4_B8_B2_E5_8F_A3_E5_91_BD_E4_BB_A4 == 48)) {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 0;
  Serial.println("ABORT: stopped early; do not use this distance");

  } else {
  if ((_E4_B8_B2_E5_8F_A3_E5_91_BD_E4_BB_A4 == 49)) {
  if ((_E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 == 0)) {
  if (((((((((((_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91 >= 0) && (_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91 <= 1)) && ((_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91 >= 0) && (_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91 <= 1))) && ((_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91 >= 0) && (_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91 <= 1))) && ((_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91 >= 0) && (_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91 <= 1))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM <= 180))) && ((_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92 > 0) && (_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92 <= 60000)))) {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  pinMode(13, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(7, OUTPUT);
  digitalWrite(13,_E5_8F_B3_E5_89_8D_E6_96_B9_E5_90_91);
  digitalWrite(8,_E5_8F_B3_E5_90_8E_E6_96_B9_E5_90_91);
  digitalWrite(12,_E5_B7_A6_E5_90_8E_E6_96_B9_E5_90_91);
  digitalWrite(7,_E5_B7_A6_E5_89_8D_E6_96_B9_E5_90_91);
  analogWrite(11, _E5_B7_A6_E5_B9_B3_E7_A7_BB4_E5_8F_B7PWM);
  analogWrite(9, _E5_B7_A6_E5_B9_B3_E7_A7_BB2_E5_8F_B7PWM);
  analogWrite(10, _E5_B7_A6_E5_B9_B3_E7_A7_BB1_E5_8F_B7PWM);
  analogWrite(6, _E5_B7_A6_E5_B9_B3_E7_A7_BB3_E5_8F_B7PWM);
  _E5_BC_80_E5_A7_8B_E6_97_B6_E5_88_BB = millis();
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 1;
  _E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0 = (_E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0 + 1);
  Serial.print("RUN trial=");
  Serial.print(_E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0);
  Serial.print(" fixed_ms=");
  Serial.println(_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92);

  } else {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  Serial.println("INVALID_SETTINGS: check direction/PWM/fixed_ms");

  }

  }

  }

  }

  }
  if (((_E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 == 1) && ((millis() - _E5_BC_80_E5_A7_8B_E6_97_B6_E5_88_BB) >= _E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92))) {
  analogWrite(10, 0);
  analogWrite(9, 0);
  analogWrite(6, 0);
  analogWrite(11, 0);
  _E8_BF_90_E8_A1_8C_E7_8A_B6_E6_80_81 = 0;
  Serial.print("DONE trial=");
  Serial.print(_E6_B5_8B_E8_AF_95_E6_AC_A1_E6_95_B0);
  Serial.print(" fixed_ms=");
  Serial.println(_E5_B7_A6_E5_B9_B3_E7_A7_BB_E6_97_B6_E9_95_BF_E6_AF_AB_E7_A7_92);
  Serial.println("Measure distance in cm; KEY1 repeats forward");

  }
  delay(5);

}
