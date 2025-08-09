#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_BME280.h>
#include "time.h"

Adafruit_BME280 bme;

// WiFi credentials
const char* ssid = "shaw";
const char* password = "1234567890";

// MQTT settings
const char* mqtt_server = "10.101.105.34"; 
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_Motor_Monitor";

// MQTT topics
const char* topic_sensors = "motor/sensors";
const char* topic_status = "motor/status";
const char* topic_control = "motor/control";
const char* topic_motor_feedback = "motor/feedback";

// NTP settings for timezone consistency
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0;           // UTC offset (0 for UTC)
const int daylightOffset_sec = 0;       // No daylight saving for consistency

// *** BTS7960 Motor Control Pins ***
#define MOTOR_RPWM  25    // RPWM pin (Forward)
#define MOTOR_R_EN  27    // R_EN pin  
#define MOTOR_LPWM  26    // LPWM pin (Reverse)
#define MOTOR_L_EN  14    // L_EN pin

// *** ACS712 Current Sensor Pin ***
#define CURRENT_SENSOR_PIN  34    // GPIO34 (ADC1_CH6) for ACS712 OUT

// ENHANCED: ACS712 Configuration (ACS712-30A) with better calibration
const float ACS712_SENSITIVITY = 0.066;  // 66mV/A for ACS712-30A
const float ACS712_ZERO_CURRENT_VOLTAGE = 1.65;  // 1.65V at 0A (for 3.3V supply)
const int ADC_RESOLUTION = 4095;  // 12-bit ADC
const float ADC_VOLTAGE_REF = 3.3;  // ESP32 ADC reference voltage
const float MOTOR_VOLTAGE = 12.0;   // FIXED: Motor supply voltage for P = V × I

// Current sensor calibration and filtering
float currentOffset = 0.0;  // Calibration offset
const int CURRENT_SAMPLES = 20;  // Increased samples for better filtering
float currentReadings[CURRENT_SAMPLES];
int currentSampleIndex = 0;
bool currentSamplesReady = false;

// Motor control variables
int currentMotorSpeed = 100;        // Current speed percentage (0-100)
String currentDirection = "forward"; // "forward", "reverse", "stopped"
String motorStatus = "normal";      // "normal", "transitioning", "emergency_stop"
bool motorEnabled = true;
unsigned long lastMotorUpdate = 0;
unsigned long transitionStartTime = 0;
int targetSpeed = 100;
String targetDirection = "forward";
bool isTransitioning = false;

// ENHANCED: Current monitoring variables with better tracking
float currentCurrent = 0.0;         // Current motor current in Amperes
float maxCurrent = 0.0;             // Peak current since startup
float totalEnergy = 0.0;            // Total energy consumed (Ah)
unsigned long lastEnergyUpdate = 0;
String currentStatus = "IDLE";      // "IDLE", "NORMAL", "HIGH", "OVERLOAD", "FAULT"
float motorPower = 0.0;             // Calculated motor power (P = V × I)

WiFiClient espClient;
PubSubClient client(espClient);

struct SensorReading {
  String timestamp;
  float temperature;
  float humidity;
  float vibration;
  float vibration_x;
  float vibration_y;
  float vibration_z;
  String temp_status;
  String vib_status;
  int motor_speed;
  String motor_direction;
  String motor_status;
  float motor_current;        // Current measurement
  float motor_power;          // Power calculation P = V × I
  float motor_voltage;        // Motor voltage for documentation
  String power_formula;       // Formula documentation
  String current_status;      // Current status
  float max_current;          // Peak current
  float total_energy;         // Energy consumption
};

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("╔═══════════════════════════════════════════════╗");
  Serial.println("║   ESP32 ENHANCED MQTT MOTOR MONITOR + ACS712  ║");
  Serial.println("║  Student: Gourav Shaw (T0436800)              ║");
  Serial.println("║  Enhanced MQTT + BTS7960 + Current Sensing    ║");
  Serial.println("║  FIXED: Timezone Consistency & Power Formula  ║");
  Serial.println("║  FIXED: P = V × I (12V Motor)                 ║");
  Serial.println("║  (Pressure monitoring removed)                ║");
  Serial.println("╚═══════════════════════════════════════════════╝");
  Serial.println("");
  
  // Initialize current sensor with enhanced calibration
  setupCurrentSensor();
  
  // Initialize motor control pins
  setupMotorControl();
  
  // Initialize WiFi
  setupWiFi();
  
  // FIXED: Setup time with consistent timezone (UTC)
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  delay(2000);
  Serial.println("🕐 Time configured for UTC consistency");
  
  // Setup MQTT with larger buffer for enhanced commands
  client.setServer(mqtt_server, mqtt_port);
  client.setBufferSize(1024);  // Increased for enhanced JSON commands
  client.setCallback(onMqttMessage);
  
  // Initialize BME280
  if (!bme.begin(0x76)) {
    if (!bme.begin(0x77)) {
      Serial.println("❌ BME280 failed");
      while (1);
    }
  }
  Serial.println("✅ BME280 initialized");
  
  // Initialize vibration sensor
  Wire.begin();
  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission();
  Serial.println("✅ MPU6050 initialized");
  
  Serial.println("");
  Serial.println("🎯 Starting enhanced MQTT motor monitoring with current sensing...");
  Serial.println("📡 Publishing to: " + String(topic_sensors));
  Serial.println("🎮 Motor control ready via: " + String(topic_control));
  Serial.println("⚡ Current monitoring active on GPIO34");
  Serial.println("🔋 Power calculation: P = V × I = 12V × Current");
  Serial.println("📊 Pressure monitoring disabled");
  Serial.println("🕐 Timezone: UTC for consistency");
  Serial.println("");
}

void loop() {
  // Maintain MQTT connection
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();
  
  // Read and process current sensor continuously
  updateCurrentReading();
  
  // Handle motor transitions
  handleMotorTransitions();
  
  // Update energy calculation
  updateEnergyCalculation();
  
  // Read sensor data and publish every 5 seconds
  static unsigned long lastSensorRead = 0;
  if (millis() - lastSensorRead >= 5000) {
    SensorReading data = readSensorData();
    displayLocalData(data);
    publishSensorData(data);
    
    // Check for automatic fault-based motor control (including current)
    checkAutomaticFaultResponse(data);
    
    lastSensorRead = millis();
  }
  
  delay(100); // Small delay for responsiveness
}

// ENHANCED: Current sensor setup with improved calibration
void setupCurrentSensor() {
  Serial.println("⚡ Initializing ACS712 Current Sensor...");
  
  // Configure ADC for current sensor
  analogSetAttenuation(ADC_11db);  // 0-3.3V range
  analogReadResolution(12);        // 12-bit resolution
  
  // Initialize current readings array
  for (int i = 0; i < CURRENT_SAMPLES; i++) {
    currentReadings[i] = 0.0;
  }
  
  // Calibrate zero current offset
  delay(1000);  // Let system stabilize
  calibrateCurrentSensor();
  
  Serial.println("✅ ACS712 Current Sensor initialized");
  Serial.println("⚡ Sensitivity: " + String(ACS712_SENSITIVITY) + " V/A");
  Serial.println("⚡ Zero current voltage: " + String(ACS712_ZERO_CURRENT_VOLTAGE) + " V");
  Serial.println("⚡ Motor voltage: " + String(MOTOR_VOLTAGE) + " V");
  Serial.println("⚡ Power formula: P = V × I");
  Serial.println("⚡ Calibration offset: " + String(currentOffset) + " V");
}

void calibrateCurrentSensor() {
  Serial.println("🔧 Calibrating current sensor (ensure motor is stopped)...");
  
  float sum = 0.0;
  const int calibrationSamples = 200;  // Increased for better calibration
  
  for (int i = 0; i < calibrationSamples; i++) {
    int rawValue = analogRead(CURRENT_SENSOR_PIN);
    float voltage = (rawValue / float(ADC_RESOLUTION)) * ADC_VOLTAGE_REF;
    sum += voltage;
    delay(10);
  }
  
  float averageVoltage = sum / calibrationSamples;
  currentOffset = averageVoltage - ACS712_ZERO_CURRENT_VOLTAGE;
  
  Serial.println("✅ Calibration complete");
  Serial.println("⚡ Measured zero voltage: " + String(averageVoltage) + " V");
  Serial.println("⚡ Calculated offset: " + String(currentOffset) + " V");
}

// ENHANCED: Current reading with improved filtering
void updateCurrentReading() {
  // Read ADC value
  int rawValue = analogRead(CURRENT_SENSOR_PIN);
  
  // Convert to voltage
  float voltage = (rawValue / float(ADC_RESOLUTION)) * ADC_VOLTAGE_REF;
  
  // Apply calibration offset
  voltage -= currentOffset;
  
  // Convert voltage to current using ACS712 sensitivity
  float current = (voltage - ACS712_ZERO_CURRENT_VOLTAGE) / ACS712_SENSITIVITY;
  
  // Take absolute value (bidirectional current measurement)
  current = abs(current);
  
  // Add to moving average filter
  currentReadings[currentSampleIndex] = current;
  currentSampleIndex = (currentSampleIndex + 1) % CURRENT_SAMPLES;
  
  if (currentSampleIndex == 0) {
    currentSamplesReady = true;
  }
  
  // Calculate filtered current (moving average)
  if (currentSamplesReady) {
    float sum = 0.0;
    for (int i = 0; i < CURRENT_SAMPLES; i++) {
      sum += currentReadings[i];
    }
    currentCurrent = sum / CURRENT_SAMPLES;
  } else {
    // Use simple average until we have enough samples
    float sum = 0.0;
    for (int i = 0; i <= currentSampleIndex; i++) {
      sum += currentReadings[i];
    }
    currentCurrent = sum / (currentSampleIndex + 1);
  }
  
  // Update peak current
  if (currentCurrent > maxCurrent) {
    maxCurrent = currentCurrent;
  }
  
  // FIXED: Calculate motor power using P = V × I
  motorPower = currentCurrent * MOTOR_VOLTAGE;
  
  // Update current status
  updateCurrentStatus();
}

// ENHANCED: Current status with better thresholds
void updateCurrentStatus() {
  if (currentCurrent < 0.1) {
    currentStatus = "IDLE";
  } else if (currentCurrent < 15.0) {
    currentStatus = "NORMAL";
  } else if (currentCurrent < 25.0) {
    currentStatus = "HIGH";
  } else if (currentCurrent < 30.0) {
    currentStatus = "OVERLOAD";
  } else {
    currentStatus = "FAULT";
  }
}

// ENHANCED: Energy calculation with better precision
void updateEnergyCalculation() {
  unsigned long currentTime = millis();
  
  if (lastEnergyUpdate == 0) {
    lastEnergyUpdate = currentTime;
    return;
  }
  
  unsigned long deltaTime = currentTime - lastEnergyUpdate;
  float deltaHours = deltaTime / 3600000.0;  // Convert ms to hours
  
  // Add current consumption to total energy (Amp-hours)
  totalEnergy += currentCurrent * deltaHours;
  
  lastEnergyUpdate = currentTime;
}

void setupMotorControl() {
  Serial.println("🔧 Initializing BTS7960 Motor Control...");
  
  // Initialize pins
  pinMode(MOTOR_RPWM, OUTPUT);
  pinMode(MOTOR_R_EN, OUTPUT);
  pinMode(MOTOR_LPWM, OUTPUT);
  pinMode(MOTOR_L_EN, OUTPUT);
  
  // Setup PWM channels
  ledcAttach(MOTOR_RPWM, 1000, 8);  // 1kHz, 8-bit resolution
  ledcAttach(MOTOR_LPWM, 1000, 8);
  
  // BTS7960 initialization - enables always HIGH
  digitalWrite(MOTOR_R_EN, HIGH);
  digitalWrite(MOTOR_L_EN, HIGH);
  
  // Start with motor at normal speed forward
  setMotorSpeed(currentMotorSpeed, currentDirection);
  
  Serial.println("✅ BTS7960 Motor Control initialized");
  Serial.println("⚙️ Initial state: " + String(currentMotorSpeed) + "% " + currentDirection);
}

void setupWiFi() {
  Serial.print("🌐 Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("✅ WiFi connected!");
    Serial.print("📡 ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("❌ WiFi failed");
    while(1);
  }
}

void reconnectMQTT() {
  while (!client.connected()) {
    Serial.print("🔄 Attempting MQTT connection to ");
    Serial.print(mqtt_server);
    Serial.print(":");
    Serial.print(mqtt_port);
    Serial.print("...");
    
    if (client.connect(mqtt_client_id)) {
      Serial.println(" connected!");
      
      // Subscribe to control topic
      client.subscribe(topic_control);
      Serial.println("📡 Subscribed to: " + String(topic_control));
      
      // Publish status with motor info
      publishMotorStatus("ESP32 Enhanced Motor Monitor with Current Sensing Online");
      
    } else {
      Serial.print(" failed, rc=");
      Serial.print(client.state());
      
      switch(client.state()) {
        case -4: Serial.println(" (Connection timeout)"); break;
        case -3: Serial.println(" (Connection lost)"); break;
        case -2: Serial.println(" (Connect failed)"); break;
        case -1: Serial.println(" (Disconnected)"); break;
        case 1: Serial.println(" (Bad protocol)"); break;
        case 2: Serial.println(" (Bad client ID)"); break;
        case 3: Serial.println(" (Unavailable)"); break;
        case 4: Serial.println(" (Bad credentials)"); break;
        case 5: Serial.println(" (Unauthorized)"); break;
        default: Serial.println(" (Unknown error)"); break;
      }
      
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.println("📨 Received MQTT: [" + String(topic) + "] " + message);
  
  if (String(topic) == topic_control) {
    handleMotorCommand(message);
  }
}

void handleMotorCommand(String message) {
  // Try to parse as enhanced JSON command first
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, message);
  
  if (error == DeserializationError::Ok && doc.containsKey("command")) {
    // Enhanced command format
    String command = doc["command"];
    
    if (command == "SET_MOTOR_STATE") {
      int speed = doc["speed"] | 100;  // Default to 100 if not specified
      String direction = doc["direction"] | "forward";
      String transition = doc["transition"] | "gradual";
      
      Serial.println("🎮 Enhanced command: " + String(speed) + "% " + direction + " (" + transition + ")");
      
      if (transition == "gradual") {
        startGradualTransition(speed, direction);
      } else {
        setMotorSpeed(speed, direction);
      }
      
      publishMotorFeedback("Enhanced command executed: " + String(speed) + "% " + direction);
      
    } else {
      handleBasicCommand(command);
    }
  } else {
    // Basic command format (backward compatibility)
    handleBasicCommand(message);
  }
}

void handleBasicCommand(String command) {
  if (command == "EMERGENCY_STOP") {
    Serial.println("🛑 Emergency stop command received!");
    emergencyStop();
    publishMotorFeedback("Emergency stop executed");
    
  } else if (command == "SPEED_REDUCE") {
    Serial.println("🐌 Speed reduction command received!");
    int newSpeed = max(0, currentMotorSpeed - 25);  // Reduce by 25%, minimum 0%
    String newDirection = (newSpeed == 0) ? "stopped" : currentDirection;
    startGradualTransition(newSpeed, newDirection);
    publishMotorFeedback("Speed reduced to " + String(newSpeed) + "%");
    
  } else if (command == "SPEED_INCREASE") {
    Serial.println("🚀 Speed increase command received!");
    int newSpeed = min(100, currentMotorSpeed + 25);  // Increase by 25%, maximum 100%
    String newDirection = (currentMotorSpeed == 0 && newSpeed > 0) ? "forward" : currentDirection;
    startGradualTransition(newSpeed, newDirection);
    publishMotorFeedback("Speed increased to " + String(newSpeed) + "%");
    
  } else if (command == "NORMAL_OPERATION") {
    Serial.println("✅ Normal operation command received!");
    motorEnabled = true;
    startGradualTransition(100, "forward");
    publishMotorFeedback("Normal operation resumed");
    
  } else {
    Serial.println("❓ Unknown command: " + command);
    publishMotorFeedback("Unknown command: " + command);
  }
}

void startGradualTransition(int newSpeed, String newDirection) {
  if (!motorEnabled && newSpeed > 0) {
    Serial.println("⚠️ Motor disabled - ignoring command");
    return;
  }
  
  targetSpeed = newSpeed;
  targetDirection = newDirection;
  isTransitioning = true;
  transitionStartTime = millis();
  motorStatus = "transitioning";
  
  Serial.println("🔄 Starting gradual transition: " + String(currentMotorSpeed) + "% " + currentDirection + 
                 " → " + String(targetSpeed) + "% " + targetDirection);
}

void handleMotorTransitions() {
  if (!isTransitioning) return;
  
  unsigned long elapsed = millis() - transitionStartTime;
  const unsigned long TRANSITION_TIME = 1500; // 1.5 seconds total transition
  
  if (elapsed >= TRANSITION_TIME) {
    // Transition complete
    currentMotorSpeed = targetSpeed;
    currentDirection = targetDirection;
    isTransitioning = false;
    motorStatus = (currentMotorSpeed == 0) ? "stopped" : "normal";
    
    setMotorSpeed(currentMotorSpeed, currentDirection);
    
    Serial.println("✅ Transition complete: " + String(currentMotorSpeed) + "% " + currentDirection);
    publishMotorFeedback("Transition complete: " + String(currentMotorSpeed) + "% " + currentDirection);
    return;
  }
  
  // Calculate intermediate values for smooth transition
  float progress = (float)elapsed / TRANSITION_TIME;
  
  // Handle direction change (stop in middle, then change direction)
  if (currentDirection != targetDirection) {
    if (progress < 0.5) {
      // First half: reduce to 0
      int intermediateSpeed = currentMotorSpeed * (1.0 - (progress * 2));
      setMotorSpeed(intermediateSpeed, currentDirection);
    } else {
      // Second half: increase in new direction
      int intermediateSpeed = targetSpeed * ((progress - 0.5) * 2);
      setMotorSpeed(intermediateSpeed, targetDirection);
      currentDirection = targetDirection; // Update direction
    }
  } else {
    // Same direction: smooth speed transition
    int intermediateSpeed = currentMotorSpeed + ((targetSpeed - currentMotorSpeed) * progress);
    setMotorSpeed(intermediateSpeed, currentDirection);
  }
}

void setMotorSpeed(int speed, String direction) {
  if (!motorEnabled && speed > 0) {
    speed = 0;
    direction = "stopped";
  }
  
  // Constrain speed
  speed = constrain(speed, 0, 100);
  
  // Convert percentage to PWM value (0-255)
  int pwmValue = map(speed, 0, 100, 0, 255);
  
  // Ensure enables are HIGH (BTS7960 requirement)
  digitalWrite(MOTOR_R_EN, HIGH);
  digitalWrite(MOTOR_L_EN, HIGH);
  
  if (speed == 0 || direction == "stopped") {
    // Stop motor
    ledcWrite(MOTOR_RPWM, 0);
    ledcWrite(MOTOR_LPWM, 0);
    currentDirection = "stopped";
  } else if (direction == "forward") {
    // Forward motion
    ledcWrite(MOTOR_LPWM, 0);
    ledcWrite(MOTOR_RPWM, pwmValue);
    currentDirection = "forward";
  } else if (direction == "reverse") {
    // Reverse motion
    ledcWrite(MOTOR_RPWM, 0);
    ledcWrite(MOTOR_LPWM, pwmValue);
    currentDirection = "reverse";
  }
  
  if (!isTransitioning) {
    currentMotorSpeed = speed;
  }
  
  lastMotorUpdate = millis();
}

void emergencyStop() {
  motorEnabled = false;
  isTransitioning = false;
  currentMotorSpeed = 0;
  currentDirection = "stopped";
  motorStatus = "emergency_stop";
  
  // Immediate stop
  ledcWrite(MOTOR_RPWM, 0);
  ledcWrite(MOTOR_LPWM, 0);
  
  Serial.println("🛑 EMERGENCY STOP ACTIVATED");
}

// ENHANCED: Automatic fault response with current monitoring
void checkAutomaticFaultResponse(SensorReading data) {
  // Enhanced automatic fault-based motor control (including current)
  if (!motorEnabled) return; // Don't interfere if manually disabled
  
  bool faultDetected = false;
  String faultReason = "";
  
  // ENHANCED: Current-based control with power analysis
  if (data.motor_current > 30.0) {
    emergencyStop();
    faultReason = "Critical current: " + String(data.motor_current) + " A (Power: " + String(data.motor_power) + " W)";
    faultDetected = true;
  } else if (data.motor_current > 25.0 && currentMotorSpeed > 25) {
    emergencyStop();
    faultReason = "Overcurrent protection: " + String(data.motor_current) + " A";
    faultDetected = true;
  } else if (data.motor_current > 20.0 && currentMotorSpeed > 50) {
    startGradualTransition(50, currentDirection);
    faultReason = "High current: reducing speed to 50%";
    faultDetected = true;
  }
  
  // Temperature-based control
  if (data.temperature > 45.0) {
    emergencyStop();
    faultReason = "Critical temperature: " + String(data.temperature) + "°C";
    faultDetected = true;
  } else if (data.temperature > 35.0 && currentMotorSpeed > 50) {
    startGradualTransition(50, currentDirection);
    faultReason = "High temperature: reducing speed to 50%";
    faultDetected = true;
  }
  
  // Vibration-based control
  if (data.vibration > 2.0) {
    emergencyStop();
    faultReason = "Critical vibration: " + String(data.vibration) + " m/s²";
    faultDetected = true;
  } else if (data.vibration > 1.0 && currentMotorSpeed > 75) {
    startGradualTransition(75, currentDirection);
    faultReason = "High vibration: reducing speed to 75%";
    faultDetected = true;
  }
  
  // ENHANCED: Power efficiency check
  if (data.motor_power > 0 && currentMotorSpeed > 0) {
    float powerPerSpeed = data.motor_power / currentMotorSpeed;
    if (powerPerSpeed > 5.0 && currentMotorSpeed > 25) {  // High power per speed ratio
      startGradualTransition(currentMotorSpeed - 25, currentDirection);
      faultReason = "Poor efficiency: reducing speed for optimization (P/S ratio: " + String(powerPerSpeed) + ")";
      faultDetected = true;
    }
  }
  
  if (faultDetected) {
    Serial.println("⚠️ Automatic fault response: " + faultReason);
    publishMotorFeedback("Automatic fault response: " + faultReason);
  }
}

// FIXED: Enhanced sensor data reading with proper timestamp and power calculation
SensorReading readSensorData() {
  SensorReading data;
  
  // FIXED: Get timestamp in UTC for consistency
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    char timestamp[25];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", &timeinfo);
    data.timestamp = String(timestamp);
  } else {
    data.timestamp = "UP_" + String(millis() / 1000);
  }
  
  // Read environmental data (removed pressure)
  data.temperature = bme.readTemperature();
  data.humidity = bme.readHumidity();
  
  // Read vibration data
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  
  if (Wire.requestFrom(0x68, 6) == 6) {
    int16_t raw_x = (Wire.read() << 8) | Wire.read();
    int16_t raw_y = (Wire.read() << 8) | Wire.read();
    int16_t raw_z = (Wire.read() << 8) | Wire.read();
    
    data.vibration_x = (float)raw_x * 2.0 * 9.81 / 32768.0;
    data.vibration_y = (float)raw_y * 2.0 * 9.81 / 32768.0;
    data.vibration_z = (float)raw_z * 2.0 * 9.81 / 32768.0;
    
    float magnitude = sqrt(data.vibration_x * data.vibration_x + 
                          data.vibration_y * data.vibration_y + 
                          data.vibration_z * data.vibration_z);
    data.vibration = abs(magnitude - 9.80665);
  } else {
    data.vibration = 0.0;
    data.vibration_x = 0.0;
    data.vibration_y = 0.0;
    data.vibration_z = 0.0;
  }
  
  // Calculate status
  if (data.temperature > 40) {
    data.temp_status = "HIGH";
  } else if (data.temperature > 30) {
    data.temp_status = "WARM";
  } else {
    data.temp_status = "NORMAL";
  }
  
  if (data.vibration > 1.0) {
    data.vib_status = "HIGH";
  } else if (data.vibration > 0.3) {
    data.vib_status = "ACTIVE";
  } else {
    data.vib_status = "LOW";
  }
  
  // Include motor data
  data.motor_speed = currentMotorSpeed;
  data.motor_direction = currentDirection;
  data.motor_status = motorStatus;
  
  // FIXED: Enhanced current sensor data with proper power calculation documentation
  data.motor_current = currentCurrent;
  data.motor_power = motorPower;              // P = V × I
  data.motor_voltage = MOTOR_VOLTAGE;         // Document voltage used
  data.power_formula = "P=V*I";              // Document formula
  data.current_status = currentStatus;
  data.max_current = maxCurrent;
  data.total_energy = totalEnergy;
  
  return data;
}

// FIXED: Enhanced MQTT payload with comprehensive power and current data
void publishSensorData(SensorReading data) {
  // Create enhanced JSON payload with motor and current data (removed pressure)
  StaticJsonDocument<550> jsonDoc;  // Increased size for additional metadata
  jsonDoc["timestamp"] = data.timestamp;
  jsonDoc["temperature"] = round(data.temperature * 10) / 10.0;
  jsonDoc["humidity"] = round(data.humidity * 10) / 10.0;
  jsonDoc["vibration"] = round(data.vibration * 1000) / 1000.0;
  jsonDoc["vibration_x"] = round(data.vibration_x * 100) / 100.0;
  jsonDoc["vibration_y"] = round(data.vibration_y * 100) / 100.0;
  jsonDoc["vibration_z"] = round(data.vibration_z * 100) / 100.0;
  jsonDoc["temp_status"] = data.temp_status;
  jsonDoc["vib_status"] = data.vib_status;
  
  // Motor data
  jsonDoc["motor_speed"] = data.motor_speed;
  jsonDoc["motor_direction"] = data.motor_direction;
  jsonDoc["motor_status"] = data.motor_status;
  jsonDoc["motor_enabled"] = motorEnabled;
  jsonDoc["is_transitioning"] = isTransitioning;
  
  // FIXED: Enhanced current sensor data with complete power calculation metadata
  jsonDoc["motor_current"] = round(data.motor_current * 100) / 100.0;
  jsonDoc["motor_power"] = round(data.motor_power * 100) / 100.0;
  jsonDoc["motor_voltage"] = data.motor_voltage;        // Document voltage
  jsonDoc["power_formula"] = data.power_formula;        // Document formula
  jsonDoc["current_status"] = data.current_status;
  jsonDoc["max_current"] = round(data.max_current * 100) / 100.0;
  jsonDoc["total_energy"] = round(data.total_energy * 1000) / 1000.0;
  
  // Enhanced metadata for debugging and documentation
  jsonDoc["device_id"] = "ESP32_001";
  jsonDoc["sensor_model"] = "ACS712-30A";           // Current sensor model
  jsonDoc["firmware_version"] = "v2.1_enhanced";    // Firmware version
  jsonDoc["timezone"] = "UTC";                      // Timezone for consistency
  
  String jsonString;
  serializeJson(jsonDoc, jsonString);
  
  Serial.print("📏 Sending ");
  Serial.print(jsonString.length());
  Serial.print(" bytes: ");
  Serial.println(jsonString);
  
  // Publish to MQTT
  if (client.publish(topic_sensors, jsonString.c_str())) {
    Serial.println("📡 Enhanced MQTT published successfully!");
    Serial.println("⚡ Power calculation: P = V × I = " + String(MOTOR_VOLTAGE) + "V × " + 
                   String(data.motor_current, 2) + "A = " + String(data.motor_power, 1) + "W");
  } else {
    Serial.print("❌ MQTT publish failed! State: ");
    Serial.println(client.state());
  }
}

// ENHANCED: Motor status publishing with power information
void publishMotorStatus(String message) {
  StaticJsonDocument<300> statusDoc;
  statusDoc["message"] = message;
  statusDoc["motor_speed"] = currentMotorSpeed;
  statusDoc["motor_direction"] = currentDirection;
  statusDoc["motor_status"] = motorStatus;
  statusDoc["motor_enabled"] = motorEnabled;
  statusDoc["motor_current"] = round(currentCurrent * 100) / 100.0;
  statusDoc["motor_power"] = round(motorPower * 100) / 100.0;
  statusDoc["motor_voltage"] = MOTOR_VOLTAGE;
  statusDoc["current_status"] = currentStatus;
  statusDoc["power_formula"] = "P=V*I";
  statusDoc["timestamp"] = millis();
  
  String statusString;
  serializeJson(statusDoc, statusString);
  
  client.publish(topic_status, statusString.c_str());
}

// ENHANCED: Motor feedback with comprehensive power data
void publishMotorFeedback(String feedback) {
  StaticJsonDocument<250> feedbackDoc;
  feedbackDoc["feedback"] = feedback;
  feedbackDoc["motor_speed"] = currentMotorSpeed;
  feedbackDoc["motor_direction"] = currentDirection;
  feedbackDoc["motor_status"] = motorStatus;
  feedbackDoc["motor_current"] = round(currentCurrent * 100) / 100.0;
  feedbackDoc["motor_power"] = round(motorPower * 100) / 100.0;
  feedbackDoc["current_status"] = currentStatus;
  feedbackDoc["timestamp"] = millis();
  
  String feedbackString;
  serializeJson(feedbackDoc, feedbackString);
  
  client.publish(topic_motor_feedback, feedbackString.c_str());
  Serial.println("📤 Motor feedback: " + feedback);
}

// FIXED: Enhanced local data display with comprehensive power analysis
void displayLocalData(SensorReading data) {
  Serial.println("═══════════════════════════════════════════════");
  Serial.print("🕐 Time (UTC): ");
  Serial.println(data.timestamp);
  Serial.print("🌡️  Temperature: ");
  Serial.print(data.temperature, 1);
  Serial.print(" °C (");
  Serial.print(data.temp_status);
  Serial.println(")");
  Serial.print("📳 Vibration: ");
  Serial.print(data.vibration, 3);
  Serial.print(" m/s² (");
  Serial.print(data.vib_status);
  Serial.println(")");
  Serial.print("💧 Humidity: ");
  Serial.print(data.humidity, 1);
  Serial.println(" %");
  
  // Enhanced motor status display with current data
  Serial.println("┌─── MOTOR STATUS ────────────────────────────┐");
  Serial.print("⚙️  Speed: ");
  Serial.print(data.motor_speed);
  Serial.print("% | Direction: ");
  Serial.print(data.motor_direction);
  Serial.print(" | Status: ");
  Serial.println(data.motor_status);
  
  // FIXED: Enhanced current monitoring display with comprehensive power analysis
  Serial.println("┌─── CURRENT MONITORING & POWER ANALYSIS ────┐");
  Serial.print("⚡ Current: ");
  Serial.print(data.motor_current, 2);
  Serial.print(" A | Voltage: ");
  Serial.print(data.motor_voltage, 1);
  Serial.print(" V | Power: ");
  Serial.print(data.motor_power, 1);
  Serial.print(" W");
  Serial.println();
  
  Serial.print("📊 Status: ");
  Serial.print(data.current_status);
  Serial.print(" | Formula: ");
  Serial.print(data.power_formula);
  Serial.print(" | Model: ACS712-30A");
  Serial.println();
  
  Serial.println("📐 Power Calculation Details:");
  Serial.println("   P = V × I = " + String(data.motor_voltage, 1) + "V × " + 
                 String(data.motor_current, 2) + "A = " + String(data.motor_power, 1) + "W");
  
  Serial.print("📈 Peak Current: ");
  Serial.print(data.max_current, 2);
  Serial.print(" A | Total Energy: ");
  Serial.print(data.total_energy, 3);
  Serial.println(" Ah");
  
  // ENHANCED: Power efficiency analysis with detailed breakdown
  if (data.motor_speed > 0 && data.motor_power > 0) {
    float efficiency = (data.motor_speed / 100.0) / (data.motor_power / 100.0);
    float powerPerSpeed = data.motor_power / data.motor_speed;
    float currentPerSpeed = data.motor_current / data.motor_speed;
    
    Serial.println("┌─── EFFICIENCY ANALYSIS ─────────────────────┐");
    Serial.print("🔋 Efficiency Ratio: ");
    Serial.print(efficiency, 3);
    Serial.print(" | Power/Speed: ");
    Serial.print(powerPerSpeed, 2);
    Serial.println(" W/%");
    
    Serial.print("⚡ Current/Speed: ");
    Serial.print(currentPerSpeed, 3);
    Serial.print(" A/% | Power Factor: ");
    Serial.print((data.motor_power / (data.motor_voltage * data.motor_current)) * 100, 1);
    Serial.println("%");
    
    // Load analysis
    Serial.print("⚙️  Motor Load Analysis: ");
    if (powerPerSpeed < 2.0) {
      Serial.println("Light Load (Optimal)");
    } else if (powerPerSpeed < 4.0) {
      Serial.println("Normal Load (Good)");
    } else if (powerPerSpeed < 6.0) {
      Serial.println("Heavy Load (Monitor)");
    } else {
      Serial.println("Overload Condition (Critical)");
    }
    
    // Efficiency rating
    Serial.print("🏆 Efficiency Rating: ");
    if (efficiency > 0.8) {
      Serial.println("Excellent (>80%)");
    } else if (efficiency > 0.6) {
      Serial.println("Good (60-80%)");
    } else if (efficiency > 0.4) {
      Serial.println("Fair (40-60%)");
    } else {
      Serial.println("Poor (<40%)");
    }
  }
  Serial.println("└─────────────────────────────────────────────┘");
  
  if (isTransitioning) {
    Serial.print("🔄 Transitioning to: ");
    Serial.print(targetSpeed);
    Serial.print("% ");
    Serial.print(targetDirection);
    Serial.print(" (Progress: ");
    float progress = (millis() - transitionStartTime) / 1500.0 * 100;
    Serial.print(min(100.0f, progress), 1);
    Serial.println("%)");
  }
  
  Serial.print("🔌 Motor Enabled: ");
  Serial.print(motorEnabled ? "YES" : "NO");
  Serial.print(" | Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");
  
  // System health indicators
  Serial.println("┌─── SYSTEM HEALTH ───────────────────────────┐");
  Serial.print("📶 WiFi: ");
  Serial.print(WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
  Serial.print(" | MQTT: ");
  Serial.print(client.connected() ? "Connected" : "Disconnected");
  Serial.println();
  
  Serial.print("🧠 Free Heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.print(" bytes | CPU Freq: ");
  Serial.print(ESP.getCpuFreqMHz());
  Serial.println(" MHz");
  
  Serial.print("🔋 Power Consumption Est: ");
  Serial.print(data.motor_power + 5.0, 1);  // Motor power + ESP32 consumption
  Serial.println(" W total");
  
  Serial.println("└─────────────────────────────────────────────┘");
  Serial.println("═══════════════════════════════════════════════");
}