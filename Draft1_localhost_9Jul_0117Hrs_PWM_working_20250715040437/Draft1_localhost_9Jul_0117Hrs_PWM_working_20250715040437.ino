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
const char* mqtt_server = "10.225.203.34";  // Your Ubuntu IP
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_Motor_Monitor";

// MQTT topics
const char* topic_sensors = "motor/sensors";
const char* topic_status = "motor/status";
const char* topic_control = "motor/control";
const char* topic_motor_feedback = "motor/feedback";

// NTP settings
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0;
const int daylightOffset_sec = 3600;

// *** BTS7960 Motor Control Pins ***
#define MOTOR_RPWM  25    // RPWM pin (Forward)
#define MOTOR_R_EN  27    // R_EN pin  
#define MOTOR_LPWM  26    // LPWM pin (Reverse)
#define MOTOR_L_EN  14    // L_EN pin

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

WiFiClient espClient;
PubSubClient client(espClient);

struct SensorReading {
  String timestamp;
  float temperature;
  float humidity;
  float pressure;
  float vibration;
  float vibration_x;
  float vibration_y;
  float vibration_z;
  String temp_status;
  String vib_status;
  int motor_speed;
  String motor_direction;
  String motor_status;
};

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("╔═══════════════════════════════════════════════╗");
  Serial.println("║    ESP32 ENHANCED MQTT MOTOR MONITOR          ║");
  Serial.println("║  Student: Gourav Shaw (T0436800)              ║");
  Serial.println("║  Enhanced MQTT + BTS7960 Motor Control        ║");
  Serial.println("╚═══════════════════════════════════════════════╝");
  Serial.println("");
  
  // Initialize motor control pins
  setupMotorControl();
  
  // Initialize WiFi
  setupWiFi();
  
  // Setup time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  delay(2000);
  
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
  Serial.println("🎯 Starting enhanced MQTT motor monitoring...");
  Serial.println("📡 Publishing to: " + String(topic_sensors));
  Serial.println("🎮 Motor control ready via: " + String(topic_control));
  Serial.println("");
}

void loop() {
  // Maintain MQTT connection
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();
  
  // Handle motor transitions
  handleMotorTransitions();
  
  // Read sensor data and publish every 5 seconds
  static unsigned long lastSensorRead = 0;
  if (millis() - lastSensorRead >= 5000) {
    SensorReading data = readSensorData();
    displayLocalData(data);
    publishSensorData(data);
    
    // Check for automatic fault-based motor control
    checkAutomaticFaultResponse(data);
    
    lastSensorRead = millis();
  }
  
  delay(100); // Small delay for responsiveness
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
      publishMotorStatus("ESP32 Enhanced Motor Monitor Online");
      
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
    int newSpeed = max(25, currentMotorSpeed - 25);  // Reduce by 25%, minimum 25%
    startGradualTransition(newSpeed, currentDirection);
    publishMotorFeedback("Speed reduced to " + String(newSpeed) + "%");
    
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

void checkAutomaticFaultResponse(SensorReading data) {
  // Automatic fault-based motor control
  if (!motorEnabled) return; // Don't interfere if manually disabled
  
  bool faultDetected = false;
  String faultReason = "";
  
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
  
  if (faultDetected) {
    Serial.println("⚠️ Automatic fault response: " + faultReason);
    publishMotorFeedback("Automatic fault response: " + faultReason);
  }
}

SensorReading readSensorData() {
  SensorReading data;
  
  // Get timestamp
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    char timestamp[25];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", &timeinfo);
    data.timestamp = String(timestamp);
  } else {
    data.timestamp = "UP_" + String(millis() / 1000);
  }
  
  // Read environmental data
  data.temperature = bme.readTemperature();
  data.humidity = bme.readHumidity();
  data.pressure = bme.readPressure() / 100.0F;
  
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
  
  return data;
}

void publishSensorData(SensorReading data) {
  // Create enhanced JSON payload with motor data
  StaticJsonDocument<400> jsonDoc;
  jsonDoc["timestamp"] = data.timestamp;
  jsonDoc["temperature"] = round(data.temperature * 10) / 10.0;
  jsonDoc["humidity"] = round(data.humidity * 10) / 10.0;
  jsonDoc["pressure"] = round(data.pressure * 10) / 10.0;
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
  
  jsonDoc["device_id"] = "ESP32_001";
  
  String jsonString;
  serializeJson(jsonDoc, jsonString);
  
  Serial.print("📏 Sending ");
  Serial.print(jsonString.length());
  Serial.print(" bytes: ");
  Serial.println(jsonString);
  
  // Publish to MQTT
  if (client.publish(topic_sensors, jsonString.c_str())) {
    Serial.println("📡 Enhanced MQTT published successfully!");
  } else {
    Serial.print("❌ MQTT publish failed! State: ");
    Serial.println(client.state());
  }
}

void publishMotorStatus(String message) {
  StaticJsonDocument<200> statusDoc;
  statusDoc["message"] = message;
  statusDoc["motor_speed"] = currentMotorSpeed;
  statusDoc["motor_direction"] = currentDirection;
  statusDoc["motor_status"] = motorStatus;
  statusDoc["motor_enabled"] = motorEnabled;
  statusDoc["timestamp"] = millis();
  
  String statusString;
  serializeJson(statusDoc, statusString);
  
  client.publish(topic_status, statusString.c_str());
}

void publishMotorFeedback(String feedback) {
  StaticJsonDocument<150> feedbackDoc;
  feedbackDoc["feedback"] = feedback;
  feedbackDoc["motor_speed"] = currentMotorSpeed;
  feedbackDoc["motor_direction"] = currentDirection;
  feedbackDoc["motor_status"] = motorStatus;
  feedbackDoc["timestamp"] = millis();
  
  String feedbackString;
  serializeJson(feedbackDoc, feedbackString);
  
  client.publish(topic_motor_feedback, feedbackString.c_str());
  Serial.println("📤 Motor feedback: " + feedback);
}

void displayLocalData(SensorReading data) {
  Serial.println("═══════════════════════════════════════════════");
  Serial.print("🕐 Time: ");
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
  Serial.print("📊 Pressure: ");
  Serial.print(data.pressure, 1);
  Serial.println(" hPa");
  
  // Motor status display
  Serial.println("┌─── MOTOR STATUS ────────────────────────────┐");
  Serial.print("⚙️  Speed: ");
  Serial.print(data.motor_speed);
  Serial.print("% | Direction: ");
  Serial.print(data.motor_direction);
  Serial.print(" | Status: ");
  Serial.println(data.motor_status);
  
  if (isTransitioning) {
    Serial.print("🔄 Transitioning to: ");
    Serial.print(targetSpeed);
    Serial.print("% ");
    Serial.println(targetDirection);
  }
  
  Serial.print("🔌 Motor Enabled: ");
  Serial.println(motorEnabled ? "YES" : "NO");
  Serial.println("└─────────────────────────────────────────────┘");
  Serial.println("═══════════════════════════════════════════════");
}