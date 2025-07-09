const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const mqtt = require('mqtt');
const path = require('path');

const app = express();

// Express middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/motor_monitoring')
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch(err => console.log('❌ MongoDB connection error:', err));

// Sensor data schema (same as before + device_id)
const SensorData = mongoose.model('SensorData', {
  timestamp: String,
  temperature: Number,
  humidity: Number,
  pressure: Number,
  vibration: Number,
  vibration_x: Number,
  vibration_y: Number,
  vibration_z: Number,
  temp_status: String,
  vib_status: String,
  device_id: String,
  created_at: { type: Date, default: Date.now }
});

// MQTT Setup
const mqttClient = mqtt.connect('mqtt://localhost:1883');

mqttClient.on('connect', () => {
  console.log('✅ Connected to MQTT broker');
  
  // Subscribe to sensor data
  mqttClient.subscribe('motor/sensors', (err) => {
    if (!err) {
      console.log('📡 Subscribed to motor/sensors');
    }
  });
  
  // Subscribe to status updates
  mqttClient.subscribe('motor/status', (err) => {
    if (!err) {
      console.log('📡 Subscribed to motor/status');
    }
  });
});

mqttClient.on('message', async (topic, message) => {
  console.log(`📨 Received MQTT: [${topic}] ${message.toString()}`);
  
  if (topic === 'motor/sensors') {
    try {
      const sensorData = JSON.parse(message.toString());
      
      // Save to database (exactly same as before)
      const dbRecord = new SensorData(sensorData);
      await dbRecord.save();
      
      console.log('✅ Sensor data saved to database');
      
      // Check for fault conditions and send commands
      checkFaultConditions(sensorData);
      
    } catch (error) {
      console.log('❌ Error processing sensor data:', error);
    }
  }
});

// Fault detection and control logic
function checkFaultConditions(data) {
  if (data.temperature > 50 || data.vibration > 2.5) {
    console.log('🚨 CRITICAL FAULT DETECTED - Emergency stop!');
    mqttClient.publish('motor/control', 'EMERGENCY_STOP');
  } else if (data.temperature > 40 || data.vibration > 1.5) {
    console.log('⚠️  WARNING CONDITION - Reducing speed');
    mqttClient.publish('motor/control', 'SPEED_REDUCE');
  } else if (data.temp_status === 'NORMAL' && data.vib_status === 'LOW') {
    mqttClient.publish('motor/control', 'NORMAL_OPERATION');
  }
}

// REST API endpoints (EXACTLY SAME AS BEFORE)
app.get('/api/latest', async (req, res) => {
  try {
    const latestData = await SensorData.find()
      .sort({ created_at: -1 })
      .limit(100);
    res.json(latestData);
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

app.get('/api/current', async (req, res) => {
  try {
    const currentData = await SensorData.findOne().sort({ created_at: -1 });
    
    if (currentData) {
      const now = new Date();
      const dataAge = (now - new Date(currentData.created_at)) / 1000;
      
      const response = {
        ...currentData.toObject(),
        isConnected: dataAge < 30,
        dataAge: Math.round(dataAge)
      };
      
      res.json(response);
    } else {
      res.json({ 
        isConnected: false,
        message: "No data available"
      });
    }
  } catch (error) {
    res.status(500).json({ 
      status: 'error', 
      message: error.message,
      isConnected: false 
    });
  }
});

app.get('/api/recent', async (req, res) => {
  try {
    const recentData = await SensorData.find()
      .sort({ created_at: -1 })
      .limit(10);
    
    res.json(recentData);
  } catch (error) {
    console.log('❌ Error getting recent data:', error);
    res.status(500).json({ status: 'error', message: error.message });
  }
});

app.get('/api/stats', async (req, res) => {
  try {
    const totalRecords = await SensorData.countDocuments();
    const oldestRecord = await SensorData.findOne().sort({ created_at: 1 });
    const newestRecord = await SensorData.findOne().sort({ created_at: -1 });
    
    let systemUptime = 'Unknown';
    if (oldestRecord && newestRecord) {
      const uptimeMs = new Date(newestRecord.created_at) - new Date(oldestRecord.created_at);
      const uptimeHours = Math.floor(uptimeMs / (1000 * 60 * 60));
      const uptimeMinutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
      systemUptime = `${uptimeHours}h ${uptimeMinutes}m`;
    }
    
    res.json({
      totalRecords,
      systemUptime,
      databaseConnected: mongoose.connection.readyState === 1,
      mqttConnected: mqttClient.connected,
      serverStartTime: new Date().toISOString()
    });
  } catch (error) {
    console.log('❌ Error getting stats:', error);
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// NEW: MQTT control endpoint for dashboard
app.post('/api/control', (req, res) => {
  const { command } = req.body;
  
  console.log(`🎮 Dashboard control command: ${command}`);
  mqttClient.publish('motor/control', command);
  
  res.json({ status: 'success', message: `Command ${command} sent via MQTT` });
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log('\n🚀 MQTT Motor Monitor Server started!');
  console.log(`📱 Dashboard: http://localhost:${PORT}`);
  console.log(`🌐 Network access: http://10.153.191.34:${PORT}`);
  console.log(`📡 MQTT Broker: mqtt://localhost:1883`);
  console.log(`🔧 Topics: motor/sensors, motor/status, motor/control`);
  console.log('\n💡 Keep this window open while monitoring\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n👋 Shutting down server...');
  mqttClient.end();
  mongoose.connection.close();
  process.exit();
});