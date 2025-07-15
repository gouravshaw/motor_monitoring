const express = require('express');
const mqtt = require('mqtt');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// MongoDB connection
const MONGODB_URI = 'mongodb://localhost:27017/motor_monitoring';
mongoose.connect(MONGODB_URI)
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch(err => console.error('❌ MongoDB connection error:', err));

// Enhanced Schema with motor data
const SensorSchema = new mongoose.Schema({
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
  motor_speed: { type: Number, default: 0 },
  motor_direction: { type: String, default: 'stopped' },
  motor_status: { type: String, default: 'stopped' },
  motor_enabled: { type: Boolean, default: true },
  is_transitioning: { type: Boolean, default: false },
  device_id: String,
  created_at: { type: Date, default: Date.now }
});

const SensorReading = mongoose.model('SensorReading', SensorSchema);

// Motor Command Schema for logging
const MotorCommandSchema = new mongoose.Schema({
  command: String,
  speed: Number,
  direction: String,
  transition: String,
  source: { type: String, default: 'dashboard' },
  executed_at: { type: Date, default: Date.now },
  feedback: String
});

const MotorCommand = mongoose.model('MotorCommand', MotorCommandSchema);

// MQTT Setup
const MQTT_BROKER = 'mqtt://localhost:1883';
const client = mqtt.connect(MQTT_BROKER);

// Global variables
let latestSensorData = null;
let lastDataTime = Date.now();
let mqttConnected = false;
let systemStartTime = Date.now();

// MQTT Topics
const TOPICS = {
  SENSORS: 'motor/sensors',
  STATUS: 'motor/status', 
  CONTROL: 'motor/control',
  FEEDBACK: 'motor/feedback'
};

client.on('connect', () => {
  console.log('🟢 Connected to MQTT broker');
  mqttConnected = true;
  
  // Subscribe to all motor topics
  Object.values(TOPICS).forEach(topic => {
    client.subscribe(topic, (err) => {
      if (err) {
        console.error(`❌ Failed to subscribe to ${topic}:`, err);
      } else {
        console.log(`📡 Subscribed to ${topic}`);
      }
    });
  });
});

client.on('error', (err) => {
  console.error('❌ MQTT connection error:', err);
  mqttConnected = false;
});

client.on('reconnect', () => {
  console.log('🔄 Reconnecting to MQTT broker...');
});

client.on('message', async (topic, message) => {
  try {
    const data = JSON.parse(message.toString());
    console.log(`📨 MQTT [${topic}]: ${message.toString().substring(0, 150)}...`);
    
    if (topic === TOPICS.SENSORS) {
      // Handle sensor data with motor information
      latestSensorData = {
        ...data,
        isConnected: true,
        dataAge: 0
      };
      lastDataTime = Date.now();
      
      // Save enhanced data to MongoDB
      try {
        const sensorReading = new SensorReading({
          timestamp: data.timestamp,
          temperature: data.temperature,
          humidity: data.humidity,
          pressure: data.pressure,
          vibration: data.vibration,
          vibration_x: data.vibration_x,
          vibration_y: data.vibration_y,
          vibration_z: data.vibration_z,
          temp_status: data.temp_status,
          vib_status: data.vib_status,
          motor_speed: data.motor_speed,
          motor_direction: data.motor_direction,
          motor_status: data.motor_status,
          motor_enabled: data.motor_enabled,
          is_transitioning: data.is_transitioning,
          device_id: data.device_id
        });
        
        await sensorReading.save();
        console.log('💾 Enhanced sensor data saved to MongoDB');
      } catch (saveError) {
        console.error('❌ Error saving to MongoDB:', saveError);
      }
      
    } else if (topic === TOPICS.STATUS) {
      // Handle motor status updates
      console.log('📊 Motor Status Update:', data);
      
    } else if (topic === TOPICS.FEEDBACK) {
      // Handle motor feedback
      console.log('📤 Motor Feedback:', data);
      
      // Save command feedback
      try {
        await MotorCommand.findOneAndUpdate(
          { executed_at: { $gte: new Date(Date.now() - 5000) } }, // Last 5 seconds
          { feedback: data.feedback },
          { sort: { executed_at: -1 } }
        );
      } catch (updateError) {
        console.error('❌ Error updating command feedback:', updateError);
      }
    }
    
  } catch (parseError) {
    console.error('❌ Error parsing MQTT message:', parseError);
  }
});

// Enhanced API Routes

// Get current sensor data with motor status
app.get('/api/current', (req, res) => {
  if (!latestSensorData) {
    return res.json({ message: 'No data available yet' });
  }
  
  const currentTime = Date.now();
  const dataAge = Math.floor((currentTime - lastDataTime) / 1000);
  const isConnected = dataAge < 30; // Consider connected if data is less than 30 seconds old
  
  res.json({
    ...latestSensorData,
    isConnected: isConnected,
    dataAge: dataAge
  });
});

// Enhanced motor control endpoint
app.post('/api/control', async (req, res) => {
  try {
    console.log('🎮 Received control command:', req.body);
    
    const command = req.body;
    
    // Log command to database
    const motorCommand = new MotorCommand({
      command: command.command || 'UNKNOWN',
      speed: command.speed,
      direction: command.direction,
      transition: command.transition,
      source: 'dashboard'
    });
    
    await motorCommand.save();
    console.log('💾 Motor command logged to database');
    
    // Publish enhanced command to MQTT
    const mqttMessage = JSON.stringify(command);
    console.log('📡 Publishing MQTT message:', mqttMessage);
    
    client.publish(TOPICS.CONTROL, mqttMessage, (err) => {
      if (err) {
        console.error('❌ Failed to publish MQTT command:', err);
        return res.status(500).json({ 
          success: false, 
          error: 'Failed to send MQTT command',
          details: err.message 
        });
      } else {
        console.log('📡 Enhanced MQTT command published successfully');
        res.json({ 
          success: true, 
          message: 'Enhanced motor command sent successfully',
          command: command,
          timestamp: new Date().toISOString()
        });
      }
    });
    
  } catch (error) {
    console.error('❌ Error in control endpoint:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Internal server error',
      details: error.message 
    });
  }
});

// NEW: Enhanced motor control endpoint for speed/direction
app.post('/api/motor/set', async (req, res) => {
  try {
    const { speed, direction, transition } = req.body;
    
    console.log(`🎮 Motor Set Command: ${speed}% ${direction} (${transition || 'gradual'})`);
    
    const command = {
      command: 'SET_MOTOR_STATE',
      speed: parseInt(speed) || 0,
      direction: direction || 'forward',
      transition: transition || 'gradual'
    };
    
    // Log command to database
    const motorCommand = new MotorCommand({
      command: 'SET_MOTOR_STATE',
      speed: command.speed,
      direction: command.direction,
      transition: command.transition,
      source: 'dashboard'
    });
    
    await motorCommand.save();
    
    // Publish to MQTT
    const mqttMessage = JSON.stringify(command);
    console.log('📡 Publishing motor control:', mqttMessage);
    
    client.publish(TOPICS.CONTROL, mqttMessage, (err) => {
      if (err) {
        console.error('❌ Failed to publish motor command:', err);
        return res.status(500).json({ 
          success: false, 
          error: 'Failed to send motor command',
          details: err.message 
        });
      } else {
        console.log('✅ Motor control command sent successfully');
        res.json({ 
          success: true, 
          message: `Motor set to ${speed}% ${direction}`,
          command: command,
          timestamp: new Date().toISOString()
        });
      }
    });
    
  } catch (error) {
    console.error('❌ Error in motor set endpoint:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Internal server error',
      details: error.message 
    });
  }
});

// Manual test endpoint for debugging (speed only)
app.get('/api/test-motor/:speed', (req, res) => {
  const speed = parseInt(req.params.speed) || 50;
  const direction = 'forward';
  
  console.log(`🧪 Manual motor test: ${speed}% ${direction}`);
  
  const command = {
    command: 'SET_MOTOR_STATE',
    speed: speed,
    direction: direction,
    transition: 'gradual'
  };
  
  console.log('🧪 Testing command:', command);
  
  client.publish(TOPICS.CONTROL, JSON.stringify(command), (err) => {
    if (err) {
      console.error('❌ Manual test command failed:', err);
      return res.status(500).json({ error: 'Manual test command failed' });
    }
    
    console.log('✅ Manual test command sent successfully');
    res.json({ 
      success: true, 
      message: `Manual test: ${speed}% ${direction}`,
      command: command 
    });
  });
});

// Manual test endpoint for debugging (speed and direction)
app.get('/api/test-motor/:speed/:direction', (req, res) => {
  const speed = parseInt(req.params.speed) || 50;
  const direction = req.params.direction || 'forward';
  
  console.log(`🧪 Manual motor test: ${speed}% ${direction}`);
  
  const command = {
    command: 'SET_MOTOR_STATE',
    speed: speed,
    direction: direction,
    transition: 'gradual'
  };
  
  console.log('🧪 Testing command:', command);
  
  client.publish(TOPICS.CONTROL, JSON.stringify(command), (err) => {
    if (err) {
      console.error('❌ Manual test command failed:', err);
      return res.status(500).json({ error: 'Manual test command failed' });
    }
    
    console.log('✅ Manual test command sent successfully');
    res.json({ 
      success: true, 
      message: `Manual test: ${speed}% ${direction}`,
      command: command 
    });
  });
});

// Get recent readings for historical view
app.get('/api/recent', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    
    const recentReadings = await SensorReading
      .find()
      .sort({ created_at: -1 })
      .limit(limit)
      .lean();
    
    console.log(`📊 Retrieved ${recentReadings.length} recent readings from MongoDB`);
    res.json(recentReadings);
    
  } catch (error) {
    console.error('❌ Error fetching recent readings:', error);
    res.status(500).json({ 
      error: 'Failed to fetch recent readings',
      details: error.message 
    });
  }
});

// Get motor command history
app.get('/api/commands', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    
    const commands = await MotorCommand
      .find()
      .sort({ executed_at: -1 })
      .limit(limit)
      .lean();
    
    console.log(`📊 Retrieved ${commands.length} motor commands from database`);
    res.json(commands);
    
  } catch (error) {
    console.error('❌ Error fetching motor commands:', error);
    res.status(500).json({ 
      error: 'Failed to fetch motor commands',
      details: error.message 
    });
  }
});

// Enhanced system statistics
app.get('/api/stats', async (req, res) => {
  try {
    const totalRecords = await SensorReading.countDocuments();
    const totalCommands = await MotorCommand.countDocuments();
    const uptimeMs = Date.now() - systemStartTime;
    const uptimeHours = Math.floor(uptimeMs / (1000 * 60 * 60));
    const uptimeMinutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
    
    // Get recent motor activity
    const recentCommands = await MotorCommand
      .find()
      .sort({ executed_at: -1 })
      .limit(5)
      .lean();
    
    // Database connectivity check
    const databaseConnected = mongoose.connection.readyState === 1;
    
    res.json({
      totalRecords,
      totalCommands,
      systemUptime: `${uptimeHours}h ${uptimeMinutes}m`,
      databaseConnected,
      mqttConnected,
      lastDataAge: latestSensorData ? Math.floor((Date.now() - lastDataTime) / 1000) : null,
      recentCommands: recentCommands.map(cmd => ({
        command: cmd.command,
        speed: cmd.speed,
        direction: cmd.direction,
        executed_at: cmd.executed_at,
        feedback: cmd.feedback
      }))
    });
    
  } catch (error) {
    console.error('❌ Error fetching enhanced stats:', error);
    res.status(500).json({ 
      error: 'Failed to fetch system statistics',
      details: error.message 
    });
  }
});

// Motor diagnostics endpoint
app.get('/api/motor/diagnostics', async (req, res) => {
  try {
    const now = new Date();
    const oneHourAgo = new Date(now - 60 * 60 * 1000);
    
    // Get motor data from last hour
    const motorData = await SensorReading
      .find({ 
        created_at: { $gte: oneHourAgo },
        motor_speed: { $exists: true }
      })
      .sort({ created_at: -1 })
      .lean();
    
    // Calculate motor diagnostics
    const diagnostics = {
      totalOperatingTime: motorData.length * 5, // 5 seconds per reading
      averageSpeed: motorData.length > 0 ? 
        motorData.reduce((sum, d) => sum + (d.motor_speed || 0), 0) / motorData.length : 0,
      speedChanges: 0,
      directionChanges: 0,
      emergencyStops: 0,
      transitions: motorData.filter(d => d.is_transitioning).length
    };
    
    // Count speed and direction changes
    for (let i = 1; i < motorData.length; i++) {
      if (motorData[i].motor_speed !== motorData[i-1].motor_speed) {
        diagnostics.speedChanges++;
      }
      if (motorData[i].motor_direction !== motorData[i-1].motor_direction) {
        diagnostics.directionChanges++;
      }
      if (motorData[i].motor_status === 'emergency_stop') {
        diagnostics.emergencyStops++;
      }
    }
    
    res.json(diagnostics);
    
  } catch (error) {
    console.error('❌ Error generating motor diagnostics:', error);
    res.status(500).json({ 
      error: 'Failed to generate motor diagnostics',
      details: error.message 
    });
  }
});

// Serve the enhanced dashboard
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║     🚀 ENHANCED MQTT MOTOR DASHBOARD           ║');
  console.log('║                                                ║');
  console.log('║  Student: Gourav Shaw (T0436800)               ║');
  console.log('║  Enhanced Motor Control + BTS7960              ║');
  console.log('║                                                ║');
  console.log(`║  🌐 Server: http://localhost:${PORT}              ║`);
  console.log('║  📡 MQTT: mqtt://localhost:1883               ║');
  console.log('║  💾 MongoDB: mongodb://localhost:27017        ║');
  console.log('║                                                ║');
  console.log('║  🎮 Features:                                  ║');
  console.log('║  • Real-time sensor monitoring                ║');
  console.log('║  • Advanced motor control (BTS7960)           ║');
  console.log('║  • Variable speed control (0-100%)            ║');
  console.log('║  • Direction control (Forward/Reverse)        ║');
  console.log('║  • Gradual speed transitions                  ║');
  console.log('║  • Automatic fault responses                  ║');
  console.log('║  • Motor command logging                      ║');
  console.log('║  • Enhanced MQTT communication               ║');
  console.log('╚════════════════════════════════════════════════╝');
});