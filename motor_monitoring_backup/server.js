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
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Enhanced Schema with current sensor data (removed pressure)
const SensorSchema = new mongoose.Schema({
  timestamp: String,
  temperature: Number,
  humidity: Number,
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
  
  // Current sensor fields
  motor_current: { type: Number, default: 0 },
  motor_power: { type: Number, default: 0 },
  motor_voltage: { type: Number, default: 12.0 }, // Added voltage field
  power_formula: { type: String, default: 'P=V*I' }, // Added formula documentation
  current_status: { type: String, default: 'IDLE' },
  max_current: { type: Number, default: 0 },
  total_energy: { type: Number, default: 0 },
  
  device_id: String,
  created_at: { type: Date, default: Date.now }
});

const SensorReading = mongoose.model('SensorReading', SensorSchema, 'sensorreadings');

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

const MotorCommand = mongoose.model('MotorCommand', MotorCommandSchema, 'motorcommands');

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
  console.log('Connected to MQTT broker');
  mqttConnected = true;
  
  // Subscribe to all motor topics
  Object.values(TOPICS).forEach(topic => {
    client.subscribe(topic, (err) => {
      if (err) {
        console.error(`Failed to subscribe to ${topic}:`, err);
      } else {
        console.log(`📡 Subscribed to ${topic}`);
      }
    });
  });
});

client.on('error', (err) => {
  console.error('MQTT connection error:', err);
  mqttConnected = false;
});

client.on('reconnect', () => {
  console.log('Reconnecting to MQTT broker...');
});

client.on('message', async (topic, message) => {
  try {
    const data = JSON.parse(message.toString());
    console.log(`📨 MQTT [${topic}]: ${message.toString().substring(0, 150)}...`);
    
    if (topic === TOPICS.SENSORS) {
      // Handle sensor data with enhanced motor and current information
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
          
          // Current sensor data
          motor_current: data.motor_current || 0,
          motor_power: data.motor_power || 0,
          motor_voltage: data.motor_voltage || 12.0,
          power_formula: data.power_formula || 'P=V*I',
          current_status: data.current_status || 'IDLE',
          max_current: data.max_current || 0,
          total_energy: data.total_energy || 0,
          
          device_id: data.device_id
        });
        
        await sensorReading.save();
        console.log('Enhanced sensor data with current monitoring saved to MongoDB');
      } catch (saveError) {
        console.error('Error saving to MongoDB:', saveError);
      }
      
    } else if (topic === TOPICS.STATUS) {
      // Handle motor status updates
      console.log('Motor Status Update:', data);
      
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
        console.error('Error updating command feedback:', updateError);
      }
    }
    
  } catch (parseError) {
    console.error('Error parsing MQTT message:', parseError);
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
    console.log('Motor command logged to database');
    
    // Publish enhanced command to MQTT
    const mqttMessage = JSON.stringify(command);
    console.log('📡 Publishing MQTT message:', mqttMessage);
    
    client.publish(TOPICS.CONTROL, mqttMessage, (err) => {
      if (err) {
        console.error('Failed to publish MQTT command:', err);
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
    console.error('Error in control endpoint:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Internal server error',
      details: error.message 
    });
  }
});

// Enhanced motor control endpoint for speed/direction
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
        console.error('Failed to publish motor command:', err);
        return res.status(500).json({ 
          success: false, 
          error: 'Failed to send motor command',
          details: err.message 
        });
      } else {
        console.log('Motor control command sent successfully');
        res.json({ 
          success: true, 
          message: `Motor set to ${speed}% ${direction}`,
          command: command,
          timestamp: new Date().toISOString()
        });
      }
    });
    
  } catch (error) {
    console.error('Error in motor set endpoint:', error);
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
      console.error('Manual test command failed:', err);
      return res.status(500).json({ error: 'Manual test command failed' });
    }
    
    console.log('Manual test command sent successfully');
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
      console.error('Manual test command failed:', err);
      return res.status(500).json({ error: 'Manual test command failed' });
    }
    
    console.log('Manual test command sent successfully');
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
    
    console.log(`Retrieved ${recentReadings.length} recent readings from MongoDB`);
    res.json(recentReadings);
    
  } catch (error) {
    console.error('Error fetching recent readings:', error);
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
    
    console.log(`Retrieved ${commands.length} motor commands from database`);
    res.json(commands);
    
  } catch (error) {
    console.error('Error fetching motor commands:', error);
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
    console.error('Error fetching enhanced stats:', error);
    res.status(500).json({ 
      error: 'Failed to fetch system statistics',
      details: error.message 
    });
  }
});

// Enhanced motor diagnostics endpoint with current data
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
    
    // Calculate enhanced motor diagnostics with current data
    const diagnostics = {
      totalOperatingTime: motorData.length * 5, // 5 seconds per reading
      averageSpeed: motorData.length > 0 ? 
        motorData.reduce((sum, d) => sum + (d.motor_speed || 0), 0) / motorData.length : 0,
      
      // Current-based diagnostics
      averageCurrent: motorData.length > 0 ?
        motorData.reduce((sum, d) => sum + (d.motor_current || 0), 0) / motorData.length : 0,
      peakCurrent: Math.max(...motorData.map(d => d.motor_current || 0)),
      totalEnergyConsumed: motorData.length > 0 ? 
        Math.max(...motorData.map(d => d.total_energy || 0)) : 0,
      averagePower: motorData.length > 0 ?
        motorData.reduce((sum, d) => sum + (d.motor_power || 0), 0) / motorData.length : 0,
      
      speedChanges: 0,
      directionChanges: 0,
      emergencyStops: 0,
      transitions: motorData.filter(d => d.is_transitioning).length,
      
      // Current anomalies
      currentSpikes: motorData.filter(d => (d.motor_current || 0) > 25).length,
      overloadEvents: motorData.filter(d => d.current_status === 'OVERLOAD').length,
      faultEvents: motorData.filter(d => d.current_status === 'FAULT').length,
      
      // Power efficiency metrics
      powerEfficiency: 0,
      currentStability: 0
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
    
    // Calculate power efficiency (Speed/Power ratio)
    const powerData = motorData.filter(d => d.motor_power > 0 && d.motor_speed > 0);
    if (powerData.length > 0) {
      diagnostics.powerEfficiency = powerData.reduce((sum, d) => 
        sum + (d.motor_speed / d.motor_power), 0) / powerData.length;
    }
    
    // Calculate current stability (coefficient of variation)
    const currentData = motorData.map(d => d.motor_current || 0).filter(c => c > 0);
    if (currentData.length > 1) {
      const mean = currentData.reduce((sum, c) => sum + c, 0) / currentData.length;
      const variance = currentData.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / currentData.length;
      const stdDev = Math.sqrt(variance);
      diagnostics.currentStability = mean > 0 ? (stdDev / mean) : 0;
    }
    
    res.json(diagnostics);
    
  } catch (error) {
    console.error('Error generating enhanced motor diagnostics:', error);
    res.status(500).json({ 
      error: 'Failed to generate motor diagnostics',
      details: error.message 
    });
  }
});

// ===== ENHANCED ANALYTICS API ENDPOINTS WITH TIMEZONE FIXES =====

// FIXED: Analytics data endpoint with filtering and aggregation - TIMEZONE FIXED
app.get('/api/analytics/data', async (req, res) => {
  try {
    const { 
      startDate, 
      endDate, 
      aggregation = 'raw',
      page = 1,
      limit = 100,
      export: exportFormat
    } = req.query;

    console.log('Analytics data request:', { startDate, endDate, aggregation, page, limit });

    // FIXED: Build date filter with proper timezone handling
    const dateFilter = {};
    if (startDate) {
      // Convert from local datetime-local input to UTC for MongoDB query
      const startDateUTC = new Date(startDate);
      dateFilter.$gte = startDateUTC;
      console.log('Start date converted:', startDate, '→', startDateUTC.toISOString());
    }
    if (endDate) {
      // Convert from local datetime-local input to UTC for MongoDB query
      const endDateUTC = new Date(endDate);
      dateFilter.$lte = endDateUTC;
      console.log('End date converted:', endDate, '→', endDateUTC.toISOString());
    }

    let query = {};
    if (Object.keys(dateFilter).length > 0) {
      query.created_at = dateFilter;
    }

    let data;
    let totalCount;

    if (aggregation === 'raw') {
      // Raw data with pagination
      const skip = (parseInt(page) - 1) * parseInt(limit);
      
      if (exportFormat) {
        // For export, get all data without pagination
        data = await SensorReading
          .find(query)
          .sort({ created_at: -1 })
          .lean();
        totalCount = data.length;
      } else {
        // For regular requests, use pagination
        data = await SensorReading
          .find(query)
          .sort({ created_at: -1 })
          .skip(skip)
          .limit(parseInt(limit))
          .lean();
        totalCount = await SensorReading.countDocuments(query);
      }
    } else {
      // Aggregated data
      const groupBy = getGroupByExpression(aggregation);
      
      data = await SensorReading.aggregate([
        { $match: query },
        {
          $group: {
            _id: groupBy,
            count: { $sum: 1 },
            avg_temperature: { $avg: '$temperature' },
            avg_humidity: { $avg: '$humidity' },
            avg_vibration: { $avg: '$vibration' },
            avg_motor_speed: { $avg: '$motor_speed' },
            avg_motor_current: { $avg: '$motor_current' },
            avg_motor_power: { $avg: '$motor_power' },
            max_temperature: { $max: '$temperature' },
            min_temperature: { $min: '$temperature' },
            max_vibration: { $max: '$vibration' },
            max_current: { $max: '$motor_current' },
            fault_count: {
              $sum: {
                $cond: [
                  { $or: [
                    { $eq: ['$temp_status', 'HIGH'] },
                    { $eq: ['$vib_status', 'HIGH'] },
                    { $eq: ['$current_status', 'OVERLOAD'] },
                    { $eq: ['$current_status', 'FAULT'] }
                  ]},
                  1, 0
                ]
              }
            }
          }
        },
        { $sort: { '_id': -1 } }
      ]);
      
      totalCount = data.length;
    }

    console.log(`Retrieved ${data.length} analytics records`);

    const response = {
      data,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: totalCount,
        pages: Math.ceil(totalCount / parseInt(limit))
      },
      aggregation,
      dateRange: { startDate, endDate },
      timezone: 'UTC', // Indicate data is stored in UTC
      localTimezone: Intl.DateTimeFormat().resolvedOptions().timeZone // Server local timezone
    };

    res.json(response);

  } catch (error) {
    console.error('Error fetching analytics data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch analytics data',
      details: error.message 
    });
  }
});

// FIXED: Enhanced analytics statistics endpoint with current data - TIMEZONE FIXED
app.get('/api/analytics/stats', async (req, res) => {
  try {
    const { startDate, endDate } = req.query;

    console.log('Analytics stats request:', { startDate, endDate });

    // FIXED: Build date filter with proper timezone handling
    const dateFilter = {};
    if (startDate) {
      const startDateUTC = new Date(startDate);
      dateFilter.$gte = startDateUTC;
      console.log('Stats start date converted:', startDate, '→', startDateUTC.toISOString());
    }
    if (endDate) {
      const endDateUTC = new Date(endDate);
      dateFilter.$lte = endDateUTC;
      console.log('Stats end date converted:', endDate, '→', endDateUTC.toISOString());
    }

    let matchQuery = {};
    if (Object.keys(dateFilter).length > 0) {
      matchQuery.created_at = dateFilter;
    }

    console.log('Match query:', matchQuery);

    const stats = await SensorReading.aggregate([
      { $match: matchQuery },
      {
        $group: {
          _id: null,
          totalReadings: { $sum: 1 },
          avgTemperature: { $avg: '$temperature' },
          minTemperature: { $min: '$temperature' },
          maxTemperature: { $max: '$temperature' },
          avgHumidity: { $avg: '$humidity' },
          avgVibration: { $avg: '$vibration' },
          maxVibration: { $max: '$vibration' },
          avgMotorSpeed: { $avg: '$motor_speed' },
          
          // Current sensor statistics
          avgMotorCurrent: { $avg: '$motor_current' },
          maxMotorCurrent: { $max: '$motor_current' },
          avgMotorPower: { $avg: '$motor_power' },
          maxMotorPower: { $max: '$motor_power' },
          totalEnergyConsumed: { $max: '$total_energy' },
          
          temperatureFaults: {
            $sum: { $cond: [{ $eq: ['$temp_status', 'HIGH'] }, 1, 0] }
          },
          vibrationFaults: {
            $sum: { $cond: [{ $eq: ['$vib_status', 'HIGH'] }, 1, 0] }
          },
          
          // Current-based fault detection
          currentOverloads: {
            $sum: { $cond: [{ $eq: ['$current_status', 'OVERLOAD'] }, 1, 0] }
          },
          currentFaults: {
            $sum: { $cond: [{ $eq: ['$current_status', 'FAULT'] }, 1, 0] }
          }
        }
      }
    ]);

    const result = stats[0] || {
      totalReadings: 0,
      avgTemperature: 0,
      minTemperature: 0,
      maxTemperature: 0,
      avgHumidity: 0,
      avgVibration: 0,
      maxVibration: 0,
      avgMotorSpeed: 0,
      avgMotorCurrent: 0,
      maxMotorCurrent: 0,
      avgMotorPower: 0,
      maxMotorPower: 0,
      totalEnergyConsumed: 0,
      temperatureFaults: 0,
      vibrationFaults: 0,
      currentOverloads: 0,
      currentFaults: 0
    };

    // Calculate fault rates
    result.temperatureFaultRate = result.totalReadings > 0 ? 
      (result.temperatureFaults / result.totalReadings * 100) : 0;
    result.vibrationFaultRate = result.totalReadings > 0 ? 
      (result.vibrationFaults / result.totalReadings * 100) : 0;
    result.currentOverloadRate = result.totalReadings > 0 ? 
      (result.currentOverloads / result.totalReadings * 100) : 0;
    result.currentFaultRate = result.totalReadings > 0 ? 
      (result.currentFaults / result.totalReadings * 100) : 0;

    // Calculate power efficiency metrics
    if (result.avgMotorSpeed > 0 && result.avgMotorPower > 0) {
      result.powerEfficiency = result.avgMotorSpeed / result.avgMotorPower;
      result.specificPowerConsumption = result.avgMotorPower / result.avgMotorSpeed; // W per %speed
    }

    // FIXED: Add timezone information to response
    result.timezone = 'UTC';
    result.queryDateRange = { startDate, endDate };

    console.log('Enhanced analytics statistics calculated:', result);
    res.json(result);

  } catch (error) {
    console.error('Error calculating enhanced analytics stats:', error);
    res.status(500).json({ 
      error: 'Failed to calculate statistics',
      details: error.message 
    });
  }
});

// FIXED: Enhanced correlations endpoint with current data - TIMEZONE FIXED
app.get('/api/analytics/correlations', async (req, res) => {
  try {
    const { startDate, endDate } = req.query;

    console.log('Analytics correlations request:', { startDate, endDate });

    // FIXED: Build date filter with proper timezone handling
    const dateFilter = {};
    if (startDate) {
      const startDateUTC = new Date(startDate);
      dateFilter.$gte = startDateUTC;
      console.log('Correlations start date converted:', startDate, '→', startDateUTC.toISOString());
    }
    if (endDate) {
      const endDateUTC = new Date(endDate);
      dateFilter.$lte = endDateUTC;
      console.log('Correlations end date converted:', endDate, '→', endDateUTC.toISOString());
    }

    let matchQuery = {};
    if (Object.keys(dateFilter).length > 0) {
      matchQuery.created_at = dateFilter;
    }

    const data = await SensorReading
      .find(matchQuery)
      .select('temperature humidity vibration motor_speed motor_current motor_power')
      .lean();

    console.log(`Retrieved ${data.length} records for correlation analysis`);

    if (data.length === 0) {
      return res.json({ 
        correlations: {}, 
        message: 'No data available for correlation analysis',
        timezone: 'UTC',
        queryDateRange: { startDate, endDate }
      });
    }

    // Calculate correlations with enhanced dataset
    const correlations = calculateEnhancedCorrelations(data);

    console.log('Enhanced correlation matrix calculated');
    res.json({ 
      correlations, 
      sampleSize: data.length,
      timezone: 'UTC',
      queryDateRange: { startDate, endDate }
    });

  } catch (error) {
    console.error('Error calculating enhanced correlations:', error);
    res.status(500).json({ 
      error: 'Failed to calculate correlations',
      details: error.message 
    });
  }
});

// FIXED: Analytics export endpoint - TIMEZONE FIXED
app.get('/api/analytics/export', async (req, res) => {
  try {
    const { startDate, endDate, format = 'csv' } = req.query;

    console.log('Export request:', { startDate, endDate, format });

    // FIXED: Build date filter with proper timezone handling
    const dateFilter = {};
    if (startDate) {
      const startDateUTC = new Date(startDate);
      dateFilter.$gte = startDateUTC;
      console.log('Export start date converted:', startDate, '→', startDateUTC.toISOString());
    }
    if (endDate) {
      const endDateUTC = new Date(endDate);
      dateFilter.$lte = endDateUTC;
      console.log('Export end date converted:', endDate, '→', endDateUTC.toISOString());
    }

    let query = {};
    if (Object.keys(dateFilter).length > 0) {
      query.created_at = dateFilter;
    }

    // Get data for export
    const data = await SensorReading
      .find(query)
      .sort({ created_at: -1 })
      .lean();

    if (format === 'csv') {
      // Convert to CSV with timezone-aware timestamps
      const csv = convertToCSVWithTimezone(data);
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="motor_data_${new Date().toISOString().split('T')[0]}.csv"`);
      res.send(csv);
    } else {
      // Return JSON with timezone information
      const exportData = {
        data: data,
        exportTimestamp: new Date().toISOString(),
        timezone: 'UTC',
        queryDateRange: { startDate, endDate },
        totalRecords: data.length,
        powerFormula: 'P = V × I (12V motor assumed)'
      };
      
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename="motor_data_${new Date().toISOString().split('T')[0]}.json"`);
      res.json(exportData);
    }

    console.log(`Data exported in ${format} format: ${data.length} records`);

  } catch (error) {
    console.error('Error exporting data:', error);
    res.status(500).json({ 
      error: 'Failed to export data',
      details: error.message 
    });
  }
});

// Helper functions for analytics

function getGroupByExpression(aggregation) {
  const timeFormats = {
    hourly: {
      year: { $year: '$created_at' },
      month: { $month: '$created_at' },
      day: { $dayOfMonth: '$created_at' },
      hour: { $hour: '$created_at' }
    },
    daily: {
      year: { $year: '$created_at' },
      month: { $month: '$created_at' },
      day: { $dayOfMonth: '$created_at' }
    },
    weekly: {
      year: { $year: '$created_at' },
      week: { $week: '$created_at' }
    }
  };

  return timeFormats[aggregation] || timeFormats.daily;
}

// Enhanced correlation calculation function
function calculateEnhancedCorrelations(data) {
  const fields = ['temperature', 'humidity', 'vibration', 'motor_speed', 'motor_current', 'motor_power'];
  const correlations = {};

  fields.forEach(field1 => {
    correlations[field1] = {};
    fields.forEach(field2 => {
      const values1 = data.map(d => d[field1]).filter(v => v != null && !isNaN(v));
      const values2 = data.map(d => d[field2]).filter(v => v != null && !isNaN(v));
      
      correlations[field1][field2] = calculatePearsonCorrelation(values1, values2);
    });
  });

  return correlations;
}

function calculatePearsonCorrelation(x, y) {
  if (x.length !== y.length || x.length === 0) return 0;

  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  return denominator === 0 ? 0 : numerator / denominator;
}

// FIXED: Enhanced CSV conversion with proper timezone handling
function convertToCSVWithTimezone(data) {
  if (!data || data.length === 0) return '';

  // Define headers including power calculation note
  const headers = [
    'created_at_utc', 'created_at_local', 'timestamp', 
    'temperature', 'humidity', 'vibration', 'vibration_x', 'vibration_y', 'vibration_z', 
    'temp_status', 'vib_status', 'motor_speed', 'motor_direction', 'motor_status', 
    'motor_enabled', 'is_transitioning', 'motor_current', 'motor_power_calculated', 
    'motor_voltage', 'power_formula', 'current_status', 'max_current', 'total_energy', 'device_id'
  ];
  
  const csvContent = [
    headers.join(','),
    `# Power calculation: P = V × I (12V motor voltage assumed)`,
    `# Export timestamp: ${new Date().toISOString()}`,
    `# Total records: ${data.length}`,
    `# Timezone: Data stored in UTC, local times calculated for display`,
    ...data.map(row => 
      headers.map(header => {
        let value = row[header];
        
        // Special handling for timestamps
        if (header === 'created_at_utc') {
          value = row.created_at ? new Date(row.created_at).toISOString() : '';
        } else if (header === 'created_at_local') {
          value = row.created_at ? new Date(row.created_at).toLocaleString() : '';
        } else if (header === 'motor_power_calculated') {
          value = row.motor_power || 0;
        }
        
        if (value === null || value === undefined) return '';
        if (typeof value === 'string' && value.includes(',')) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',')
    )
  ].join('\n');

  return csvContent;
}

// Serve the enhanced dashboard
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Serve the analytics dashboard
app.get('/analytics', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'analytics.html'));
});

// Start server
app.listen(PORT, () => {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║     ENHANCED MQTT MOTOR DASHBOARD              ║');
  console.log('║                                                ║');
  console.log('║  Student: Gourav Shaw (T0436800)               ║');
  console.log('║  Enhanced Motor Control with Timezone Fixes   ║');
  console.log('║                                                ║');
  console.log(`║  Server: http://localhost:${PORT}                 ║`);
  console.log(`║  Analytics: http://localhost:${PORT}/analytics    ║`);
  console.log('║  MQTT: mqtt://localhost:1883                    ║');
  console.log('║  MongoDB: mongodb://localhost:27017             ║');
  console.log('║                                                ║');
  console.log('║  Features:                                      ║');
  console.log('║  • Real-time sensor monitoring                ║');
  console.log('║  • Advanced motor control (BTS7960)           ║');
  console.log('║  • ACS712 current sensor monitoring           ║');
  console.log('║  • Professional analytics dashboard           ║');
  console.log('║  • Statistical analysis & correlations       ║');
  console.log('║  • Data export (CSV/JSON)                     ║');
  console.log('║  • Motor command logging                      ║');
  console.log('║  • Enhanced MQTT communication               ║');
  console.log('║  • FIXED: Timezone handling consistency      ║');
  console.log('║  • FIXED: Power calculation P = V × I        ║');
  console.log('║  • FIXED: Analytics controls alignment       ║');
  console.log('║  • Removed pressure monitoring               ║');
  console.log('╚════════════════════════════════════════════════╝');
});