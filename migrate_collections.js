const mongoose = require('mongoose');

// MongoDB connection
const MONGODB_URI = 'mongodb://localhost:27017/motor_monitoring';

async function migrateCollections() {
  try {
    console.log('🔄 Connecting to MongoDB...');
    await mongoose.connect(MONGODB_URI);
    console.log('✅ Connected to MongoDB');

    const db = mongoose.connection.db;
    
    // Check existing collections
    const collections = await db.listCollections().toArray();
    console.log('📊 Existing collections:', collections.map(c => c.name));

    // Remove duplicate collections if they exist
    const collectionNames = collections.map(c => c.name);
    
    if (collectionNames.includes('sensorReading')) {
      console.log('🔄 Dropping duplicate collection: sensorReading');
      await db.dropCollection('sensorReading');
    }
    
    if (collectionNames.includes('motorCommand')) {
      console.log('🔄 Dropping duplicate collection: motorCommand');
      await db.dropCollection('motorCommand');
    }

    // Enhanced indexes for better performance with current sensor data
    const sensorReadings = db.collection('sensorreadings');
    const motorCommands = db.collection('motorcommands');

    // Create enhanced indexes including current sensor fields
    console.log('🔄 Creating enhanced database indexes...');
    
    // Time-based indexes
    await sensorReadings.createIndex({ created_at: -1 }, { background: true });
    await sensorReadings.createIndex({ timestamp: -1 }, { background: true });
    
    // Device and status indexes
    await sensorReadings.createIndex({ device_id: 1 }, { background: true });
    await sensorReadings.createIndex({ temp_status: 1 }, { background: true });
    await sensorReadings.createIndex({ vib_status: 1 }, { background: true });
    
    // NEW: Current sensor indexes
    await sensorReadings.createIndex({ current_status: 1 }, { background: true });
    await sensorReadings.createIndex({ motor_current: -1 }, { background: true });
    await sensorReadings.createIndex({ motor_power: -1 }, { background: true });
    await sensorReadings.createIndex({ total_energy: -1 }, { background: true });
    
    // Motor control indexes
    await sensorReadings.createIndex({ motor_speed: 1 }, { background: true });
    await sensorReadings.createIndex({ motor_direction: 1 }, { background: true });
    await sensorReadings.createIndex({ motor_status: 1 }, { background: true });
    
    // Composite indexes for analytics
    await sensorReadings.createIndex({ 
      created_at: -1, 
      motor_speed: 1, 
      motor_current: 1 
    }, { background: true });
    
    await sensorReadings.createIndex({ 
      temp_status: 1, 
      vib_status: 1, 
      current_status: 1 
    }, { background: true });
    
    // Motor command indexes
    await motorCommands.createIndex({ executed_at: -1 }, { background: true });
    await motorCommands.createIndex({ command: 1 }, { background: true });
    await motorCommands.createIndex({ speed: 1 }, { background: true });
    await motorCommands.createIndex({ direction: 1 }, { background: true });

    console.log('✅ Enhanced database indexes created');

    // Count documents and analyze current sensor data
    const sensorCount = await sensorReadings.countDocuments();
    const commandCount = await motorCommands.countDocuments();
    
    // NEW: Analyze current sensor data migration
    const currentSensorCount = await sensorReadings.countDocuments({
      motor_current: { $exists: true }
    });
    
    const maxCurrent = await sensorReadings.findOne(
      { motor_current: { $exists: true } },
      { sort: { motor_current: -1 } }
    );
    
    const totalEnergyRecords = await sensorReadings.countDocuments({
      total_energy: { $exists: true, $gt: 0 }
    });
    
    console.log('📊 Enhanced database analysis:');
    console.log(`   - Total sensor readings: ${sensorCount}`);
    console.log(`   - Motor commands: ${commandCount}`);
    console.log(`   - Records with current data: ${currentSensorCount}`);
    console.log(`   - Peak current recorded: ${maxCurrent?.motor_current?.toFixed(2) || 'N/A'} A`);
    console.log(`   - Records with energy data: ${totalEnergyRecords}`);

    // Migration for existing records (add current sensor fields if missing)
    console.log('🔄 Migrating existing records to include current sensor fields...');
    
    const migrationResult = await sensorReadings.updateMany(
      { 
        motor_current: { $exists: false }
      },
      {
        $set: {
          motor_current: 0,
          motor_power: 0,
          current_status: 'IDLE',
          max_current: 0,
          total_energy: 0
        }
      }
    );
    
    console.log(`✅ Migration completed: ${migrationResult.modifiedCount} records updated with current sensor fields`);

    // Verify data integrity
    console.log('🔍 Verifying enhanced data integrity...');
    
    const sampleRecord = await sensorReadings.findOne({
      motor_current: { $exists: true },
      motor_power: { $exists: true },
      current_status: { $exists: true }
    });
    
    if (sampleRecord) {
      console.log('✅ Sample enhanced record found:');
      console.log('   - Temperature:', sampleRecord.temperature?.toFixed(1) || 'N/A', '°C');
      console.log('   - Vibration:', sampleRecord.vibration?.toFixed(3) || 'N/A', 'm/s²');
      console.log('   - Motor Speed:', sampleRecord.motor_speed || 'N/A', '%');
      console.log('   - Motor Current:', sampleRecord.motor_current?.toFixed(2) || 'N/A', 'A');
      console.log('   - Motor Power:', sampleRecord.motor_power?.toFixed(1) || 'N/A', 'W');
      console.log('   - Current Status:', sampleRecord.current_status || 'N/A');
      console.log('   - Total Energy:', sampleRecord.total_energy?.toFixed(3) || 'N/A', 'Ah');
    }

    // Performance optimization suggestions
    console.log('');
    console.log('🚀 Performance optimization recommendations:');
    console.log('   • Current sensor indexes created for fast queries');
    console.log('   • Composite indexes for analytics performance');
    console.log('   • Time-series optimization for real-time monitoring');
    console.log('   • Energy consumption tracking enabled');
    console.log('   • Fault detection indexes for quick alerts');

    console.log('');
    console.log('✅ Enhanced migration completed successfully!');
    console.log('⚡ ACS712 current sensor integration ready');
    console.log('📊 Advanced analytics with power monitoring enabled');
    
  } catch (error) {
    console.error('❌ Enhanced migration failed:', error);
    console.error('Stack trace:', error.stack);
  } finally {
    await mongoose.disconnect();
    console.log('🔌 Disconnected from MongoDB');
  }
}

// Run enhanced migration
migrateCollections();