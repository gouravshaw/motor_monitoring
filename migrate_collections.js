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

    // Create indexes for better performance
    const sensorReadings = db.collection('sensorreadings');
    const motorCommands = db.collection('motorcommands');

    // Create indexes
    await sensorReadings.createIndex({ created_at: -1 }, { background: true });
    await sensorReadings.createIndex({ device_id: 1 }, { background: true });
    await sensorReadings.createIndex({ temp_status: 1 }, { background: true });
    await sensorReadings.createIndex({ vib_status: 1 }, { background: true });
    
    await motorCommands.createIndex({ executed_at: -1 }, { background: true });
    await motorCommands.createIndex({ command: 1 }, { background: true });

    console.log('✅ Database indexes created');

    // Count documents
    const sensorCount = await sensorReadings.countDocuments();
    const commandCount = await motorCommands.countDocuments();
    
    console.log('📊 Current data:');
    console.log(`   - Sensor readings: ${sensorCount}`);
    console.log(`   - Motor commands: ${commandCount}`);

    console.log('✅ Migration completed successfully!');
    
  } catch (error) {
    console.error('❌ Migration failed:', error);
  } finally {
    await mongoose.disconnect();
    console.log('🔌 Disconnected from MongoDB');
  }
}

// Run migration
migrateCollections();