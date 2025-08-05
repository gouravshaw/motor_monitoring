#!/usr/bin/env python3
"""
MongoDB Import Script for Motor Fault Dataset
Generated for: Gourav Shaw (T0436800)
Project: MSc Dissertation - Motor Fault Detection System
"""

import pymongo
import json
import os
from datetime import datetime

def import_to_mongodb():
    """Import the synthetic dataset to MongoDB"""
    
    print("🔄 Connecting to MongoDB...")
    try:
        # Connect to MongoDB (adjust connection string as needed)
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['motor_monitoring']
        collection = db['sensorreadings']
        
        print("📊 Loading dataset...")
        with open('motor_fault_dataset.json', 'r') as f:
            data = json.load(f)
        
        print(f"📥 Importing {len(data):,} records...")
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        # collection.delete_many({})
        # print("🗑️  Existing data cleared")
        
        # Insert new data
        result = collection.insert_many(data)
        
        print(f"✅ Successfully imported {len(result.inserted_ids):,} records")
        print(f"📊 Database: motor_monitoring")
        print(f"📊 Collection: sensorreadings")
        
        # Create indexes for better performance
        print("🔍 Creating database indexes...")
        collection.create_index([("created_at", -1)])
        collection.create_index([("device_id", 1)])
        collection.create_index([("temp_status", 1)])
        collection.create_index([("vib_status", 1)])
        collection.create_index([("fault_type", 1)])
        collection.create_index([("motor_speed", 1)])
        print("✅ Indexes created")
        
        # Display summary statistics
        print("\n📈 Dataset Summary:")
        total_records = collection.count_documents({})
        operating_records = collection.count_documents({"motor_speed": {"$gt": 0}})
        fault_records = collection.count_documents({"fault_type": {"$ne": "normal_operation"}})
        
        print(f"   Total records: {total_records:,}")
        print(f"   Operating records: {operating_records:,}")
        print(f"   Fault records: {fault_records:,}")
        print(f"   Fault percentage: {(fault_records/total_records)*100:.1f}%")
        
        print("\n🎯 Import completed successfully!")
        print("🌐 Your dashboard should now show the synthetic data")
        
    except Exception as e:
        print(f"❌ Error importing data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = import_to_mongodb()
    if success:
        print("\n🚀 Ready for ML training and validation!")
    else:
        print("\n❌ Import failed. Check MongoDB connection and try again.")
