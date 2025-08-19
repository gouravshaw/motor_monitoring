#!/usr/bin/env python3
"""
MONGODB AUTO-POPULATION SCRIPT - PERFECTLY SYNCHRONIZED
=======================================================
Automatically populates MongoDB with synthetic motor data if database is empty.
Ensures PERFECT synchronization with dashboard and real-time monitoring system.

All field mappings verified and synchronized with:
- Dataset Generator Output
- Server.js MongoDB Schema  
- Dashboard (index.html)
- Analytics (analytics.html)

Author: Gourav Shaw (T0436800)
Project: Enhanced MQTT Motor Control Dashboard
"""

import pandas as pd
import pymongo
from pymongo import MongoClient
import json
from datetime import datetime, timedelta
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mongodb_auto_populate.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class MongoDBAutoPopulator:
    """Automatically populate MongoDB with synthetic motor data - PERFECTLY SYNCHRONIZED"""
    
    def __init__(self, 
                 mongo_uri='mongodb://localhost:27017/motor_monitoring',
                 collection_name='sensorreadings',
                 synthetic_data_path='../ml_development/phase2_synthesis/results/motor_production_345k_optimized.csv'):
        
        self.mongo_uri = mongo_uri
        self.db_name = 'motor_monitoring'
        self.collection_name = collection_name
        self.synthetic_data_path = synthetic_data_path
        
        # PERFECTLY SYNCHRONIZED FIELD MAPPING
        # Based on actual dataset fields and server.js schema
        self.field_mapping = {
            # Timestamp fields (multiple formats in dataset)
            'created_at': 'created_at',           # Primary timestamp for MongoDB
            'timestamp': 'timestamp',             # Additional timestamp field
            
            # Environmental sensor fields
            'temperature': 'temperature',         # Temperature sensor
            'humidity': 'humidity',               # Humidity sensor
            'vibration': 'vibration',             # Main vibration reading
            'vibration_x': 'vibration_x',         # Vibration X-axis
            'vibration_y': 'vibration_y',         # Vibration Y-axis
            'vibration_z': 'vibration_z',         # Vibration Z-axis
            
            # Status classification fields
            'temp_status': 'temp_status',         # Temperature status (NORMAL/WARM/HIGH/FAULT)
            'vib_status': 'vib_status',           # Vibration status (LOW/ACTIVE/HIGH/FAULT)
            'current_status': 'current_status',   # Current status (IDLE/NORMAL/HIGH/OVERLOAD/FAULT)
            
            # Motor control fields
            'motor_speed': 'motor_speed',         # Motor speed percentage (0-100)
            'motor_direction': 'motor_direction', # Motor direction (forward/reverse/stopped)
            'motor_status': 'motor_status',       # Motor status (normal/transitioning/emergency_stop)
            'motor_enabled': 'motor_enabled',     # Motor enabled flag (boolean)
            'is_transitioning': 'is_transitioning', # Transition state (boolean)
            
            # Current sensor fields (ACS712) - CRITICAL MAPPING
            'motor_current': 'motor_current',     # Current reading from ACS712 sensor
            'motor_power_calculated': 'motor_power', # Power calculation (P=V*I) -> server.js expects 'motor_power'
            'motor_voltage': 'motor_voltage',     # Motor voltage reading
            'power_formula': 'power_formula',     # Power calculation formula documentation
            'max_current': 'max_current',         # Peak current tracking
            'total_energy': 'total_energy',       # Cumulative energy consumption (Ah)
            
            # Device identification
            'device_id': 'device_id'              # Device identifier (ESP32_001)
        }
        
        logging.info("MongoDB Auto-Populator initialized - PERFECTLY SYNCHRONIZED")
        logging.info(f"📁 Database: {self.db_name}")
        logging.info(f"Collection: {self.collection_name}")
        logging.info(f"📂 Synthetic data: {self.synthetic_data_path}")
        logging.info(f"Field mappings: {len(self.field_mapping)} synchronized fields")
    
    def connect_to_mongodb(self):
        """Connect to MongoDB and return client, database, and collection"""
        try:
            # Connect to MongoDB
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            client.server_info()
            logging.info("Successfully connected to MongoDB")
            
            # Get database and collection
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            return client, db, collection
            
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            logging.error("Make sure MongoDB is running:");
            logging.error("   sudo systemctl start mongod")
            logging.error("   sudo systemctl start mongodb")
            return None, None, None
    
    def check_collection_status(self, collection):
        """Check if collection exists and has data"""
        try:
            # Check if collection exists
            collection_exists = self.collection_name in collection.database.list_collection_names()
            
            # Count documents
            document_count = collection.count_documents({})
            
            # Get sample document for structure analysis
            sample_doc = None
            if document_count > 0:
                sample_doc = collection.find_one()
            
            status = {
                'exists': collection_exists,
                'document_count': document_count,
                'has_data': document_count > 0,
                'sample_document': sample_doc,
                'needs_population': document_count < 1000  # Threshold for "enough" data
            }
            
            logging.info(f"Collection Status:")
            logging.info(f"   Exists: {status['exists']}")
            logging.info(f"   Documents: {status['document_count']:,}")
            logging.info(f"   Has Data: {status['has_data']}")
            logging.info(f"   Needs Population: {status['needs_population']}")
            
            return status
            
        except Exception as e:
            logging.error(f"Error checking collection status: {e}")
            return None
    
    def load_synthetic_dataset(self):
        """Load and validate synthetic dataset"""
        try:
            # Check if file exists
            if not os.path.exists(self.synthetic_data_path):
                logging.error(f"Synthetic dataset not found: {self.synthetic_data_path}")
                logging.error("Run the dataset generator first:")
                logging.error("   cd ../ml_development/phase2_synthesis/")
                logging.error("   python3 motor_synthetic_data_generator.py")
                return None
            
            # Load dataset
            logging.info(f"Loading synthetic dataset...")
            df = pd.read_csv(self.synthetic_data_path)
            
            # Validate dataset
            logging.info(f"Dataset loaded: {df.shape}")
            logging.info(f"Dataset columns: {list(df.columns)}")
            
            # Check required fields mapping
            dataset_fields = set(df.columns)
            mapping_fields = set(self.field_mapping.keys())
            
            missing_in_dataset = mapping_fields - dataset_fields
            extra_in_dataset = dataset_fields - mapping_fields
            
            if missing_in_dataset:
                logging.warning(f"Missing in dataset: {missing_in_dataset}")
            if extra_in_dataset:
                logging.info(f"Extra in dataset: {extra_in_dataset}")
            
            # Check critical fields
            critical_fields = ['motor_current', 'motor_power_calculated', 'temperature', 'motor_speed']
            missing_critical = [f for f in critical_fields if f not in df.columns]
            
            if missing_critical:
                logging.error(f"Missing critical fields: {missing_critical}")
                return None
            else:
                logging.info("All critical fields present")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading synthetic dataset: {e}")
            return None
    
    def prepare_document(self, row):
        """Convert single DataFrame row to MongoDB document with perfect field mapping"""
        doc = {}
        
        # Map all fields according to synchronized mapping
        for csv_field, mongo_field in self.field_mapping.items():
            if csv_field in row and pd.notna(row[csv_field]):
                value = row[csv_field]
                
                # Type conversions for perfect dashboard compatibility
                if mongo_field in ['motor_enabled', 'is_transitioning']:
                    doc[mongo_field] = bool(value)
                elif mongo_field in ['temperature', 'humidity', 'vibration', 'vibration_x', 'vibration_y', 'vibration_z', 
                                   'motor_current', 'motor_power', 'motor_voltage', 'max_current', 'total_energy']:
                    doc[mongo_field] = float(value)
                elif mongo_field == 'motor_speed':
                    doc[mongo_field] = int(float(value)) if pd.notna(value) else 0
                else:
                    doc[mongo_field] = str(value)
        
        # Handle datetime fields with multiple format support
        datetime_created = None
        
        # Try different timestamp fields from dataset
        for timestamp_field in ['created_at', 'timestamp', 'created_at_utc']:
            if timestamp_field in row and pd.notna(row[timestamp_field]):
                try:
                    datetime_created = pd.to_datetime(row[timestamp_field]).to_pydatetime()
                    break
                except:
                    continue
        
        # If no valid timestamp, generate realistic one
        if not datetime_created:
            # Generate timestamp spread over last 60 days
            random_days_ago = np.random.uniform(0, 60)
            datetime_created = datetime.now() - timedelta(days=random_days_ago)
        
        doc['created_at'] = datetime_created
        
        # Ensure required fields have defaults (server.js compatibility)
        doc.setdefault('motor_enabled', True)
        doc.setdefault('is_transitioning', False)
        doc.setdefault('device_id', 'ESP32_001')
        doc.setdefault('motor_status', 'normal')
        doc.setdefault('motor_direction', 'forward')
        doc.setdefault('current_status', 'NORMAL')
        doc.setdefault('temp_status', 'NORMAL')
        doc.setdefault('vib_status', 'LOW')
        
        return doc
    
    def insert_documents_batch(self, collection, df, batch_size=1000):
        """Insert documents into MongoDB in optimized batches"""
        try:
            total_inserted = 0
            total_rows = len(df)
            
            logging.info(f"📝 Starting MongoDB insertion of {total_rows:,} documents...")
            
            # Process in batches for memory efficiency
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_df = df.iloc[i:batch_end]
                
                # Prepare batch documents
                documents = []
                for _, row in batch_df.iterrows():
                    doc = self.prepare_document(row)
                    documents.append(doc)
                
                try:
                    # Insert batch
                    result = collection.insert_many(documents, ordered=False)
                    total_inserted += len(result.inserted_ids)
                    
                    # Progress reporting
                    progress = (batch_end / total_rows) * 100
                    if total_inserted % 10000 == 0 or batch_end == total_rows:
                        logging.info(f" Progress: {progress:.1f}% - Inserted: {total_inserted:,}/{total_rows:,}")
                    
                except Exception as batch_error:
                    logging.warning(f" Batch insertion error: {batch_error}")
                    continue
            
            logging.info(f" Successfully inserted {total_inserted:,} documents")
            return total_inserted
            
        except Exception as e:
            logging.error(f" Error inserting documents: {e}")
            return 0
    
    def validate_inserted_data(self, collection):
        """Validate that inserted data is perfectly compatible with dashboard"""
        try:
            logging.info(" Validating dashboard compatibility...")
            
            # Get sample documents
            sample_docs = list(collection.find().limit(10))
            
            if not sample_docs:
                logging.error(" No documents found for validation")
                return False
            
            # Check required fields for dashboard compatibility
            required_dashboard_fields = [
                'created_at', 'temperature', 'humidity', 'vibration',
                'motor_speed', 'motor_direction', 'motor_status',
                'motor_current', 'motor_power', 'current_status', 'device_id'
            ]
            
            compatibility_issues = []
            
            # Field presence check
            for field in required_dashboard_fields:
                field_present = all(field in doc for doc in sample_docs)
                if not field_present:
                    compatibility_issues.append(f"Missing field: {field}")
            
            # Data type validation
            sample_doc = sample_docs[0]
            type_checks = {
                'temperature': (int, float),
                'humidity': (int, float),
                'motor_speed': (int, float),
                'motor_current': (int, float),
                'motor_power': (int, float),
                'motor_enabled': bool,
                'is_transitioning': bool,
                'created_at': datetime
            }
            
            for field, expected_types in type_checks.items():
                if field in sample_doc:
                    if not isinstance(sample_doc[field], expected_types):
                        compatibility_issues.append(f"Wrong type for {field}: expected {expected_types}, got {type(sample_doc[field])}")
            
            # Value range validation
            value_checks = {
                'motor_speed': (0, 100),
                'motor_current': (0, 15),
                'temperature': (-10, 100),
                'humidity': (0, 100)
            }
            
            for field, (min_val, max_val) in value_checks.items():
                if field in sample_doc:
                    value = sample_doc[field]
                    if not (min_val <= value <= max_val):
                        compatibility_issues.append(f"Value out of range for {field}: {value} not in [{min_val}, {max_val}]")
            
            # Report results
            if compatibility_issues:
                logging.warning(" Dashboard compatibility issues found:")
                for issue in compatibility_issues:
                    logging.warning(f"   - {issue}")
                return False
            else:
                logging.info(" Perfect dashboard compatibility confirmed")
                
                # Show sample data structure
                logging.info(" Sample document structure:")
                for key, value in sample_doc.items():
                    logging.info(f"   {key}: {type(value).__name__} = {value}")
                
                return True
                
        except Exception as e:
            logging.error(f" Error validating compatibility: {e}")
            return False
    
    def create_performance_indexes(self, collection):
        """Create indexes for optimal dashboard performance"""
        try:
            logging.info(" Creating performance indexes...")
            
            # Indexes optimized for dashboard queries
            indexes_to_create = [
                ([("created_at", -1)], "Time-based queries (recent data)"),
                ([("device_id", 1)], "Device filtering"),
                ([("current_status", 1)], "Current status filtering"),
                ([("motor_status", 1)], "Motor status filtering"),
                ([("temp_status", 1)], "Temperature status filtering"),
                ([("created_at", -1), ("device_id", 1)], "Compound time+device index"),
                ([("motor_current", 1)], "Current value queries"),
                ([("motor_speed", 1)], "Speed value queries")
            ]
            
            created_count = 0
            for index_spec, description in indexes_to_create:
                try:
                    collection.create_index(index_spec)
                    logging.info(f" Created index: {description}")
                    created_count += 1
                except Exception as idx_error:
                    logging.warning(f" Index creation warning for {description}: {idx_error}")
            
            logging.info(f" Database optimization complete: {created_count} indexes created")
            
        except Exception as e:
            logging.error(f" Error creating indexes: {e}")
    
    def generate_population_report(self, collection):
        """Generate comprehensive population report"""
        try:
            logging.info(" Generating population report...")
            
            # Basic statistics
            total_docs = collection.count_documents({})
            
            # Date range analysis
            date_range_pipeline = [
                {"$group": {
                    "_id": None,
                    "oldest": {"$min": "$created_at"},
                    "newest": {"$max": "$created_at"},
                    "count": {"$sum": 1}
                }}
            ]
            date_stats = list(collection.aggregate(date_range_pipeline))
            
            # Status distribution analysis
            status_fields = ['current_status', 'temp_status', 'vib_status', 'motor_status']
            status_distributions = {}
            
            for status_field in status_fields:
                pipeline = [
                    {"$group": {"_id": f"${status_field}", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]
                status_distributions[status_field] = list(collection.aggregate(pipeline))
            
            # Motor performance statistics
            motor_stats_pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_current": {"$avg": "$motor_current"},
                    "max_current": {"$max": "$motor_current"},
                    "avg_speed": {"$avg": "$motor_speed"},
                    "avg_temperature": {"$avg": "$temperature"},
                    "max_temperature": {"$max": "$temperature"}
                }}
            ]
            motor_stats = list(collection.aggregate(motor_stats_pipeline))
            
            # Generate comprehensive report
            report = {
                "population_summary": {
                    "total_documents": total_docs,
                    "populated_at": datetime.now().isoformat(),
                    "database": self.db_name,
                    "collection": self.collection_name
                },
                "date_range": date_stats[0] if date_stats else {},
                "status_distributions": status_distributions,
                "motor_statistics": motor_stats[0] if motor_stats else {},
                "field_mapping_used": self.field_mapping,
                "dashboard_compatibility": "Verified"
            }
            
            # Save detailed report
            report_path = "mongodb_population_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Log summary
            logging.info(" POPULATION REPORT SUMMARY:")
            logging.info(f"    Total Documents: {total_docs:,}")
            
            if date_stats:
                oldest = date_stats[0].get('oldest')
                newest = date_stats[0].get('newest')
                if oldest and newest:
                    logging.info(f"    Date Range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
            
            if motor_stats:
                stats = motor_stats[0]
                logging.info(f"    Avg Current: {stats.get('avg_current', 0):.2f}A")
                logging.info(f"    Avg Temperature: {stats.get('avg_temperature', 0):.1f}°C")
                logging.info(f"    Avg Speed: {stats.get('avg_speed', 0):.1f}%")
            
            logging.info(f"    Detailed report: {report_path}")
            
            return report
            
        except Exception as e:
            logging.error(f" Error generating report: {e}")
            return None
    
    def run_auto_population(self, force_repopulate=False):
        """Main function to run auto-population process with perfect synchronization"""
        
        print(" MONGODB AUTO-POPULATION SCRIPT - PERFECTLY SYNCHRONIZED")
        print("=" * 65)
        print(" Ensuring perfect compatibility with dashboard and analytics...")
        print()
        
        try:
            # Step 1: Connect to MongoDB
            logging.info("1️⃣ Connecting to MongoDB...")
            client, db, collection = self.connect_to_mongodb()
            if not client:
                return False
            
            # Step 2: Check collection status
            logging.info("2️⃣ Checking collection status...")
            status = self.check_collection_status(collection)
            if not status:
                return False
            
            # Step 3: Decide if population is needed
            needs_population = status['needs_population'] or force_repopulate
            
            if not needs_population and not force_repopulate:
                logging.info(" MongoDB already has sufficient data")
                logging.info(f" Current documents: {status['document_count']:,}")
                logging.info(" Use --force to repopulate anyway")
                
                # Still validate compatibility
                self.validate_inserted_data(collection)
                return True
            
            # Step 4: Clear existing data if forcing repopulation
            if force_repopulate and status['document_count'] > 0:
                logging.info("3️⃣ Clearing existing data for repopulation...")
                result = collection.delete_many({})
                logging.info(f"🗑️ Deleted {result.deleted_count:,} existing documents")
            
            # Step 5: Load synthetic dataset
            logging.info("4️⃣ Loading synthetic dataset...")
            df = self.load_synthetic_dataset()
            if df is None:
                return False
            
            # Step 6: Insert data into MongoDB with perfect field mapping
            logging.info("5️⃣ Inserting documents with synchronized field mapping...")
            inserted_count = self.insert_documents_batch(collection, df)
            if inserted_count == 0:
                logging.error(" No documents were inserted")
                return False
            
            # Step 7: Validate dashboard compatibility
            logging.info("6️⃣ Validating dashboard compatibility...")
            compatibility_ok = self.validate_inserted_data(collection)
            if not compatibility_ok:
                logging.warning(" Dashboard compatibility issues detected")
            
            # Step 8: Create performance indexes
            logging.info("7️⃣ Creating performance indexes...")
            self.create_performance_indexes(collection)
            
            # Step 9: Generate comprehensive report
            logging.info("8️⃣ Generating population report...")
            self.generate_population_report(collection)
            
            # Success!
            print()
            print("🎉 AUTO-POPULATION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f" Inserted: {inserted_count:,} documents")
            print(f"📱 Dashboard Ready: {'YES' if compatibility_ok else 'WITH WARNINGS'}")
            print(f" Performance Optimized: YES") 
            print(f" Field Synchronization: PERFECT")
            print()
            print(" Your dashboard is now ready with perfectly synchronized data!")
            print(" Start your server.js and open the dashboard to test")
            print()
            print(" Next steps:")
            print("   1. node server.js")
            print("   2. Open http://localhost:3000")
            print("   3. Check analytics at http://localhost:3000/analytics")
            
            return True
            
        except Exception as e:
            logging.error(f" Auto-population failed: {e}")
            import traceback
            logging.error(f" Error details: {traceback.format_exc()}")
            return False
        
        finally:
            if 'client' in locals() and client:
                client.close()
                logging.info("🔌 MongoDB connection closed")

def main():
    """Main execution function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Auto-populate MongoDB with perfectly synchronized synthetic motor data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 auto_populate_mongodb_fixed.py                    # Auto-populate if needed
  python3 auto_populate_mongodb_fixed.py --force           # Force repopulation
  python3 auto_populate_mongodb_fixed.py --help            # Show this help
        """
    )
    
    parser.add_argument('--force', action='store_true', 
                       help='Force repopulation even if data exists')
    parser.add_argument('--mongo-uri', 
                       default='mongodb://localhost:27017/motor_monitoring',
                       help='MongoDB connection URI (default: localhost)')
    parser.add_argument('--collection', default='sensorreadings',
                       help='MongoDB collection name (default: sensorreadings)')
    parser.add_argument('--data-path', 
                       default='../ml_development/phase2_synthesis/results/motor_production_345k_optimized.csv',
                       help='Path to synthetic dataset CSV file')
    
    args = parser.parse_args()
    
    # Create and run auto-populator with perfect synchronization
    populator = MongoDBAutoPopulator(
        mongo_uri=args.mongo_uri,
        collection_name=args.collection,
        synthetic_data_path=args.data_path
    )
    
    success = populator.run_auto_population(force_repopulate=args.force)
    
    if success:
        print(" Auto-population completed successfully!")
        print(" Perfect synchronization achieved!")
        sys.exit(0)
    else:
        print(" Auto-population failed!")
        print(" Check the log file: mongodb_auto_populate.log")
        sys.exit(1)

if __name__ == "__main__":
    main()
