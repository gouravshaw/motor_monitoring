# real_data_pattern_extractor.py
"""
MOTOR REAL DATA PATTERN EXTRACTOR
==================================
This script analyzes your real motor data to extract patterns for synthetic generation.
Perfect for beginners - lots of comments explaining what each part does!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class RealDataPatternExtractor:
    """
    This class analyzes your real motor data and extracts patterns.
    Think of it as a detective that learns how your motor normally behaves.
    """
    
    def __init__(self):
        # These will store all the patterns we discover
        self.patterns = {}
        self.sensor_stats = {}
        self.motor_behaviors = {}
        
    def analyze_motor_data(self, csv_file_path):
        """
        Main function that analyzes your real motor data
        
        Parameters:
        csv_file_path: path to your real motor data CSV file
        
        Returns:
        Dictionary containing all discovered patterns
        """
        
        print("🔍 ANALYZING YOUR REAL MOTOR DATA")
        print("=" * 50)
        
        # Step 1: Load and validate your data
        print("\n📊 Step 1: Loading your real data...")
        df = self.load_and_validate_data(csv_file_path)
        
        # Step 2: Analyze each sensor's behavior
        print("\n📈 Step 2: Analyzing sensor behaviors...")
        self.analyze_sensor_patterns(df)
        
        # Step 3: Understand motor operating modes
        print("\n⚙️ Step 3: Identifying motor operating modes...")
        self.analyze_operating_modes(df)
        
        # Step 4: Look for fault signatures
        print("\n⚠️ Step 4: Searching for fault patterns...")
        self.analyze_fault_signatures(df)
        
        # Step 5: Study time-based patterns
        print("\n⏰ Step 5: Analyzing time patterns...")
        self.analyze_temporal_patterns(df)
        
        # Step 6: Check motor physics
        print("\n🔧 Step 6: Validating motor physics...")
        self.validate_motor_physics(df)
        
        # Step 7: Create summary
        print("\n📋 Step 7: Creating analysis summary...")
        self.create_summary_report()
        
        print("\n✅ ANALYSIS COMPLETE!")
        return self.patterns
    
    def load_and_validate_data(self, csv_file_path):
        """Load your CSV file and check if it looks correct"""
        
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file_path)
            print(f"   ✅ Successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Show first few rows so you can verify it looks right
            print(f"   📋 First 3 rows preview:")
            print(df.head(3).to_string())
            
            # Check for required columns
            required_columns = ['timestamp', 'motor_current', 'temperature', 'motor_voltage']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"   ⚠️ Warning: Missing expected columns: {missing_columns}")
            else:
                print(f"   ✅ All essential columns found")
                
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"   📅 Data timespan: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            print(f"   💡 Make sure your file path is correct: {csv_file_path}")
            raise
    
    def analyze_sensor_patterns(self, df):
        """Analyze how each sensor behaves in your real data"""
        
        # List of sensor columns to analyze
        sensor_columns = [
            'motor_current', 'temperature', 'humidity', 'vibration',
            'vibration_x', 'vibration_y', 'vibration_z', 'motor_speed',
            'motor_power_calculated', 'motor_voltage'
        ]
        
        self.sensor_stats = {}
        
        for sensor in sensor_columns:
            if sensor in df.columns:
                data = df[sensor].dropna()  # Remove any missing values
                
                # Calculate statistical properties
                stats = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'q95': float(data.quantile(0.95)),
                    'count': int(len(data))
                }
                
                self.sensor_stats[sensor] = stats
                
                print(f"   📊 {sensor}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.1f}, {stats['max']:.1f}]")
        
        self.patterns['sensor_statistics'] = self.sensor_stats
        print(f"   ✅ Analyzed {len(self.sensor_stats)} sensors")
    
    def analyze_operating_modes(self, df):
        """Identify different ways your motor operates"""
        
        operating_modes = {}
        
        # Classify motor load based on current
        if 'motor_current' in df.columns:
            current_data = df['motor_current']
            
            # Define load categories based on your motor's current levels
            df['load_category'] = pd.cut(
                current_data,
                bins=[0, 2, 6, 12, 25],
                labels=['IDLE', 'LIGHT_LOAD', 'NORMAL_LOAD', 'HEAVY_LOAD'],
                include_lowest=True
            )
            
            load_distribution = df['load_category'].value_counts(normalize=True)
            operating_modes['load_distribution'] = load_distribution.to_dict()
            
            print(f"   ⚙️ Motor load distribution:")
            for load_type, percentage in load_distribution.items():
                print(f"      {load_type}: {percentage:.1%}")
        
        # Classify temperature conditions
        if 'temperature' in df.columns:
            temp_data = df['temperature']
            
            df['temp_category'] = pd.cut(
                temp_data,
                bins=[0, 35, 50, 65, 100],
                labels=['COLD', 'NORMAL', 'WARM', 'HOT'],
                include_lowest=True
            )
            
            temp_distribution = df['temp_category'].value_counts(normalize=True)
            operating_modes['temperature_distribution'] = temp_distribution.to_dict()
            
            print(f"   🌡️ Temperature distribution:")
            for temp_type, percentage in temp_distribution.items():
                print(f"      {temp_type}: {percentage:.1%}")
        
        self.patterns['operating_modes'] = operating_modes
        print(f"   ✅ Identified operating mode patterns")
    
    def analyze_fault_signatures(self, df):
        """Look for signs of faults in your real data"""
        
        fault_info = {}
        
        # Check status columns for faults
        status_columns = ['current_status', 'temp_status', 'motor_status', 'vib_status']
        
        total_faults = 0
        fault_types = {}
        
        for status_col in status_columns:
            if status_col in df.columns:
                # Count non-NORMAL statuses as faults
                if status_col == 'motor_status':
                    fault_count = (df[status_col] != 'normal').sum()
                else:
                    fault_count = (df[status_col] != 'NORMAL').sum()
                
                fault_rate = fault_count / len(df)
                fault_types[status_col] = {
                    'fault_count': int(fault_count),
                    'fault_rate': float(fault_rate)
                }
                total_faults += fault_count
                
                print(f"   ⚠️ {status_col}: {fault_count} faults ({fault_rate:.1%})")
        
        # Calculate overall fault rate
        overall_fault_rate = total_faults / (len(df) * len(fault_types)) if fault_types else 0
        
        fault_info = {
            'fault_types': fault_types,
            'total_fault_instances': int(total_faults),
            'overall_fault_rate': float(overall_fault_rate),
            'has_fault_data': len(fault_types) > 0
        }
        
        self.patterns['fault_signatures'] = fault_info
        
        print(f"   📊 Overall fault rate: {overall_fault_rate:.1%}")
        print(f"   ✅ Fault analysis complete")
    
    def analyze_temporal_patterns(self, df):
        """Study how your motor behaves over time"""
        
        temporal_info = {}
        
        if 'timestamp' in df.columns:
            # Calculate time intervals between measurements
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
            avg_interval = df['time_diff'].median()
            
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Study patterns by hour
            if 'motor_current' in df.columns:
                hourly_patterns = df.groupby('hour')['motor_current'].agg(['mean', 'std']).to_dict()
                temporal_info['hourly_current_patterns'] = hourly_patterns
            
            temporal_info['average_sampling_interval_minutes'] = float(avg_interval)
            temporal_info['total_time_span_hours'] = float((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
            
            print(f"   ⏰ Average sampling interval: {avg_interval:.1f} minutes")
            print(f"   📅 Total time span: {temporal_info['total_time_span_hours']:.1f} hours")
        
        self.patterns['temporal_patterns'] = temporal_info
        print(f"   ✅ Temporal analysis complete")
    
    def validate_motor_physics(self, df):
        """Check if your data follows expected motor physics"""
        
        physics_info = {}
        
        # HUSETOO 775 motor specifications (from your previous conversations)
        motor_specs = {
            'max_current_A': 23.13,
            'voltage_V': 12.0,
            'max_power_W': 277.56
        }
        
        physics_checks = {}
        
        # Check current is within motor limits
        if 'motor_current' in df.columns:
            max_observed_current = df['motor_current'].max()
            current_within_limits = max_observed_current <= motor_specs['max_current_A']
            
            physics_checks['current_within_limits'] = bool(current_within_limits)
            physics_checks['max_observed_current'] = float(max_observed_current)
            
            print(f"   🔌 Max observed current: {max_observed_current:.2f}A (limit: {motor_specs['max_current_A']}A)")
            print(f"   ✅ Current within limits: {current_within_limits}")
        
        # Check power calculation if available
        if 'motor_power_calculated' in df.columns and 'motor_current' in df.columns:
            # Power should approximately equal Voltage × Current
            calculated_power = df['motor_voltage'] * df['motor_current']
            power_correlation = df['motor_power_calculated'].corr(calculated_power)
            
            physics_checks['power_correlation'] = float(power_correlation)
            
            print(f"   ⚡ Power calculation correlation: {power_correlation:.3f}")
        
        physics_info = {
            'motor_specifications': motor_specs,
            'physics_checks': physics_checks
        }
        
        self.patterns['motor_physics'] = physics_info
        print(f"   ✅ Physics validation complete")
    
    def create_summary_report(self):
        """Create a summary of everything we learned about your motor"""
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_quality_score': self.calculate_data_quality_score(),
            'recommendations': self.generate_recommendations()
        }
        
        self.patterns['summary'] = summary
        
        print(f"   📊 Data Quality Score: {summary['data_quality_score']:.1f}/10")
        print(f"   💡 Recommendations generated")
    
    def calculate_data_quality_score(self):
        """Calculate how good your data is for ML training (0-10 scale)"""
        
        score = 0
        
        # Sensor coverage (0-3 points)
        sensor_count = len(self.sensor_stats)
        score += min(sensor_count / 8, 1) * 3
        
        # Fault data availability (0-2 points)
        if self.patterns.get('fault_signatures', {}).get('has_fault_data', False):
            score += 2
        
        # Time span coverage (0-2 points)
        temporal = self.patterns.get('temporal_patterns', {})
        if temporal.get('total_time_span_hours', 0) > 24:  # At least 1 day
            score += 2
        
        # Physics compliance (0-2 points)
        physics = self.patterns.get('motor_physics', {}).get('physics_checks', {})
        if physics.get('current_within_limits', False):
            score += 2
        
        # Operating diversity (0-1 point)
        operating_modes = self.patterns.get('operating_modes', {}).get('load_distribution', {})
        if len(operating_modes) >= 3:  # At least 3 different load levels
            score += 1
        
        return min(score, 10)
    
    def generate_recommendations(self):
        """Generate recommendations for synthetic data generation"""
        
        recommendations = []
        
        # Data quality recommendations
        quality_score = self.calculate_data_quality_score()
        if quality_score >= 8:
            recommendations.append("Excellent data quality - ready for advanced synthetic generation")
        elif quality_score >= 6:
            recommendations.append("Good data quality - suitable for synthetic generation with minor enhancements")
        else:
            recommendations.append("Data quality needs improvement - consider collecting more diverse data")
        
        # Fault data recommendations
        fault_rate = self.patterns.get('fault_signatures', {}).get('overall_fault_rate', 0)
        if fault_rate < 0.01:  # Less than 1% faults
            recommendations.append("Low fault rate detected - will generate more fault scenarios for balanced training")
        elif fault_rate > 0.1:  # More than 10% faults
            recommendations.append("High fault rate detected - will ensure normal operation is well represented")
        
        # Synthetic dataset size recommendation
        data_size = len(self.sensor_stats.get('motor_current', {}).get('count', 0))
        if data_size < 1000:
            recommendations.append("Recommend generating 50,000+ synthetic samples for robust ML training")
        else:
            recommendations.append("Recommend generating 100,000+ synthetic samples for production-grade ML")
        
        return recommendations
    
    def save_patterns(self, output_file='results/motor_patterns.json'):
        """Save all discovered patterns to a file for the synthetic generator"""
        
        # Convert any numpy types to Python types for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: clean_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            return obj
        
        clean_patterns = clean_for_json(self.patterns)
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(clean_patterns, f, indent=2, default=str)
        
        print(f"\n💾 Patterns saved to: {output_file}")
        return output_file
    
    def create_visualization_report(self):
        """Create charts showing your data patterns (optional but helpful!)"""
        
        print(f"\n📊 Creating visualization report...")
        
        # This is optional - creates charts if you want to see your data patterns
        # We'll implement this in the next step if you're interested
        
        pass

# Example usage function
def analyze_my_motor_data(csv_file_path):
    """
    Simple function to analyze your motor data
    Just call this with your CSV file path!
    """
    
    print("🚀 STARTING MOTOR DATA ANALYSIS")
    print("=" * 60)
    
    # Create the analyzer
    analyzer = RealDataPatternExtractor()
    
    # Analyze your data
    patterns = analyzer.analyze_motor_data(csv_file_path)
    
    # Save the patterns for synthetic generation
    patterns_file = analyzer.save_patterns()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📋 Discovered {len(patterns)} pattern categories")
    print(f"💾 Patterns saved for synthetic generation")
    print(f"📁 Next step: Use these patterns to generate synthetic data")
    
    return patterns, patterns_file

# Run this if the script is executed directly
if __name__ == "__main__":
    # Replace this with your actual data file path
    csv_file_path = "real_data/motor_real_data_original.csv"
    
    try:
        patterns, patterns_file = analyze_my_motor_data(csv_file_path)
        print(f"\nSuccess! Ready for synthetic data generation.")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Make sure your data file exists at: {csv_file_path}")
