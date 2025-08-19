# enhanced_comprehensive_dataset_validation.py
"""
ENHANCED COMPREHENSIVE SYNTHETIC DATA VALIDATION FRAMEWORK
=========================================================
Complete validation suite for synthetic motor fault detection data including:
- Statistical distribution validation
- Visual comparison analysis  
- Discriminative testing (Real vs Synthetic classifier)
- Domain-specific motor physics validation
- Dashboard compatibility validation
- Optimized threshold consistency validation
- Fault progression realism validation
- Utility testing (Train Synthetic Test Real)
- Quality metrics and comprehensive reporting

FIXED: Updated file paths for motor_monitoring project structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import json
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print("ENHANCED COMPREHENSIVE SYNTHETIC DATA VALIDATION FRAMEWORK")
print("=" * 70)
print("Target: Validate synthetic motor data for optimized dashboard compatibility")
print("Scope: Statistical, Visual, Discriminative, Domain, Dashboard, Utility testing")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

class EnhancedComprehensiveDataValidator:
    """Complete enhanced validation framework for synthetic motor fault detection data"""
    
    def __init__(self, output_dir='results/enhanced_validation'):
        self.output_dir = output_dir
        self.validation_results = {}
        self.plots_created = []
        
        # ALIGNED WITH DATASET GENERATOR - Optimized thresholds matching generator and dashboard
        self.OPTIMIZED_THRESHOLDS = {
            'current': {
                'IDLE': {'min': 0.0, 'max': 0.4},
                'NORMAL': {'min': 0.4, 'max': 4.5},
                'HIGH': {'min': 4.5, 'max': 6.0},
                'OVERLOAD': {'min': 6.0, 'max': 7.5},
                'FAULT': {'min': 7.5, 'max': 11.0}
            },
            'temperature': {
                'NORMAL': {'min': 10, 'max': 35},
                'WARM': {'min': 35, 'max': 42},
                'HIGH': {'min': 42, 'max': 50},
                'FAULT': {'min': 50, 'max': 85}
            },
            'vibration': {
                'LOW': {'min': 0.0, 'max': 3.0},
                'ACTIVE': {'min': 3.0, 'max': 8.0},
                'HIGH': {'min': 8.0, 'max': 12.0},
                'FAULT': {'min': 12.0, 'max': 25.0}
            }
        }
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
        print("Enhanced comprehensive validator initialized")
        print(f"📁 Output directory: {output_dir}")
        print("Optimized thresholds loaded for dashboard compatibility")
    
    def load_datasets(self, real_data_path: str, synthetic_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load real and synthetic datasets for comparison"""
        
        print(f"\nLOADING DATASETS FOR ENHANCED VALIDATION")
        print("-" * 50)
        
        try:
            # Load real data
            real_df = pd.read_csv(real_data_path)
            print(f"✅ Real data loaded: {real_df.shape}")
            
            # Load synthetic data
            synthetic_df = pd.read_csv(synthetic_data_path)
            print(f"✅ Synthetic data loaded: {synthetic_df.shape}")
            
            # Basic compatibility checks
            common_columns = set(real_df.columns) & set(synthetic_df.columns)
            print(f"📋 Common columns: {len(common_columns)}")
            
            if len(common_columns) < 5:
                print("⚠️ Warning: Few common columns between datasets")
            
            return real_df, synthetic_df
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return None, None
    
    def classify_with_optimized_thresholds(self, value: float, sensor_type: str) -> str:
        """Apply the exact same logic as dashboard and generator"""
        if sensor_type not in self.OPTIMIZED_THRESHOLDS:
            return 'UNKNOWN'
            
        thresholds = self.OPTIMIZED_THRESHOLDS[sensor_type]
        for status, limits in thresholds.items():
            if limits['min'] <= value <= limits['max']:
                return status
        return 'FAULT'
    
    def get_optimized_current_status(self, current: float) -> str:
        """Get current status using optimized thresholds"""
        return self.classify_with_optimized_thresholds(current, 'current')
    
    def get_optimized_temp_status(self, temperature: float) -> str:
        """Get temperature status using optimized thresholds"""
        return self.classify_with_optimized_thresholds(temperature, 'temperature')
    
    def get_optimized_vib_status(self, vibration: float) -> str:
        """Get vibration status using optimized thresholds"""
        return self.classify_with_optimized_thresholds(vibration, 'vibration')
    
    def validate_threshold_consistency(self, synthetic_df: pd.DataFrame) -> Dict:
        """Validate that synthetic data uses the exact same threshold logic as dashboard"""
        
        print(f"\n🎯 THRESHOLD CONSISTENCY VALIDATION")
        print("-" * 40)
        
        threshold_results = {
            'current_consistency': 0.0,
            'temperature_consistency': 0.0,
            'vibration_consistency': 0.0,
            'overall_consistency': 0.0,
            'threshold_errors': []
        }
        
        # Sample data for testing (use smaller sample for performance)
        sample_size = min(2000, len(synthetic_df))
        sample_data = synthetic_df.sample(sample_size, random_state=42)
        
        current_matches = 0
        temp_matches = 0
        vib_matches = 0
        
        for idx, row in sample_data.iterrows():
            # Test current status classification
            if 'motor_current' in row and 'current_status' in row:
                expected_current = self.get_optimized_current_status(row['motor_current'])
                actual_current = row['current_status']
                if expected_current == actual_current:
                    current_matches += 1
                else:
                    threshold_results['threshold_errors'].append({
                        'type': 'current',
                        'value': row['motor_current'],
                        'expected': expected_current,
                        'actual': actual_current
                    })
            
            # Test temperature status classification
            if 'temperature' in row and 'temp_status' in row:
                expected_temp = self.get_optimized_temp_status(row['temperature'])
                actual_temp = row['temp_status']
                if expected_temp == actual_temp:
                    temp_matches += 1
                else:
                    threshold_results['threshold_errors'].append({
                        'type': 'temperature',
                        'value': row['temperature'],
                        'expected': expected_temp,
                        'actual': actual_temp
                    })
            
            # Test vibration status classification
            if 'vibration' in row and 'vib_status' in row:
                expected_vib = self.get_optimized_vib_status(row['vibration'])
                actual_vib = row['vib_status']
                if expected_vib == actual_vib:
                    vib_matches += 1
                else:
                    threshold_results['threshold_errors'].append({
                        'type': 'vibration',
                        'value': row['vibration'],
                        'expected': expected_vib,
                        'actual': actual_vib
                    })
        
        # Calculate consistency percentages
        threshold_results['current_consistency'] = current_matches / sample_size
        threshold_results['temperature_consistency'] = temp_matches / sample_size
        threshold_results['vibration_consistency'] = vib_matches / sample_size
        threshold_results['overall_consistency'] = (current_matches + temp_matches + vib_matches) / (sample_size * 3)
        
        print(f"⚡ Current threshold consistency: {threshold_results['current_consistency']:.3f}")
        print(f"🌡️ Temperature threshold consistency: {threshold_results['temperature_consistency']:.3f}")
        print(f"📳 Vibration threshold consistency: {threshold_results['vibration_consistency']:.3f}")
        print(f"🎯 Overall threshold consistency: {threshold_results['overall_consistency']:.3f}")
        
        # Limit error reporting
        if len(threshold_results['threshold_errors']) > 10:
            threshold_results['threshold_errors'] = threshold_results['threshold_errors'][:10]
            threshold_results['error_note'] = f"Showing first 10 errors out of {len(threshold_results['threshold_errors'])} total"
        
        return threshold_results
    
    def validate_dashboard_compatibility(self, synthetic_df: pd.DataFrame) -> Dict:
        """Ensure synthetic data works perfectly with optimized dashboard"""
        
        print(f"\n📱 DASHBOARD COMPATIBILITY VALIDATION")
        print("-" * 45)
        
        compatibility_results = {}
        
        # 1. Check required columns exist - ALIGNED WITH DATASET GENERATOR
        required_columns = [
            'created_at', 'temperature', 'humidity', 'vibration',
            'motor_current', 'motor_power_calculated', 'motor_voltage',
            'current_status', 'temp_status', 'vib_status',
            'motor_speed', 'motor_direction', 'device_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in synthetic_df.columns]
        compatibility_results['missing_columns'] = missing_columns
        compatibility_results['column_completeness'] = (len(required_columns) - len(missing_columns)) / len(required_columns)
        
        print(f"📋 Column completeness: {compatibility_results['column_completeness']:.3f}")
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
        
        # 2. Validate data types and formats
        format_issues = []
        
        # Check datetime format
        if 'created_at' in synthetic_df.columns:
            try:
                pd.to_datetime(synthetic_df['created_at'].iloc[0])
                datetime_valid = True
            except:
                datetime_valid = False
                format_issues.append("Invalid datetime format in 'created_at'")
        
        # Check numeric ranges - ALIGNED WITH DATASET GENERATOR
        numeric_checks = {
            'motor_current': (0, 15),
            'temperature': (10, 90),
            'vibration': (0, 30),
            'motor_speed': (0, 100),
            'motor_voltage': (10, 15)
        }
        
        range_validations = {}
        for col, (min_val, max_val) in numeric_checks.items():
            if col in synthetic_df.columns:
                within_range = ((synthetic_df[col] >= min_val) & (synthetic_df[col] <= max_val)).mean()
                range_validations[col] = within_range
                if within_range < 0.95:
                    format_issues.append(f"{col} values outside expected range ({min_val}-{max_val})")
        
        compatibility_results['format_issues'] = format_issues
        compatibility_results['range_validations'] = range_validations
        compatibility_results['datetime_valid'] = datetime_valid
        
        # 3. Check status value consistency - ALIGNED WITH DATASET GENERATOR
        status_columns = {
            'current_status': ['IDLE', 'NORMAL', 'HIGH', 'OVERLOAD', 'FAULT'],
            'temp_status': ['NORMAL', 'WARM', 'HIGH', 'FAULT'],
            'vib_status': ['LOW', 'ACTIVE', 'HIGH', 'FAULT'],
            'motor_direction': ['forward', 'reverse', 'stopped']
        }
        
        status_validations = {}
        for col, valid_values in status_columns.items():
            if col in synthetic_df.columns:
                valid_percentage = synthetic_df[col].isin(valid_values).mean()
                status_validations[col] = valid_percentage
                if valid_percentage < 0.95:
                    format_issues.append(f"Invalid values in {col}")
        
        compatibility_results['status_validations'] = status_validations
        
        # 4. Overall dashboard compatibility score
        compatibility_score = (
            compatibility_results['column_completeness'] * 0.3 +
            (1 - len(format_issues) / 10) * 0.3 +  # Penalize format issues
            np.mean(list(range_validations.values())) * 0.2 +
            np.mean(list(status_validations.values())) * 0.2
        )
        
        compatibility_results['overall_compatibility'] = max(0, compatibility_score)
        compatibility_results['dashboard_ready'] = compatibility_score > 0.9
        
        print(f"📱 Dashboard compatibility score: {compatibility_score:.3f}")
        print(f"✅ Dashboard ready: {compatibility_results['dashboard_ready']}")
        
        return compatibility_results
    
    def validate_fault_progressions(self, synthetic_df: pd.DataFrame) -> Dict:
        """Validate that fault progressions are realistic"""
        
        print(f"\n📈 FAULT PROGRESSION VALIDATION")
        print("-" * 35)
        
        progression_results = {}
        
        try:
            # Sort by timestamp to check progressions
            df_sorted = synthetic_df.sort_values('created_at').copy()
            
            # Check that faults don't appear/disappear randomly
            fault_transitions = 0
            unrealistic_jumps = 0
            abrupt_recoveries = 0
            
            for i in range(1, min(len(df_sorted), 10000)):  # Limit for performance
                prev_status = df_sorted.iloc[i-1]['current_status']
                curr_status = df_sorted.iloc[i]['current_status']
                
                if prev_status != curr_status:
                    fault_transitions += 1
                    
                    # Check for unrealistic jumps (NORMAL directly to FAULT)
                    if prev_status == 'NORMAL' and curr_status in ['OVERLOAD', 'FAULT']:
                        unrealistic_jumps += 1
                    
                    # Check for abrupt recoveries (FAULT directly to NORMAL)
                    if prev_status in ['FAULT', 'OVERLOAD'] and curr_status == 'NORMAL':
                        abrupt_recoveries += 1
            
            # Analyze fault duration patterns
            fault_episodes = []
            current_episode = {'start': None, 'duration': 0, 'severity': None}
            
            for i, status in enumerate(df_sorted['current_status'][:5000]):  # Limit for performance
                if status in ['HIGH', 'OVERLOAD', 'FAULT']:
                    if current_episode['start'] is None:
                        current_episode['start'] = i
                        current_episode['severity'] = status
                    current_episode['duration'] += 1
                else:
                    if current_episode['start'] is not None:
                        fault_episodes.append(current_episode.copy())
                        current_episode = {'start': None, 'duration': 0, 'severity': None}
            
            # Calculate progression quality metrics
            progression_quality = 1.0
            if fault_transitions > 0:
                progression_quality = max(0, 1 - (unrealistic_jumps + abrupt_recoveries) / fault_transitions)
            
            progression_results = {
                'total_transitions': fault_transitions,
                'unrealistic_jumps': unrealistic_jumps,
                'abrupt_recoveries': abrupt_recoveries,
                'fault_episodes': len(fault_episodes),
                'avg_fault_duration': np.mean([ep['duration'] for ep in fault_episodes]) if fault_episodes else 0,
                'progression_quality': progression_quality,
                'realistic_progression': progression_quality > 0.8
            }
            
            print(f"🔄 Fault transitions: {fault_transitions}")
            print(f"⚠️ Unrealistic jumps: {unrealistic_jumps}")
            print(f"📈 Progression quality: {progression_quality:.3f}")
            print(f"✅ Realistic progression: {progression_results['realistic_progression']}")
            
        except Exception as e:
            print(f"❌ Error in fault progression validation: {e}")
            progression_results = {'error': str(e)}
        
        return progression_results
    
    def validate_operating_modes(self, synthetic_df: pd.DataFrame) -> Dict:
        """Validate realistic distribution of operating modes"""
        
        print(f"\n⚙️ OPERATING MODE VALIDATION")
        print("-" * 35)
        
        mode_results = {}
        
        try:
            # Classify operating modes based on current levels - ALIGNED WITH DATASET GENERATOR
            modes = []
            for current in synthetic_df['motor_current']:
                if current <= 0.4:
                    modes.append('IDLE')
                elif current <= 2.0:
                    modes.append('LIGHT')
                elif current <= 4.5:
                    modes.append('NORMAL')
                elif current <= 6.0:
                    modes.append('HEAVY')
                else:
                    modes.append('OVERLOAD')
            
            mode_distribution = pd.Series(modes).value_counts(normalize=True)
            
            # Expected realistic distribution (based on typical industrial operation)
            expected_distribution = {
                'IDLE': 0.15,      # 15% idle time
                'LIGHT': 0.25,     # 25% light load
                'NORMAL': 0.45,    # 45% normal operation
                'HEAVY': 0.12,     # 12% heavy load
                'OVERLOAD': 0.03   # 3% overload conditions
            }
            
            # Calculate distribution similarity
            distribution_differences = []
            for mode, expected in expected_distribution.items():
                actual = mode_distribution.get(mode, 0)
                difference = abs(actual - expected)
                distribution_differences.append(difference)
            
            distribution_similarity = max(0, 1 - sum(distribution_differences) / 2)
            
            # Time-based pattern analysis
            df_with_modes = synthetic_df.copy()
            df_with_modes['operating_mode'] = modes
            df_with_modes['hour'] = pd.to_datetime(df_with_modes['created_at']).dt.hour
            
            # Check for realistic day/night patterns
            day_hours = df_with_modes[df_with_modes['hour'].between(6, 18)]
            night_hours = df_with_modes[~df_with_modes['hour'].between(6, 18)]
            
            day_activity = (day_hours['operating_mode'] != 'IDLE').mean()
            night_activity = (night_hours['operating_mode'] != 'IDLE').mean()
            
            day_night_pattern_realistic = day_activity > night_activity  # More activity during day
            
            mode_results = {
                'actual_distribution': mode_distribution.to_dict(),
                'expected_distribution': expected_distribution,
                'distribution_similarity': distribution_similarity,
                'distribution_realistic': distribution_similarity > 0.8,
                'day_activity_rate': day_activity,
                'night_activity_rate': night_activity,
                'day_night_pattern_realistic': day_night_pattern_realistic,
                'overall_mode_quality': (distribution_similarity + (0.1 if day_night_pattern_realistic else 0)) / 1.1
            }
            
            print(f"📊 Distribution similarity: {distribution_similarity:.3f}")
            print(f"🌅 Day activity rate: {day_activity:.3f}")
            print(f"🌙 Night activity rate: {night_activity:.3f}")
            print(f"✅ Realistic patterns: {mode_results['distribution_realistic']}")
            
        except Exception as e:
            print(f"❌ Error in operating mode validation: {e}")
            mode_results = {'error': str(e)}
        
        return mode_results
    
    def enhanced_motor_physics_validation(self, synthetic_df: pd.DataFrame) -> Dict:
        """Enhanced motor physics validation specific to HUSETOO 775 motor"""
        
        print(f"\n🔧 ENHANCED MOTOR PHYSICS VALIDATION")
        print("-" * 45)
        
        enhanced_results = {}
        
        try:
            # 1. Current-Speed relationship validation
            if all(col in synthetic_df.columns for col in ['motor_current', 'motor_speed']):
                # Filter out idle conditions for better correlation
                active_data = synthetic_df[synthetic_df['motor_speed'] > 10]
                if len(active_data) > 100:
                    speed_current_corr = np.corrcoef(active_data['motor_speed'], active_data['motor_current'])[0, 1]
                else:
                    speed_current_corr = 0
                
                enhanced_results['speed_current_physics'] = {
                    'correlation': speed_current_corr,
                    'realistic': speed_current_corr > 0.3,
                    'status': '✅ PASS' if speed_current_corr > 0.3 else '❌ FAIL'
                }
                print(f"⚡ Speed-Current correlation: {speed_current_corr:.3f} {enhanced_results['speed_current_physics']['status']}")
            
            # 2. Voltage stability validation (12V motor should stay ~11.5-12.5V)
            if 'motor_voltage' in synthetic_df.columns:
                voltage_stability = (
                    (synthetic_df['motor_voltage'] >= 11.0) & 
                    (synthetic_df['motor_voltage'] <= 13.0)
                ).mean()
                
                voltage_mean = synthetic_df['motor_voltage'].mean()
                voltage_std = synthetic_df['motor_voltage'].std()
                
                enhanced_results['voltage_stability'] = {
                    'within_range_percentage': voltage_stability,
                    'mean_voltage': voltage_mean,
                    'voltage_std': voltage_std,
                    'stable': voltage_stability > 0.95 and 11.5 <= voltage_mean <= 12.5,
                    'status': '✅ PASS' if voltage_stability > 0.95 and 11.5 <= voltage_mean <= 12.5 else '❌ FAIL'
                }
                print(f"🔋 Voltage stability: {voltage_stability:.3f} (μ={voltage_mean:.1f}V) {enhanced_results['voltage_stability']['status']}")
            
            # 3. Idle current validation
            idle_samples = synthetic_df[synthetic_df['motor_speed'] < 5]
            if len(idle_samples) > 0:
                idle_current_realistic = (idle_samples['motor_current'] < 0.5).mean()
                mean_idle_current = idle_samples['motor_current'].mean()
                
                enhanced_results['idle_current_physics'] = {
                    'realistic_idle_percentage': idle_current_realistic,
                    'mean_idle_current': mean_idle_current,
                    'realistic': idle_current_realistic > 0.8 and mean_idle_current < 0.3,
                    'status': '✅ PASS' if idle_current_realistic > 0.8 and mean_idle_current < 0.3 else '❌ FAIL'
                }
                print(f"💤 Idle current physics: {idle_current_realistic:.3f} (μ={mean_idle_current:.2f}A) {enhanced_results['idle_current_physics']['status']}")
            
            # 4. Power efficiency validation
            if all(col in synthetic_df.columns for col in ['motor_speed', 'motor_power_calculated']):
                active_samples = synthetic_df[(synthetic_df['motor_speed'] > 20) & (synthetic_df['motor_power_calculated'] > 1)]
                if len(active_samples) > 100:
                    efficiency_values = active_samples['motor_speed'] / active_samples['motor_power_calculated']
                    efficiency_mean = efficiency_values.mean()
                    efficiency_realistic = (efficiency_values > 0.5).mean()  # Reasonable efficiency range
                    
                    enhanced_results['power_efficiency'] = {
                        'mean_efficiency': efficiency_mean,
                        'efficiency_realistic_percentage': efficiency_realistic,
                        'realistic': efficiency_realistic > 0.7,
                        'status': '✅ PASS' if efficiency_realistic > 0.7 else '❌ FAIL'
                    }
                    print(f"⚙️ Power efficiency: {efficiency_realistic:.3f} (μ={efficiency_mean:.2f}) {enhanced_results['power_efficiency']['status']}")
            
            # 5. Temperature rise correlation with load
            if all(col in synthetic_df.columns for col in ['motor_current', 'temperature']):
                # Calculate temperature rise from baseline
                baseline_temp = synthetic_df[synthetic_df['motor_current'] < 0.5]['temperature'].mean()
                temp_rise = synthetic_df['temperature'] - baseline_temp
                current_temp_corr = np.corrcoef(synthetic_df['motor_current'], temp_rise)[0, 1]
                
                enhanced_results['temperature_load_physics'] = {
                    'baseline_temperature': baseline_temp,
                    'current_temp_rise_correlation': current_temp_corr,
                    'realistic': current_temp_corr > 0.4,  # Should be strong positive correlation
                    'status': '✅ PASS' if current_temp_corr > 0.4 else '❌ FAIL'
                }
                print(f"🌡️ Temp-Load correlation: {current_temp_corr:.3f} {enhanced_results['temperature_load_physics']['status']}")
            
            # 6. Vibration patterns validation
            if all(col in synthetic_df.columns for col in ['motor_speed', 'vibration', 'motor_current']):
                # Vibration should increase with speed and load
                speed_vib_corr = np.corrcoef(synthetic_df['motor_speed'], synthetic_df['vibration'])[0, 1]
                current_vib_corr = np.corrcoef(synthetic_df['motor_current'], synthetic_df['vibration'])[0, 1]
                
                enhanced_results['vibration_physics'] = {
                    'speed_vibration_correlation': speed_vib_corr,
                    'current_vibration_correlation': current_vib_corr,
                    'realistic': speed_vib_corr > 0.2 and current_vib_corr > 0.2,
                    'status': '✅ PASS' if speed_vib_corr > 0.2 and current_vib_corr > 0.2 else '❌ FAIL'
                }
                print(f"📳 Vibration physics: Speed={speed_vib_corr:.3f}, Current={current_vib_corr:.3f} {enhanced_results['vibration_physics']['status']}")
            
            # Calculate overall enhanced physics score
            physics_tests = [result for result in enhanced_results.values() 
                           if isinstance(result, dict) and 'realistic' in result]
            physics_passes = sum(1 for test in physics_tests if test['realistic'])
            
            enhanced_results['enhanced_physics_summary'] = {
                'tests_passed': physics_passes,
                'total_tests': len(physics_tests),
                'pass_rate': physics_passes / len(physics_tests) if physics_tests else 0,
                'overall_valid': physics_passes >= max(1, len(physics_tests) * 0.8)
            }
            
            print(f"🔧 Enhanced physics validation: {physics_passes}/{len(physics_tests)} tests passed")
            
        except Exception as e:
            print(f"❌ Error in enhanced physics validation: {e}")
            enhanced_results['error'] = str(e)
        
        return enhanced_results
    
    def statistical_validation(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Comprehensive statistical validation of synthetic data"""
        
        print(f"\n📊 STATISTICAL VALIDATION")
        print("-" * 30)
        
        stat_results = {
            'distribution_tests': {},
            'correlation_analysis': {},
            'summary_statistics': {},
            'overall_similarity': 0.0
        }
        
        # Get numeric columns for analysis
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        common_numeric_cols = [col for col in numeric_cols if col in synthetic_df.columns]
        
        print(f"🔢 Analyzing {len(common_numeric_cols)} numeric columns...")
        
        # Distribution tests for each column
        similarity_scores = []
        
        for col in common_numeric_cols:
            try:
                real_values = real_df[col].dropna()
                synthetic_values = synthetic_df[col].dropna()
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(real_values, synthetic_values)
                
                # Jensen-Shannon divergence
                bins = np.histogram_bin_edges(np.concatenate([real_values, synthetic_values]), bins=50)
                real_hist, _ = np.histogram(real_values, bins=bins, density=True)
                synthetic_hist, _ = np.histogram(synthetic_values, bins=bins, density=True)
                
                # Normalize to create probability distributions
                real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
                synthetic_hist = synthetic_hist / synthetic_hist.sum() if synthetic_hist.sum() > 0 else synthetic_hist
                
                # Add small epsilon to avoid zero division
                epsilon = 1e-10
                real_hist += epsilon
                synthetic_hist += epsilon
                
                js_divergence = jensenshannon(real_hist, synthetic_hist)
                
                # Calculate similarity score (0-1, higher is better)
                similarity = 1 - min(js_divergence, 1.0)
                similarity_scores.append(similarity)
                
                stat_results['distribution_tests'][col] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'ks_similar': ks_p > 0.05,
                    'js_divergence': js_divergence,
                    'js_similarity': similarity
                }
                
                print(f"   {col}: JS={js_divergence:.3f}, KS_p={ks_p:.3f}, Sim={similarity:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {col}: {e}")
                continue
        
        # Correlation analysis
        try:
            real_corr = real_df[common_numeric_cols].corr()
            synthetic_corr = synthetic_df[common_numeric_cols].corr()
            
            corr_diff = np.abs(real_corr - synthetic_corr)
            mean_corr_diff = corr_diff.mean().mean()
            max_corr_diff = corr_diff.max().max()
            
            stat_results['correlation_analysis'] = {
                'mean_correlation_difference': mean_corr_diff,
                'max_correlation_difference': max_corr_diff,
                'correlation_preserved': mean_corr_diff < 0.1
            }
            
            print(f"📈 Correlation analysis: Mean diff={mean_corr_diff:.3f}, Max diff={max_corr_diff:.3f}")
            
        except Exception as e:
            print(f"❌ Error in correlation analysis: {e}")
        
        # Summary statistics comparison
        stat_results['summary_statistics'] = self.compare_summary_stats(real_df, synthetic_df, common_numeric_cols)
        
        # Overall similarity score
        if similarity_scores:
            stat_results['overall_similarity'] = np.mean(similarity_scores)
            print(f"📊 Overall statistical similarity: {stat_results['overall_similarity']:.3f}")
        
        return stat_results
    
    def compare_summary_stats(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]) -> Dict:
        """Compare summary statistics between real and synthetic data"""
        
        summary_results = {}
        
        for col in columns:
            try:
                real_stats = real_df[col].describe()
                synthetic_stats = synthetic_df[col].describe()
                
                # Calculate relative differences
                mean_diff = abs(real_stats['mean'] - synthetic_stats['mean']) / abs(real_stats['mean'] + 1e-10)
                std_diff = abs(real_stats['std'] - synthetic_stats['std']) / abs(real_stats['std'] + 1e-10)
                
                summary_results[col] = {
                    'real_mean': real_stats['mean'],
                    'synthetic_mean': synthetic_stats['mean'],
                    'mean_relative_diff': mean_diff,
                    'real_std': real_stats['std'],
                    'synthetic_std': synthetic_stats['std'],
                    'std_relative_diff': std_diff,
                    'stats_similar': mean_diff < 0.1 and std_diff < 0.2
                }
                
            except Exception as e:
                print(f"   ❌ Error comparing stats for {col}: {e}")
                continue
        
        return summary_results
    
    def visual_validation(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Create comprehensive visual validation plots"""
        
        print(f"\n📈 VISUAL VALIDATION")
        print("-" * 25)
        
        visual_results = {'plots_created': []}
        
        # Get numeric columns
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        common_numeric_cols = [col for col in numeric_cols if col in synthetic_df.columns][:6]
        
        # Create validation plots
        try:
            self.create_distribution_plots(real_df, synthetic_df, common_numeric_cols)
            visual_results['plots_created'].append('distribution_comparisons.png')
            
            self.create_correlation_comparison(real_df, synthetic_df, common_numeric_cols)
            visual_results['plots_created'].append('correlation_comparison.png')
            
            self.create_box_plot_comparison(real_df, synthetic_df, common_numeric_cols)
            visual_results['plots_created'].append('box_plot_comparison.png')
            
            self.create_scatter_matrix(real_df, synthetic_df, common_numeric_cols[:4])
            visual_results['plots_created'].append('scatter_matrix.png')
            
            self.create_pca_visualization(real_df, synthetic_df, common_numeric_cols)
            visual_results['plots_created'].append('pca_visualization.png')
            
            print(f"✅ Created {len(visual_results['plots_created'])} visualization plots")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
        
        return visual_results
    
    def create_distribution_plots(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]):
        """Create distribution comparison plots"""
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            row, col_idx = i // 3, i % 3
            ax = axes[row, col_idx]
            
            try:
                ax.hist(real_df[col].dropna(), bins=50, alpha=0.6, label='Real', color='blue', density=True)
                ax.hist(synthetic_df[col].dropna(), bins=50, alpha=0.6, label='Synthetic', color='red', density=True)
                
                ax.set_title(f'{col} Distribution Comparison', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', transform=ax.transAxes)
        
        # Remove empty subplots
        for i in range(len(columns), n_rows * 3):
            row, col_idx = i // 3, i % 3
            axes[row, col_idx].remove()
        
        plt.suptitle('📊 Distribution Comparison: Real vs Synthetic Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/distribution_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_comparison(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]):
        """Create correlation matrix comparison"""
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        try:
            real_corr = real_df[columns].corr()
            synthetic_corr = synthetic_df[columns].corr()
            corr_diff = real_corr - synthetic_corr
            
            sns.heatmap(real_corr, annot=True, cmap='RdBu_r', center=0, ax=axes[0], 
                       cbar_kws={'label': 'Correlation'}, fmt='.2f')
            axes[0].set_title('Real Data Correlations', fontweight='bold')
            
            sns.heatmap(synthetic_corr, annot=True, cmap='RdBu_r', center=0, ax=axes[1],
                       cbar_kws={'label': 'Correlation'}, fmt='.2f')
            axes[1].set_title('Synthetic Data Correlations', fontweight='bold')
            
            sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0, ax=axes[2],
                       cbar_kws={'label': 'Difference'}, fmt='.2f')
            axes[2].set_title('Correlation Differences', fontweight='bold')
            
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error creating correlation plots: {e}', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.suptitle('📈 Correlation Matrix Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_box_plot_comparison(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]):
        """Create box plot comparison"""
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            row, col_idx = i // 3, i % 3
            ax = axes[row, col_idx]
            
            try:
                data_to_plot = [real_df[col].dropna(), synthetic_df[col].dropna()]
                ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'])
                ax.set_title(f'{col} Distribution', fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', transform=ax.transAxes)
        
        # Remove empty subplots
        for i in range(len(columns), n_rows * 3):
            row, col_idx = i // 3, i % 3
            axes[row, col_idx].remove()
        
        plt.suptitle('📦 Box Plot Comparison: Real vs Synthetic Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/box_plot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_scatter_matrix(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]):
        """Create scatter plot matrix for key variables"""
        
        if len(columns) < 2:
            return
        
        fig, axes = plt.subplots(len(columns), len(columns), figsize=(15, 15))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograms
                    ax.hist(real_df[col1].dropna(), bins=30, alpha=0.6, label='Real', color='blue', density=True)
                    ax.hist(synthetic_df[col1].dropna(), bins=30, alpha=0.6, label='Synthetic', color='red', density=True)
                    ax.set_title(col1)
                    if i == 0:
                        ax.legend()
                else:
                    # Off-diagonal: scatter plots
                    real_sample = real_df[[col1, col2]].dropna().sample(min(1000, len(real_df)))
                    synthetic_sample = synthetic_df[[col1, col2]].dropna().sample(min(1000, len(synthetic_df)))
                    
                    ax.scatter(real_sample[col2], real_sample[col1], alpha=0.5, s=10, label='Real', color='blue')
                    ax.scatter(synthetic_sample[col2], synthetic_sample[col1], alpha=0.5, s=10, label='Synthetic', color='red')
                    
                    if i == 0 and j == 1:
                        ax.legend()
                
                if i == len(columns) - 1:
                    ax.set_xlabel(col2)
                if j == 0:
                    ax.set_ylabel(col1)
        
        plt.suptitle('🔍 Scatter Matrix: Real vs Synthetic Data Relationships', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_pca_visualization(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]):
        """Create PCA visualization to compare data distributions"""
        
        try:
            # Prepare data
            real_data = real_df[columns].dropna()
            synthetic_data = synthetic_df[columns].dropna()
            
            # Sample data if too large
            if len(real_data) > 5000:
                real_data = real_data.sample(5000)
            if len(synthetic_data) > 5000:
                synthetic_data = synthetic_data.sample(5000)
            
            # Combine and scale data
            combined_data = pd.concat([real_data, synthetic_data])
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(combined_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create labels
            labels = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            for label, color in [('Real', 'blue'), ('Synthetic', 'red')]:
                mask = np.array(labels) == label
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                          alpha=0.6, label=label, color=color, s=20)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_title('🔍 PCA Visualization: Real vs Synthetic Data Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/pca_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating PCA visualization: {e}")
    
    def discriminative_validation(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Train classifier to distinguish real from synthetic data"""
        
        print(f"\n🎯 DISCRIMINATIVE VALIDATION")
        print("-" * 35)
        
        try:
            # Get common numeric columns
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
            common_cols = [col for col in numeric_cols if col in synthetic_df.columns]
            
            # Prepare data
            real_sample = real_df[common_cols].dropna().sample(min(10000, len(real_df)))
            synthetic_sample = synthetic_df[common_cols].dropna().sample(min(10000, len(synthetic_df)))
            
            # Create labels
            real_labeled = real_sample.copy()
            real_labeled['is_real'] = 1
            synthetic_labeled = synthetic_sample.copy()
            synthetic_labeled['is_real'] = 0
            
            # Combine datasets
            combined = pd.concat([real_labeled, synthetic_labeled], ignore_index=True)
            
            # Prepare features and target
            X = combined[common_cols].fillna(combined[common_cols].median())
            y = combined['is_real']
            
            # Train discriminator
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest discriminator
            discriminator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            discriminator.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_accuracy = discriminator.score(X_train_scaled, y_train)
            test_accuracy = discriminator.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(discriminator, X_train_scaled, y_train, cv=5)
            
            # Feature importance
            feature_importance = dict(zip(common_cols, discriminator.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'quality_assessment': self.assess_discrimination_quality(test_accuracy),
                'top_discriminative_features': top_features
            }
            
            print(f"🎯 Discriminator accuracy: {test_accuracy:.3f}")
            print(f"📊 CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"✅ Quality assessment: {results['quality_assessment']}")
            
            # Print top discriminative features
            print("🔍 Top discriminative features:")
            for feature, importance in top_features:
                print(f"   {feature}: {importance:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in discriminative validation: {e}")
            return {'error': str(e)}
    
    def assess_discrimination_quality(self, accuracy: float) -> str:
        """Assess synthetic data quality based on discriminator accuracy"""
        
        if accuracy < 0.55:
            return "Excellent (Very hard to distinguish)"
        elif accuracy < 0.65:
            return "Good (Moderately hard to distinguish)"
        elif accuracy < 0.75:
            return "Fair (Somewhat distinguishable)"
        elif accuracy < 0.85:
            return "Poor (Easily distinguishable)"
        else:
            return "Very Poor (Highly distinguishable)"
    
    def utility_validation(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Validate utility through Train Synthetic Test Real (TSTR)"""
        
        print(f"\n🛠️ UTILITY VALIDATION (TSTR)")
        print("-" * 35)
        
        try:
            # Prepare target variable (fault detection)
            if 'current_status' in real_df.columns:
                real_target = (real_df['current_status'] == 'HIGH').astype(int)
                synthetic_target = (synthetic_df['current_status'] == 'HIGH').astype(int)
            else:
                print("⚠️ No fault labels found for utility testing")
                return {'error': 'No fault labels available'}
            
            # Get common numeric features
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col in synthetic_df.columns and col != 'is_fault'][:10]
            
            # Prepare datasets
            real_X = real_df[feature_cols].fillna(real_df[feature_cols].median())
            real_y = real_target
            
            synthetic_X = synthetic_df[feature_cols].fillna(synthetic_df[feature_cols].median())
            synthetic_y = synthetic_target
            
            # Split real data for testing
            _, real_X_test, _, real_y_test = train_test_split(
                real_X, real_y, test_size=0.3, random_state=42, stratify=real_y
            )
            
            # Train models
            scaler = StandardScaler()
            
            # TRTR: Train Real Test Real
            real_X_train_scaled = scaler.fit_transform(real_X)
            real_X_test_scaled = scaler.transform(real_X_test)
            
            trtr_model = RandomForestClassifier(n_estimators=100, random_state=42)
            trtr_model.fit(real_X_train_scaled, real_y)
            trtr_score = trtr_model.score(real_X_test_scaled, real_y_test)
            
            # TSTR: Train Synthetic Test Real
            synthetic_X_scaled = scaler.fit_transform(synthetic_X)
            real_X_test_scaled_tstr = scaler.transform(real_X_test)
            
            tstr_model = RandomForestClassifier(n_estimators=100, random_state=42)
            tstr_model.fit(synthetic_X_scaled, synthetic_y)
            tstr_score = tstr_model.score(real_X_test_scaled_tstr, real_y_test)
            
            # Utility score (how close TSTR is to TRTR)
            utility_score = 1 - abs(trtr_score - tstr_score)
            
            results = {
                'trtr_score': trtr_score,
                'tstr_score': tstr_score,
                'utility_score': utility_score,
                'utility_assessment': self.assess_utility_quality(utility_score),
                'performance_gap': abs(trtr_score - tstr_score)
            }
            
            print(f"📊 TRTR Score (Real→Real): {trtr_score:.3f}")
            print(f"📊 TSTR Score (Synthetic→Real): {tstr_score:.3f}")
            print(f"🛠️ Utility Score: {utility_score:.3f}")
            print(f"✅ Assessment: {results['utility_assessment']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in utility validation: {e}")
            return {'error': str(e)}
    
    def assess_utility_quality(self, utility_score: float) -> str:
        """Assess synthetic data utility quality"""
        
        if utility_score > 0.95:
            return "Excellent utility (Very close performance)"
        elif utility_score > 0.90:
            return "Good utility (Close performance)"
        elif utility_score > 0.80:
            return "Fair utility (Acceptable performance gap)"
        elif utility_score > 0.70:
            return "Poor utility (Large performance gap)"
        else:
            return "Very Poor utility (Significant performance degradation)"
    
    def generate_enhanced_comprehensive_report(self) -> Dict:
        """Generate enhanced comprehensive validation report"""
        
        print(f"\n📋 GENERATING ENHANCED COMPREHENSIVE REPORT")
        print("-" * 50)
        
        # Collect all validation results
        overall_results = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'validation_passed': False,
                'overall_quality_score': 0.0,
                'dashboard_ready': False,
                'recommendations': []
            },
            'detailed_results': self.validation_results
        }
        
        # Calculate enhanced overall quality score
        quality_components = []
        
        # Statistical similarity (0-1)
        if 'statistical' in self.validation_results:
            stat_score = self.validation_results['statistical'].get('overall_similarity', 0)
            quality_components.append(('statistical_similarity', stat_score, 0.15))
        
        # Discriminative quality (inverted accuracy)
        if 'discriminative' in self.validation_results:
            disc_acc = self.validation_results['discriminative'].get('test_accuracy', 1.0)
            disc_score = max(0, 1 - (disc_acc - 0.5) * 2)
            quality_components.append(('discriminative_quality', disc_score, 0.15))
        
        # Physics validation
        if 'physics' in self.validation_results:
            # Combine original and enhanced physics results
            physics_score = 0
            total_physics = 0
            
            # Original physics tests
            original_physics = self.validation_results['physics'].get('overall_physics', {})
            if 'pass_rate' in original_physics:
                physics_score += original_physics['pass_rate'] * 0.5
                total_physics += 0.5
            
            # Enhanced physics tests
            enhanced_physics = self.validation_results['physics'].get('enhanced_physics_summary', {})
            if 'pass_rate' in enhanced_physics:
                physics_score += enhanced_physics['pass_rate'] * 0.5
                total_physics += 0.5
            
            final_physics_score = physics_score / total_physics if total_physics > 0 else 0
            quality_components.append(('physics_preservation', final_physics_score, 0.20))
        
        # Threshold consistency
        if 'threshold_consistency' in self.validation_results:
            threshold_score = self.validation_results['threshold_consistency'].get('overall_consistency', 0)
            quality_components.append(('threshold_consistency', threshold_score, 0.15))
        
        # Dashboard compatibility
        if 'dashboard_compatibility' in self.validation_results:
            dashboard_score = self.validation_results['dashboard_compatibility'].get('overall_compatibility', 0)
            quality_components.append(('dashboard_compatibility', dashboard_score, 0.15))
        
        # Fault progression realism
        if 'fault_progression' in self.validation_results:
            progression_score = self.validation_results['fault_progression'].get('progression_quality', 0)
            quality_components.append(('fault_progression', progression_score, 0.10))
        
        # Operating mode distribution
        if 'operating_modes' in self.validation_results:
            mode_score = self.validation_results['operating_modes'].get('overall_mode_quality', 0)
            quality_components.append(('operating_modes', mode_score, 0.05))
        
        # Utility score
        if 'utility' in self.validation_results:
            utility_score = self.validation_results['utility'].get('utility_score', 0)
            quality_components.append(('utility_performance', utility_score, 0.05))
        
        # Calculate weighted overall score
        if quality_components:
            total_weight = sum(weight for _, _, weight in quality_components)
            overall_score = sum(score * weight for _, score, weight in quality_components) / total_weight
            overall_results['validation_summary']['overall_quality_score'] = overall_score
        
        # Enhanced validation requirements
        enhanced_requirements = {
            'statistical_similarity': 0.7,
            'discriminative_quality': 0.4,
            'physics_preservation': 0.8,
            'threshold_consistency': 0.95,
            'dashboard_compatibility': 0.9,
            'fault_progression': 0.8,
            'operating_modes': 0.8,
            'utility_performance': 0.8
        }
        
        passed_components = []
        failed_components = []
        
        for name, score, _ in quality_components:
            min_req = enhanced_requirements.get(name, 0.7)
            if score >= min_req:
                passed_components.append(name)
            else:
                failed_components.append((name, score, min_req))
        
        # Overall pass/fail
        validation_passed = len(failed_components) == 0
        overall_results['validation_summary']['validation_passed'] = validation_passed
        
        # Dashboard readiness
        dashboard_critical = ['threshold_consistency', 'dashboard_compatibility', 'physics_preservation']
        dashboard_ready = all(name in passed_components for name in dashboard_critical if name in [c[0] for c in quality_components])
        overall_results['validation_summary']['dashboard_ready'] = dashboard_ready
        
        # Generate enhanced recommendations
        recommendations = []
        
        if validation_passed:
            recommendations.append("✅ ENHANCED VALIDATION PASSED - Dataset is ready for optimized dashboard ML training")
            recommendations.append("🎯 All quality thresholds met including dashboard compatibility")
            recommendations.append("🚀 Proceed to Phase 3: ML Training with Optimized Thresholds")
        else:
            recommendations.append("❌ ENHANCED VALIDATION FAILED - Dataset needs improvement")
            
            for component, score, min_req in failed_components:
                if component == 'threshold_consistency':
                    recommendations.append(f"🎯 Fix threshold consistency (current: {score:.2f}, required: {min_req:.2f})")
                    recommendations.append("   - Ensure generator uses exact same threshold logic as dashboard")
                    recommendations.append("   - Verify classification functions match optimized thresholds")
                
                elif component == 'dashboard_compatibility':
                    recommendations.append(f"📱 Improve dashboard compatibility (current: {score:.2f}, required: {min_req:.2f})")
                    recommendations.append("   - Check all required columns are present")
                    recommendations.append("   - Validate data formats and ranges")
                
                elif component == 'fault_progression':
                    recommendations.append(f"📈 Improve fault progression realism (current: {score:.2f}, required: {min_req:.2f})")
                    recommendations.append("   - Add more gradual fault development patterns")
                    recommendations.append("   - Reduce abrupt status changes")
                
                elif component == 'operating_modes':
                    recommendations.append(f"⚙️ Fix operating mode distribution (current: {score:.2f}, required: {min_req:.2f})")
                    recommendations.append("   - Adjust mode probabilities to match realistic patterns")
                    recommendations.append("   - Improve day/night operational patterns")
        
        if dashboard_ready:
            recommendations.append("✅ DASHBOARD READY - Critical compatibility tests passed")
        else:
            recommendations.append("❌ DASHBOARD NOT READY - Fix critical compatibility issues first")
        
        overall_results['validation_summary']['recommendations'] = recommendations
        
        # Print enhanced summary
        print(f"📊 Enhanced Quality Score: {overall_score:.3f}")
        print(f"✅ Validation Status: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"📱 Dashboard Ready: {'YES' if dashboard_ready else 'NO'}")
        print(f"📋 Component Status:")
        for name, score, _ in quality_components:
            status = "✅ PASS" if name in passed_components else "❌ FAIL"
            print(f"   {name}: {score:.3f} {status}")
        
        print(f"\n💡 Enhanced Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
        
        # Save enhanced report
        report_path = f"{self.output_dir}/reports/enhanced_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        print(f"\n💾 Enhanced comprehensive report saved: {report_path}")
        
        return overall_results
    
    def run_enhanced_complete_validation(self, real_data_path: str, synthetic_data_path: str) -> Dict:
        """Run enhanced complete validation pipeline"""
        
        print(f"\n🚀 RUNNING ENHANCED COMPLETE VALIDATION PIPELINE")
        print("=" * 65)
        
        # Load datasets
        real_df, synthetic_df = self.load_datasets(real_data_path, synthetic_data_path)
        
        if real_df is None or synthetic_df is None:
            return {'error': 'Failed to load datasets'}
        
        # Run all enhanced validation tests
        print(f"\n1️⃣ Statistical Validation...")
        self.validation_results['statistical'] = self.statistical_validation(real_df, synthetic_df)
        
        print(f"\n2️⃣ Visual Validation...")
        self.validation_results['visual'] = self.visual_validation(real_df, synthetic_df)
        
        print(f"\n3️⃣ Discriminative Validation...")
        self.validation_results['discriminative'] = self.discriminative_validation(real_df, synthetic_df)
        
        print(f"\n4️⃣ Motor Physics Validation...")
        self.validation_results['physics'] = self.enhanced_motor_physics_validation(synthetic_df)
        
        print(f"\n5️⃣ Threshold Consistency Validation...")
        self.validation_results['threshold_consistency'] = self.validate_threshold_consistency(synthetic_df)
        
        print(f"\n6️⃣ Dashboard Compatibility Validation...")
        self.validation_results['dashboard_compatibility'] = self.validate_dashboard_compatibility(synthetic_df)
        
        print(f"\n7️⃣ Fault Progression Validation...")
        self.validation_results['fault_progression'] = self.validate_fault_progressions(synthetic_df)
        
        print(f"\n8️⃣ Operating Mode Validation...")
        self.validation_results['operating_modes'] = self.validate_operating_modes(synthetic_df)
        
        print(f"\n9️⃣ Utility Validation...")
        self.validation_results['utility'] = self.utility_validation(real_df, synthetic_df)
        
        print(f"\n🔟 Enhanced Comprehensive Report Generation...")
        final_results = self.generate_enhanced_comprehensive_report()
        
        print(f"\n🎉 ENHANCED COMPLETE VALIDATION FINISHED!")
        print("=" * 50)
        
        return final_results

# Main execution function
def main():
    """Execute enhanced comprehensive synthetic data validation"""
    
    print(f"\n🔍 STARTING ENHANCED COMPREHENSIVE SYNTHETIC DATA VALIDATION")
    print("=" * 70)
    
    # Initialize enhanced validator
    validator = EnhancedComprehensiveDataValidator()
    
    # FIXED: Define file paths for motor_monitoring project structure
    real_data_path = '../ml_development/real_data/motor_real_data_original.csv'
    synthetic_data_path = '../ml_development/phase2_synthesis/results/motor_production_345k_optimized.csv'  # Correct path to generated dataset
    
    try:
        # Run enhanced complete validation
        results = validator.run_enhanced_complete_validation(real_data_path, synthetic_data_path)
        
        if 'error' not in results:
            validation_passed = results['validation_summary']['validation_passed']
            dashboard_ready = results['validation_summary']['dashboard_ready']
            overall_score = results['validation_summary']['overall_quality_score']
            
            print(f"\n🏆 ENHANCED VALIDATION RESULTS:")
            print(f"📊 Overall Quality Score: {overall_score:.3f}")
            print(f"✅ Validation Status: {'PASSED ✅' if validation_passed else 'FAILED ❌'}")
            print(f"📱 Dashboard Ready: {'YES ✅' if dashboard_ready else 'NO ❌'}")
            
            if validation_passed and dashboard_ready:
                print(f"\n🚀 READY FOR OPTIMIZED ML TRAINING!")
                print(f"🎯 Dataset is fully compatible with optimized dashboard system")
                print(f"📈 Proceed to Phase 3 with confidence")
            elif dashboard_ready:
                print(f"\n⚠️ DASHBOARD COMPATIBLE BUT NEEDS MINOR IMPROVEMENTS")
                print(f"📱 Can proceed with dashboard testing while improving data quality")
            else:
                print(f"\n❌ DATASET NEEDS CRITICAL IMPROVEMENTS")
                print(f"🔧 Fix dashboard compatibility issues before proceeding")
            
            print(f"\n📋 Enhanced Outputs Generated:")
            print(f"   📊 Validation plots: {validator.output_dir}/plots/")
            print(f"   📄 Detailed report: {validator.output_dir}/reports/enhanced_validation_report.json")
            
        else:
            print(f"❌ Enhanced validation failed with error: {results['error']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in enhanced validation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
