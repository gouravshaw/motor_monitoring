import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import math
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CompleteMotorDatasetGenerator:
    """
    Complete Industrial Motor Fault Detection Dataset Generator
    
    Specifications matched to your project:
    - 12V DC Motor with BTS7960 driver
    - 20A power supply capability
    - BME280 (Temperature/Humidity/Pressure)
    - MPU6050 (3-axis vibration)  
    - ACS712 (Hall effect current sensor)
    - ESP32 MQTT data structure
    - MongoDB integration ready
    """
    
    def __init__(self):
        print("🔧 Initializing Motor Dataset Generator...")
        print("📊 Target: 50,000 data points over 42 days")
        print("⚙️ Motor: 12V DC with 20A capability")
        print("🔌 Sensors: BME280 + MPU6050 + ACS712")
        print("")
        
        # Your exact motor specifications
        self.motor_specs = {
            'voltage_nominal': 12.0,
            'voltage_range': (11.6, 12.4),
            'max_current': 20.0,
            'typical_operating_current': (2.0, 15.0),
            'max_speed_rpm': 9970,
            'power_rating_max': 240,  # 12V × 20A theoretical max
            'typical_power_range': (24, 180),  # 2A to 15A × 12V
            'thermal_time_constant': 300,  # 5 minutes (300 seconds)
            'cooling_time_constant': 600,  # 10 minutes cooling
            'bearing_frequencies': [580, 1160, 1740, 2320]  # Motor-specific
        }
        
        # Environmental conditions (educational lab setting)
        self.environment = {
            'ambient_temp_range': (18, 28),    # Typical lab temperature
            'humidity_range': (35, 65),        # Indoor humidity
            'pressure_range': (1005, 1018),   # Local atmospheric
            'daily_temp_variation': 4,         # °C swing per day
            'seasonal_temp_drift': 2           # °C over 42 days
        }
        
        # Sensor specifications (exact hardware)
        self.sensors = {
            'bme280': {
                'temp_accuracy': 0.5,      # ±0.5°C
                'temp_resolution': 0.01,   # 0.01°C
                'humidity_accuracy': 3.0,  # ±3%RH
                'pressure_accuracy': 1.0,  # ±1hPa
                'noise_temp': 0.1,         # °C RMS noise
                'noise_humidity': 0.5,     # %RH RMS noise
                'noise_pressure': 0.2      # hPa RMS noise
            },
            'mpu6050': {
                'range': 16,               # ±16g
                'resolution': 0.0049,      # g per LSB
                'noise_density': 400,      # μg/√Hz
                'bandwidth': 260,          # Hz
                'sample_rate': 1000,       # Hz internal
                'vibration_noise_floor': 0.005  # m/s² noise floor
            },
            'acs712': {
                'model': 'ACS712-20A',
                'sensitivity': 100,        # mV/A (for 20A version)
                'zero_current_offset': 2.5, # V
                'accuracy': 0.1,           # ±100mA
                'bandwidth': 80,           # kHz
                'noise_current': 0.02,     # A RMS noise
                'temperature_drift': 0.002 # %/°C
            }
        }
        
        # Comprehensive fault scenarios
        self.fault_scenarios = {
            'normal_operation': {
                'probability': 0.65,
                'description': 'Normal healthy operation',
                'current_factor': 1.0,
                'temp_increase': 0,
                'vibration_factor': 1.0,
                'efficiency_factor': 1.0
            },
            'bearing_wear_early': {
                'probability': 0.10,
                'description': 'Early stage bearing wear',
                'current_factor': 1.05,
                'temp_increase': (2, 6),
                'vibration_factor': (1.3, 1.8),
                'frequency_signatures': [580, 1160],
                'progression_rate': 0.0008,
                'efficiency_factor': 0.98
            },
            'bearing_wear_advanced': {
                'probability': 0.06,
                'description': 'Advanced bearing degradation',
                'current_factor': 1.15,
                'temp_increase': (8, 18),
                'vibration_factor': (2.2, 3.5),
                'frequency_signatures': [580, 1160, 1740],
                'progression_rate': 0.003,
                'efficiency_factor': 0.92
            },
            'electrical_fault_minor': {
                'probability': 0.05,
                'description': 'Minor electrical issues',
                'current_spike_prob': 0.15,
                'current_spike_factor': (1.2, 1.6),
                'temp_increase': (5, 12),
                'vibration_factor': 1.1,
                'voltage_fluctuation': 0.3,
                'efficiency_factor': 0.95
            },
            'electrical_fault_major': {
                'probability': 0.03,
                'description': 'Significant electrical faults',
                'current_spike_prob': 0.3,
                'current_spike_factor': (1.6, 2.2),
                'temp_increase': (15, 30),
                'vibration_factor': 1.3,
                'voltage_fluctuation': 0.8,
                'efficiency_factor': 0.85
            },
            'thermal_stress': {
                'probability': 0.04,
                'description': 'Thermal overload conditions',
                'temp_rise_rate': (1.5, 4.0),  # °C/minute
                'current_factor': 1.08,
                'vibration_factor': 1.2,
                'thermal_runaway_risk': True,
                'efficiency_factor': 0.90
            },
            'mechanical_overload': {
                'probability': 0.04,
                'description': 'Mechanical binding/overload',
                'current_increase': (1.3, 1.9),
                'temp_increase': (10, 25),
                'vibration_factor': (1.6, 2.8),
                'speed_reduction_factor': 0.85,
                'efficiency_factor': 0.80
            },
            'power_supply_stress': {
                'probability': 0.02,
                'description': 'Power supply limitations',
                'voltage_sag': (0.4, 1.2),     # Voltage drop under load
                'current_ripple': 0.15,        # Current fluctuation
                'temp_increase': (3, 8),
                'efficiency_factor': 0.93
            },
            'sensor_drift': {
                'probability': 0.01,
                'description': 'Sensor calibration drift',
                'temp_offset_drift': (0.5, 2.0),
                'current_offset_drift': (0.1, 0.3),
                'vibration_scale_drift': (0.05, 0.15)
            }
        }
        
        # Operating patterns (realistic daily usage)
        self.operating_patterns = {
            'daily_schedule': {
                'start_hour': 8,
                'end_hour': 17,
                'peak_hours': (10, 15),
                'operation_probability': 0.75
            },
            'weekly_pattern': {
                'weekday_factor': 1.0,
                'saturday_factor': 0.3,
                'sunday_factor': 0.1
            },
            'speed_distribution': {
                'stopped': 0.15,
                '25_percent': 0.20,
                '50_percent': 0.25,
                '75_percent': 0.25,
                '100_percent': 0.15
            },
            'direction_distribution': {
                'forward': 0.82,
                'reverse': 0.13,
                'stopped': 0.05
            }
        }
    
    def generate_complete_dataset(self, num_points=50000, time_span_days=42):
        """
        Generate the complete synthetic dataset
        
        Args:
            num_points: Total number of data points to generate
            time_span_days: Time span in days for the simulation
            
        Returns:
            pandas.DataFrame: Complete dataset ready for use
        """
        print(f"🚀 Starting dataset generation...")
        print(f"📊 Generating {num_points:,} data points")
        print(f"📅 Time span: {time_span_days} days")
        print(f"⏱️  Sampling interval: 5 seconds")
        print("")
        
        # Initialize dataset storage
        dataset = []
        
        # Time parameters
        start_time = datetime.now() - timedelta(days=time_span_days)
        time_interval = timedelta(seconds=5)
        
        # Progressive states
        cumulative_operating_hours = 0.0
        bearing_wear_level = 0.0
        thermal_stress_accumulated = 0.0
        electrical_stress_level = 0.0
        sensor_drift_level = 0.0
        
        # Progress tracking
        progress_milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        next_milestone_idx = 0
        
        print("🔄 Generation progress:")
        
        for i in range(num_points):
            # Calculate current timestamp
            current_time = start_time + (i * time_interval)
            
            # Check progress
            progress = (i + 1) / num_points
            if next_milestone_idx < len(progress_milestones) and progress >= progress_milestones[next_milestone_idx]:
                print(f"   {progress_milestones[next_milestone_idx]*100:.0f}% complete - {i+1:,}/{num_points:,} points generated")
                next_milestone_idx += 1
            
            # Determine if motor is operating
            is_operating = self._should_motor_operate(current_time)
            
            if is_operating:
                # Generate operational data point
                data_point = self._generate_operational_reading(
                    current_time,
                    bearing_wear_level,
                    thermal_stress_accumulated,
                    electrical_stress_level,
                    sensor_drift_level,
                    cumulative_operating_hours
                )
                
                # Update progressive states
                time_delta_hours = 5.0 / 3600  # 5 seconds to hours
                cumulative_operating_hours += time_delta_hours
                
                # Progressive wear updates
                bearing_wear_level += random.uniform(0, 0.00008)
                if data_point['temperature'] > 40:
                    thermal_stress_accumulated += 0.008
                if data_point['current'] > 12:
                    electrical_stress_level += 0.005
                sensor_drift_level += random.uniform(0, 0.00002)
                
            else:
                # Generate idle/stopped reading
                data_point = self._generate_idle_reading(
                    current_time,
                    sensor_drift_level
                )
            
            dataset.append(data_point)
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        print(f"✅ Dataset generation complete!")
        print(f"📊 Generated {len(df):,} total records")
        print(f"⚙️  Operating records: {len(df[df['motor_speed'] > 0]):,}")
        print(f"⏸️  Idle records: {len(df[df['motor_speed'] == 0]):,}")
        print("")
        
        return df
    
    def _should_motor_operate(self, current_time):
        """Determine if motor should be operating at given time"""
        hour = current_time.hour
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Basic operating hours check
        if hour < self.operating_patterns['daily_schedule']['start_hour'] or \
           hour > self.operating_patterns['daily_schedule']['end_hour']:
            return False
        
        # Weekend reduction
        if weekday == 5:  # Saturday
            if random.random() > self.operating_patterns['weekly_pattern']['saturday_factor']:
                return False
        elif weekday == 6:  # Sunday
            if random.random() > self.operating_patterns['weekly_pattern']['sunday_factor']:
                return False
        
        # Peak hours boost
        if self.operating_patterns['daily_schedule']['peak_hours'][0] <= hour <= \
           self.operating_patterns['daily_schedule']['peak_hours'][1]:
            operation_prob = self.operating_patterns['daily_schedule']['operation_probability'] * 1.2
        else:
            operation_prob = self.operating_patterns['daily_schedule']['operation_probability']
        
        return random.random() < operation_prob
    
    def _generate_operational_reading(self, timestamp, bearing_wear, thermal_stress, 
                                    electrical_stress, sensor_drift, operating_hours):
        """Generate a complete operational reading"""
        
        # Determine operating conditions
        motor_speed_percent = self._select_motor_speed()
        motor_direction = self._select_motor_direction()
        
        if motor_direction == 'stopped':
            motor_speed_percent = 0
        
        # Environmental base conditions
        ambient_temp = self._calculate_ambient_temperature(timestamp)
        humidity = self._calculate_humidity(timestamp, ambient_temp)
        pressure = self._calculate_atmospheric_pressure(timestamp)
        
        # Select fault scenario
        fault_type = self._select_fault_scenario(bearing_wear, thermal_stress, 
                                               electrical_stress, operating_hours)
        fault_config = self.fault_scenarios[fault_type]
        
        # Calculate motor electrical characteristics
        base_current = self._calculate_base_current(motor_speed_percent, ambient_temp)
        actual_current = self._apply_current_faults(base_current, fault_config, motor_speed_percent)
        voltage = self._calculate_supply_voltage(actual_current, fault_config)
        
        # Calculate motor temperature
        motor_temperature = self._calculate_motor_temperature(
            motor_speed_percent, actual_current, ambient_temp, fault_config, thermal_stress
        )
        
        # Calculate vibration characteristics  
        vibration_data = self._calculate_vibration_components(
            motor_speed_percent, fault_config, bearing_wear, actual_current
        )
        
        # Apply sensor drift effects
        motor_temperature = self._apply_sensor_drift(motor_temperature, 'temperature', sensor_drift)
        actual_current = self._apply_sensor_drift(actual_current, 'current', sensor_drift)
        
        # Calculate derived values
        power_consumption = voltage * actual_current
        motor_efficiency = self._calculate_motor_efficiency(motor_speed_percent, actual_current, motor_temperature)
        
        # Determine status classifications
        temp_status = self._classify_temperature_status(motor_temperature)
        vib_status = self._classify_vibration_status(vibration_data['magnitude'])
        current_status = self._classify_current_status(actual_current)
        
        # Motor control states
        motor_enabled = self._determine_motor_enabled_state(motor_temperature, actual_current, vibration_data['magnitude'])
        is_transitioning = random.random() < 0.04  # 4% chance of transitioning
        motor_status = self._determine_motor_status(fault_type, motor_enabled, motor_temperature, actual_current)
        
        # Build complete data point
        return {
            # Timestamp and identification
            'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
            'device_id': 'ESP32_001',
            'created_at': timestamp,
            
            # Environmental sensors (BME280)
            'temperature': round(motor_temperature, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            
            # Vibration sensors (MPU6050)
            'vibration': round(vibration_data['magnitude'], 3),
            'vibration_x': round(vibration_data['x'], 3),
            'vibration_y': round(vibration_data['y'], 3),
            'vibration_z': round(vibration_data['z'], 3),
            
            # Current sensor (ACS712)
            'current': round(actual_current, 2),
            'voltage': round(voltage, 1),
            'power_consumption': round(power_consumption, 1),
            
            # Motor control parameters
            'motor_speed': motor_speed_percent,
            'motor_direction': motor_direction,
            'motor_status': motor_status,
            'motor_enabled': motor_enabled,
            'is_transitioning': is_transitioning,
            'motor_efficiency': round(motor_efficiency, 1),
            
            # Status classifications (matching your ESP32 code)
            'temp_status': temp_status,
            'vib_status': vib_status,
            'current_status': current_status,
            
            # Fault detection labels (for ML training)
            'fault_type': fault_type,
            'fault_severity': self._calculate_fault_severity(fault_type, motor_temperature, actual_current, vibration_data['magnitude']),
            
            # Progressive wear indicators
            'bearing_wear_level': round(bearing_wear, 6),
            'thermal_stress_level': round(thermal_stress, 4),
            'electrical_stress_level': round(electrical_stress, 4),
            'operating_hours': round(operating_hours, 2),
            
            # Environmental factors
            'ambient_temperature': round(ambient_temp, 1),
            'temperature_rise': round(motor_temperature - ambient_temp, 1)
        }
    
    def _generate_idle_reading(self, timestamp, sensor_drift):
        """Generate reading when motor is idle/stopped"""
        
        ambient_temp = self._calculate_ambient_temperature(timestamp)
        humidity = self._calculate_humidity(timestamp, ambient_temp)
        pressure = self._calculate_atmospheric_pressure(timestamp)
        
        # Apply sensor drift to idle readings too
        ambient_temp = self._apply_sensor_drift(ambient_temp, 'temperature', sensor_drift)
        
        return {
            'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
            'device_id': 'ESP32_001',
            'created_at': timestamp,
            'temperature': round(ambient_temp + random.uniform(-0.5, 0.5), 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'vibration': round(random.uniform(0.005, 0.015), 3),
            'vibration_x': round(random.uniform(-0.01, 0.01), 3),
            'vibration_y': round(random.uniform(-0.01, 0.01), 3),
            'vibration_z': round(random.uniform(-0.01, 0.01), 3),
            'current': round(random.uniform(0.05, 0.12), 2),
            'voltage': round(12.0 + random.uniform(-0.1, 0.1), 1),
            'power_consumption': round(random.uniform(0.6, 1.4), 1),
            'motor_speed': 0,
            'motor_direction': 'stopped',
            'motor_status': 'stopped',
            'motor_enabled': True,
            'is_transitioning': False,
            'motor_efficiency': 0,
            'temp_status': 'NORMAL',
            'vib_status': 'LOW',
            'current_status': 'LOW',
            'fault_type': 'normal_operation',
            'fault_severity': 0,
            'bearing_wear_level': 0,
            'thermal_stress_level': 0,
            'electrical_stress_level': 0,
            'operating_hours': 0,
            'ambient_temperature': round(ambient_temp, 1),
            'temperature_rise': 0
        }
    
    def _select_motor_speed(self):
        """Select motor speed based on realistic distribution"""
        speed_choices = [0, 25, 50, 75, 100]
        speed_weights = [
            self.operating_patterns['speed_distribution']['stopped'],
            self.operating_patterns['speed_distribution']['25_percent'],
            self.operating_patterns['speed_distribution']['50_percent'],
            self.operating_patterns['speed_distribution']['75_percent'],
            self.operating_patterns['speed_distribution']['100_percent']
        ]
        return random.choices(speed_choices, weights=speed_weights)[0]
    
    def _select_motor_direction(self):
        """Select motor direction based on realistic distribution"""
        directions = ['forward', 'reverse', 'stopped']
        direction_weights = [
            self.operating_patterns['direction_distribution']['forward'],
            self.operating_patterns['direction_distribution']['reverse'],
            self.operating_patterns['direction_distribution']['stopped']
        ]
        return random.choices(directions, weights=direction_weights)[0]
    
    def _calculate_ambient_temperature(self, timestamp):
        """Calculate realistic ambient temperature with daily and seasonal variation"""
        base_temp = 22.0  # Base lab temperature
        
        # Daily variation (cooler at night, warmer in afternoon)
        hour = timestamp.hour
        daily_variation = self.environment['daily_temp_variation'] * math.sin(
            (hour - 6) * math.pi / 12
        ) * 0.5
        
        # Seasonal drift over 42 days
        day_of_simulation = timestamp.day
        seasonal_variation = self.environment['seasonal_temp_drift'] * math.sin(
            day_of_simulation * math.pi / 21
        ) * 0.3
        
        # Random variation
        random_variation = random.uniform(-1, 1)
        
        ambient_temp = base_temp + daily_variation + seasonal_variation + random_variation
        return max(15, min(ambient_temp, 32))  # Realistic lab bounds
    
    def _calculate_humidity(self, timestamp, ambient_temp):
        """Calculate humidity with temperature correlation"""
        base_humidity = 50.0
        
        # Inverse correlation with temperature
        temp_effect = -(ambient_temp - 22) * 1.5
        
        # Daily variation
        hour = timestamp.hour
        daily_variation = 10 * math.sin((hour - 3) * math.pi / 12) * 0.3
        
        # Random variation
        random_variation = random.uniform(-5, 5)
        
        humidity = base_humidity + temp_effect + daily_variation + random_variation
        return max(25, min(humidity, 75))  # Realistic indoor bounds
    
    def _calculate_atmospheric_pressure(self, timestamp):
        """Calculate atmospheric pressure with weather simulation"""
        base_pressure = 1013.25
        
        # Slow weather patterns (multi-day cycles)
        day = timestamp.day
        weather_cycle = 8 * math.sin(day * math.pi / 5) + 5 * math.sin(day * math.pi / 3)
        
        # Random variations
        random_variation = random.uniform(-2, 2)
        
        pressure = base_pressure + weather_cycle + random_variation
        return max(990, min(pressure, 1030))
    
    def _calculate_base_current(self, speed_percent, ambient_temp):
        """Calculate base motor current based on speed and temperature"""
        if speed_percent == 0:
            return random.uniform(0.08, 0.15)  # Standby current
        
        # Current curve: approximately quadratic with speed but not perfect
        speed_factor = speed_percent / 100.0
        base_current = 2.2 + (speed_factor ** 1.6) * 10.5
        
        # Temperature coefficient (higher temp = slightly higher resistance losses)
        temp_coefficient = 1 + (ambient_temp - 22) * 0.003
        
        # Load variations (random mechanical load changes)
        load_factor = random.uniform(0.85, 1.25)
        
        # Efficiency variations at different speeds
        if speed_percent < 30:
            efficiency_penalty = 1.15  # Less efficient at very low speeds
        elif speed_percent > 90:
            efficiency_penalty = 1.08  # Less efficient at very high speeds
        else:
            efficiency_penalty = 1.0
        
        current = base_current * temp_coefficient * load_factor * efficiency_penalty
        return max(0.1, min(current, 18.0))  # Physical limits
    
    def _apply_current_faults(self, base_current, fault_config, speed_percent):
        """Apply fault-specific current modifications"""
        current = base_current
        
        # Apply current factor modifications
        if 'current_factor' in fault_config:
            current *= fault_config['current_factor']
        
        # Apply current increase for mechanical faults
        if 'current_increase' in fault_config:
            if isinstance(fault_config['current_increase'], tuple):
                increase = random.uniform(*fault_config['current_increase'])
            else:
                increase = fault_config['current_increase']
            current *= increase
        
        # Apply current spikes for electrical faults
        if 'current_spike_prob' in fault_config:
            if random.random() < fault_config['current_spike_prob']:
                spike_factor = random.uniform(*fault_config['current_spike_factor'])
                current *= spike_factor
        
        # Add ACS712 sensor noise and drift
        sensor_noise = random.uniform(-self.sensors['acs712']['noise_current'], 
                                    self.sensors['acs712']['noise_current'])
        current += sensor_noise
        
        return max(0.05, min(current, 19.5))  # Sensor and safety limits
    
    def _calculate_supply_voltage(self, current, fault_config):
        """Calculate supply voltage with load effects and faults"""
        base_voltage = 12.0
        
        # Voltage sag under load (power supply internal resistance)
        voltage_sag = current * 0.02  # 20mΩ internal resistance
        
        # Apply power supply stress faults
        if 'voltage_sag' in fault_config:
            additional_sag = random.uniform(*fault_config['voltage_sag'])
            voltage_sag += additional_sag
        
        # Voltage fluctuations
        if 'voltage_fluctuation' in fault_config:
            fluctuation = random.uniform(-fault_config['voltage_fluctuation'], 
                                       fault_config['voltage_fluctuation'])
            base_voltage += fluctuation
        
        # Normal supply variations
        normal_variation = random.uniform(-0.15, 0.15)
        
        voltage = base_voltage - voltage_sag + normal_variation
        return max(10.5, min(voltage, 13.5))  # Realistic power supply bounds
    
    def _calculate_motor_temperature(self, speed_percent, current, ambient_temp, fault_config, thermal_stress):
        """Calculate motor temperature with thermal dynamics"""
        
        # Base temperature rise from power dissipation
        power_loss = current * 12.0 * 0.12  # Assume 12% power loss as heat
        thermal_resistance = 0.35  # °C/W (motor to ambient)
        base_temp_rise = power_loss * thermal_resistance
        
        # Speed-dependent cooling (self-ventilation effect)
        if speed_percent > 0:
            cooling_effectiveness = 1 + (speed_percent / 100) * 0.25
            effective_temp_rise = base_temp_rise / cooling_effectiveness
        else:
            effective_temp_rise = base_temp_rise * 1.1  # Reduced cooling when stopped
        
        motor_temp = ambient_temp + effective_temp_rise
        
        # Apply fault-specific temperature increases
        if 'temp_increase' in fault_config:
            if isinstance(fault_config['temp_increase'], tuple):
                temp_increase = random.uniform(*fault_config['temp_increase'])
            else:
                temp_increase = fault_config['temp_increase']
            motor_temp += temp_increase
        
        # Apply thermal stress effects
        if 'temp_rise_rate' in fault_config:
            temp_rate = random.uniform(*fault_config['temp_rise_rate'])
            motor_temp += temp_rate * (thermal_stress / 60)  # Convert thermal stress to minutes
        
        # Add accumulated thermal stress
        motor_temp += thermal_stress * 0.5
        
        # Add BME280 sensor noise
        sensor_noise = random.uniform(-self.sensors['bme280']['noise_temp'], 
                                    self.sensors['bme280']['noise_temp'])
        motor_temp += sensor_noise
        
        return max(ambient_temp, min(motor_temp, 85.0))  # Physical temperature limits
    
    def _calculate_vibration_components(self, speed_percent, fault_config, bearing_wear, current):
        """Calculate realistic 3-axis vibration with fault signatures"""
        
        if speed_percent == 0:
            # Stopped motor - only environmental noise
            base_magnitude = self.sensors['mpu6050']['vibration_noise_floor']
        else:
            # Operating motor - speed and load dependent vibration
            rpm = (speed_percent / 100) * self.motor_specs['max_speed_rpm']
            
            # Base vibration proportional to speed and current (load indicator)
            speed_vibration = 0.08 + (speed_percent / 100) * 0.25
            load_vibration = (current / 15.0) * 0.15
            base_magnitude = speed_vibration + load_vibration
        
        # Apply fault-specific vibration increases
        vibration_magnitude = base_magnitude
        
        if 'vibration_factor' in fault_config:
            if isinstance(fault_config['vibration_factor'], tuple):
                vib_factor = random.uniform(*fault_config['vibration_factor'])
            else:
                vib_factor = fault_config['vibration_factor']
            vibration_magnitude *= vib_factor
        
        # Add bearing wear progressive increase
        bearing_wear_contribution = bearing_wear * 35  # Amplify bearing wear effect
        vibration_magnitude += bearing_wear_contribution
        
        # Add frequency-specific components for bearing faults
        if 'frequency_signatures' in fault_config and speed_percent > 0:
            for freq in fault_config['frequency_signatures']:
                # Simulate frequency-specific vibration components
                freq_amplitude = 0.02 * random.uniform(0.5, 1.5)
                vibration_magnitude += freq_amplitude
        
        # Distribute vibration across 3 axes with realistic patterns
        # X-axis typically highest (motor mounting direction)
        # Y-axis moderate (perpendicular to mounting)
        # Z-axis lowest (vertical, dampened by mass)
        
        vibration_x = vibration_magnitude * random.uniform(0.7, 1.0)
        vibration_y = vibration_magnitude * random.uniform(0.5, 0.8) 
        vibration_z = vibration_magnitude * random.uniform(0.3, 0.6)
        
        # Add MPU6050 sensor noise to each axis
        noise_level = self.sensors['mpu6050']['vibration_noise_floor']
        vibration_x += random.uniform(-noise_level, noise_level)
        vibration_y += random.uniform(-noise_level, noise_level)
        vibration_z += random.uniform(-noise_level, noise_level)
        
        # Recalculate magnitude from components
        actual_magnitude = math.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2)
        
        return {
            'magnitude': max(0.005, min(actual_magnitude, 8.0)),
            'x': max(-4.0, min(vibration_x, 4.0)),
            'y': max(-4.0, min(vibration_y, 4.0)),
            'z': max(-4.0, min(vibration_z, 4.0))
        }
    
    def _calculate_motor_efficiency(self, speed_percent, current, temperature):
        """Calculate motor efficiency based on operating conditions"""
        if speed_percent == 0 or current < 0.5:
            return 0
        
        # Efficiency curve - peak around 70-80% speed
        speed_factor = speed_percent / 100.0
        speed_efficiency = 1 - 0.3 * (speed_factor - 0.75)**2
        speed_efficiency = max(0.6, min(speed_efficiency, 1.0))
        
        # Current efficiency - less efficient at very high currents
        current_efficiency = 1 - max(0, (current - 12) / 20) * 0.2
        current_efficiency = max(0.7, min(current_efficiency, 1.0))
        
        # Temperature derating
        temp_efficiency = max(0.6, 1 - max(0, (temperature - 45) / 40))
        
        # Base efficiency for this motor type
        base_efficiency = 82  # Percent
        
        total_efficiency = base_efficiency * speed_efficiency * current_efficiency * temp_efficiency
        return max(25, min(total_efficiency, 92))
    
    def _select_fault_scenario(self, bearing_wear, thermal_stress, electrical_stress, operating_hours):
        """Select fault scenario based on progressive wear conditions"""
        
        # Start with base probabilities
        fault_weights = {}
        for fault_type, config in self.fault_scenarios.items():
            weight = config['probability']
            
            # Modify probabilities based on progressive conditions
            if 'bearing_wear' in fault_type and bearing_wear > 0.05:
                weight *= (1 + bearing_wear * 15)  # Increase bearing fault probability
            
            if 'thermal' in fault_type and thermal_stress > 0.3:
                weight *= (1 + thermal_stress * 3)  # Increase thermal fault probability
            
            if 'electrical' in fault_type and electrical_stress > 0.2:
                weight *= (1 + electrical_stress * 4)  # Increase electrical fault probability
            
            # Operating hours effect (more faults with age)
            age_factor = min(2.0, 1 + operating_hours / 1000)  # Gradual increase with hours
            if fault_type != 'normal_operation':
                weight *= age_factor
            
            fault_weights[fault_type] = weight
        
        # Normalize weights and select
        total_weight = sum(fault_weights.values())
        normalized_weights = [fault_weights[ft] / total_weight for ft in fault_weights.keys()]
        
        return random.choices(list(fault_weights.keys()), weights=normalized_weights)[0]
    
    def _apply_sensor_drift(self, value, sensor_type, drift_level):
        """Apply sensor drift effects"""
        if sensor_type == 'temperature':
            drift_amount = drift_level * random.uniform(-2.0, 2.0)
        elif sensor_type == 'current':
            drift_amount = drift_level * random.uniform(-0.3, 0.3)
        else:
            drift_amount = 0
        
        return value + drift_amount
    
    def _classify_temperature_status(self, temperature):
        """Classify temperature status (matching your ESP32 code)"""
        if temperature > 40:
            return 'HIGH'
        elif temperature > 30:
            return 'WARM'
        else:
            return 'NORMAL'
    
    def _classify_vibration_status(self, vibration):
        """Classify vibration status (matching your ESP32 code)"""
        if vibration > 1.0:
            return 'HIGH'
        elif vibration > 0.3:
            return 'ACTIVE'
        else:
            return 'LOW'
    
    def _classify_current_status(self, current):
        """Classify current status for ACS712 readings"""
        if current > 15:
            return 'HIGH'
        elif current > 10:
            return 'ELEVATED'
        elif current > 2:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def _determine_motor_enabled_state(self, temperature, current, vibration):
        """Determine if motor should be enabled based on safety conditions"""
        # Safety thresholds (matching your ESP32 code)
        if temperature > 45:  # Emergency temperature limit
            return False
        if current > 18:  # Overcurrent protection
            return False
        if vibration > 2.0:  # Excessive vibration
            return False
        return True
    
    def _determine_motor_status(self, fault_type, motor_enabled, temperature, current):
        """Determine overall motor status"""
        if not motor_enabled:
            return 'emergency_stop'
        elif fault_type != 'normal_operation':
            if 'major' in fault_type or 'advanced' in fault_type:
                return 'fault_detected'
            else:
                return 'caution'
        elif temperature > 35 or current > 12:
            return 'caution'
        else:
            return 'normal'
    
    def _calculate_fault_severity(self, fault_type, temperature, current, vibration):
        """Calculate fault severity score (0-10 scale)"""
        severity = 0
        
        # Base severity by fault type
        fault_severity_map = {
            'normal_operation': 0,
            'bearing_wear_early': 2,
            'bearing_wear_advanced': 6,
            'electrical_fault_minor': 3,
            'electrical_fault_major': 7,
            'thermal_stress': 5,
            'mechanical_overload': 6,
            'power_supply_stress': 2,
            'sensor_drift': 1
        }
        
        severity = fault_severity_map.get(fault_type, 0)
        
        # Adjust based on actual readings
        if temperature > 45:
            severity += 3
        elif temperature > 40:
            severity += 2
        elif temperature > 35:
            severity += 1
        
        if current > 16:
            severity += 2
        elif current > 14:
            severity += 1
        
        if vibration > 2.0:
            severity += 3
        elif vibration > 1.5:
            severity += 2
        elif vibration > 1.0:
            severity += 1
        
        return min(10, severity)  # Cap at 10

def generate_and_save_dataset():
    """
    Main function to generate and save the complete dataset
    """
    print("=" * 60)
    print("🚀 COMPLETE MOTOR SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    print("📋 Student: Gourav Shaw (T0436800)")
    print("🎓 Project: MSc Dissertation - Motor Fault Detection")
    print("⚙️ Hardware: 12V DC Motor + BTS7960 + ESP32")
    print("🔌 Sensors: BME280 + MPU6050 + ACS712")
    print("=" * 60)
    print("")
    
    # Initialize generator
    generator = CompleteMotorDatasetGenerator()
    
    # Generate complete dataset
    print("🔄 Starting dataset generation...")
    dataset = generator.generate_complete_dataset(num_points=50000, time_span_days=42)
    
    print("💾 Saving dataset in multiple formats...")
    
    # Create output directory
    output_dir = "motor_fault_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save as CSV (for analysis and ML training)
    csv_filename = os.path.join(output_dir, "motor_fault_dataset.csv")
    dataset.to_csv(csv_filename, index=False)
    print(f"✅ CSV saved: {csv_filename}")
    
    # 2. Save as JSON (matching your MQTT format)
    json_data = dataset.to_dict(orient='records')
    json_filename = os.path.join(output_dir, "motor_fault_dataset.json")
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"✅ JSON saved: {json_filename}")
    
    # 3. Create MongoDB import script
    mongodb_script_content = f'''#!/usr/bin/env python3
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
        
        print(f"📥 Importing {{len(data):,}} records...")
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        # collection.delete_many({{}})
        # print("🗑️  Existing data cleared")
        
        # Insert new data
        result = collection.insert_many(data)
        
        print(f"✅ Successfully imported {{len(result.inserted_ids):,}} records")
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
        print("\\n📈 Dataset Summary:")
        total_records = collection.count_documents({{}})
        operating_records = collection.count_documents({{"motor_speed": {{"$gt": 0}}}})
        fault_records = collection.count_documents({{"fault_type": {{"$ne": "normal_operation"}}}})
        
        print(f"   Total records: {{total_records:,}}")
        print(f"   Operating records: {{operating_records:,}}")
        print(f"   Fault records: {{fault_records:,}}")
        print(f"   Fault percentage: {{(fault_records/total_records)*100:.1f}}%")
        
        print("\\n🎯 Import completed successfully!")
        print("🌐 Your dashboard should now show the synthetic data")
        
    except Exception as e:
        print(f"❌ Error importing data: {{e}}")
        return False
    
    return True

if __name__ == "__main__":
    success = import_to_mongodb()
    if success:
        print("\\n🚀 Ready for ML training and validation!")
    else:
        print("\\n❌ Import failed. Check MongoDB connection and try again.")
'''
    
    mongodb_script_filename = os.path.join(output_dir, "import_to_mongodb.py")
    with open(mongodb_script_filename, 'w') as f:
        f.write(mongodb_script_content)
    print(f"✅ MongoDB import script saved: {mongodb_script_filename}")
    
    # 4. Create ML training script template
    ml_script_content = '''#!/usr/bin/env python3
"""
ML Training Script for Motor Fault Detection
Generated for: Gourav Shaw (T0436800)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the synthetic dataset for ML training"""
    
    print("📊 Loading synthetic dataset...")
    df = pd.read_csv('motor_fault_dataset.csv')
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"📊 Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Feature selection for ML
    feature_columns = [
        'temperature', 'humidity', 'pressure',
        'vibration', 'vibration_x', 'vibration_y', 'vibration_z',
        'current', 'voltage', 'power_consumption',
        'motor_speed', 'motor_efficiency',
        'bearing_wear_level', 'thermal_stress_level', 'electrical_stress_level'
    ]
    
    # Target variable
    target_column = 'fault_type'
    
    # Filter operating data only (exclude stopped motor data for fault detection)
    operating_data = df[df['motor_speed'] > 0].copy()
    
    print(f"📊 Operating data: {len(operating_data):,} records")
    
    X = operating_data[feature_columns]
    y = operating_data[target_column]
    
    return X, y, operating_data

def train_anomaly_detection():
    """Train anomaly detection model"""
    
    print("\\n🤖 Training Anomaly Detection Model...")
    
    X, y, data = load_and_prepare_data()
    
    # Use only normal operation data for anomaly detection training
    normal_data = X[y == 'normal_operation']
    
    print(f"📊 Normal operation samples: {len(normal_data):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(normal_data)
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    isolation_forest.fit(X_normal_scaled)
    
    # Test on full dataset
    X_all_scaled = scaler.transform(X)
    anomaly_predictions = isolation_forest.predict(X_all_scaled)
    
    # -1 = anomaly, 1 = normal
    anomaly_score = (anomaly_predictions == -1).sum() / len(anomaly_predictions)
    print(f"📊 Detected {anomaly_score*100:.1f}% anomalies")
    
    # Save models
    joblib.dump(isolation_forest, 'anomaly_detection_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("✅ Anomaly detection model saved")
    
    return isolation_forest, scaler

def train_fault_classification():
    """Train fault classification model"""
    
    print("\\n🎯 Training Fault Classification Model...")
    
    X, y, data = load_and_prepare_data()
    
    # Encode fault types
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("📊 Fault type distribution:")
    fault_counts = pd.Series(y).value_counts()
    for fault, count in fault_counts.items():
        print(f"   {fault}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_classifier.predict(X_test_scaled)
    
    print("\\n📊 Classification Results:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\n🔍 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save models
    joblib.dump(rf_classifier, 'fault_classification_model.pkl')
    joblib.dump(label_encoder, 'fault_label_encoder.pkl')
    joblib.dump(scaler, 'classification_scaler.pkl')
    
    print("✅ Classification model saved")
    
    return rf_classifier, label_encoder, scaler

if __name__ == "__main__":
    print("🚀 Starting ML Model Training...")
    print("=" * 50)
    
    # Train both models
    anomaly_model, anomaly_scaler = train_anomaly_detection()
    classification_model, label_encoder, classification_scaler = train_fault_classification()
    
    print("\\n✅ ML Training Complete!")
    print("🎯 Models ready for deployment and validation")
    print("📁 Files created:")
    print("   • anomaly_detection_model.pkl")
    print("   • fault_classification_model.pkl")
    print("   • feature_scaler.pkl")
    print("   • classification_scaler.pkl") 
    print("   • fault_label_encoder.pkl")
'''
    
    ml_script_filename = os.path.join(output_dir, "train_ml_models.py")
    with open(ml_script_filename, 'w') as f:
        f.write(ml_script_content)
    print(f"✅ ML training script saved: {ml_script_filename}")
    
    # 5. Generate comprehensive statistics
    print(f"\\n📊 DATASET STATISTICS:")
    print(f"=" * 40)
    
    # Basic statistics
    print(f"Total records: {len(dataset):,}")
    print(f"Operating records: {len(dataset[dataset['motor_speed'] > 0]):,}")
    print(f"Idle records: {len(dataset[dataset['motor_speed'] == 0]):,}")
    print(f"Time span: {dataset['created_at'].min()} to {dataset['created_at'].max()}")
    
    # Fault distribution
    print(f"\\n🔍 Fault Type Distribution:")
    fault_distribution = dataset['fault_type'].value_counts()
    for fault_type, count in fault_distribution.items():
        percentage = (count / len(dataset)) * 100
        print(f"   {fault_type}: {count:,} ({percentage:.1f}%)")
    
    # Operating conditions
    operating_data = dataset[dataset['motor_speed'] > 0]
    if len(operating_data) > 0:
        print(f"\\n⚙️ Operating Conditions:")
        print(f"   Temperature: {operating_data['temperature'].min():.1f}°C to {operating_data['temperature'].max():.1f}°C")
        print(f"   Current: {operating_data['current'].min():.2f}A to {operating_data['current'].max():.2f}A") 
        print(f"   Vibration: {operating_data['vibration'].min():.3f} to {operating_data['vibration'].max():.3f} m/s²")
        print(f"   Power: {operating_data['power_consumption'].min():.1f}W to {operating_data['power_consumption'].max():.1f}W")
    
    # Status distributions
    print(f"\\n🚨 Status Classifications:")
    print(f"   Temperature Status: {dict(dataset['temp_status'].value_counts())}")
    print(f"   Vibration Status: {dict(dataset['vib_status'].value_counts())}")
    print(f"   Current Status: {dict(dataset['current_status'].value_counts())}")
    
    print(f"\\n" + "=" * 60)
    print(f"✅ DATASET GENERATION COMPLETE!")
    print(f"=" * 60)
    print(f"📁 Output directory: {output_dir}/")
    print(f"📊 Files created:")
    print(f"   • motor_fault_dataset.csv (for analysis)")
    print(f"   • motor_fault_dataset.json (MQTT format)")
    print(f"   • import_to_mongodb.py (database import)")
    print(f"   • train_ml_models.py (ML training)")
    print(f"")
    print(f"🚀 NEXT STEPS:")
    print(f"   1. Run: python {mongodb_script_filename}")
    print(f"   2. Check your dashboard - synthetic data should appear")
    print(f"   3. Run: python {ml_script_filename}")
    print(f"   4. Collect 12-15 hours real validation data")
    print(f"   5. Compare synthetic vs real patterns")
    print(f"")
    print(f"🎯 Your 50,000-point synthetic dataset is ready!")
    print(f"💡 Perfect for MSc dissertation validation approach")
    
    return dataset

# Run the complete generation
if __name__ == "__main__":
    dataset = generate_and_save_dataset()