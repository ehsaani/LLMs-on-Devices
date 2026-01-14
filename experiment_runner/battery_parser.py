import statistics

battery_log_path = "run_logcat.txt"

def battery_parser(battery_log_path, llama_metrics=None):
    # Initialize default empty dict if not provided
    if llama_metrics is None:
        llama_metrics = {}

    # Initialize lists and accumulators before the loop
    currents_A = []
    voltages_V = []
    power_readings = []
    capacities_pct = []
    temps_c = []
    
    total_energy_joules = 0.0
    prev_time = None
    prev_power = 0.0

    try:
        # Open the file safely
        with open(battery_log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if "BatteryMgr:DataCollectionService: stats =>" in line:
                    try:
                        csv_part = line.split("stats => ")[1].strip()
                        parts = csv_part.split(",")
                        
                        ts = int(parts[0])
                        curr_raw = int(parts[1]) # Unit: µA
                        volt_mV = int(parts[2])  # Unit: mV
                        cap_pct = int(parts[3])  # Unit: %
                        temp_raw = int(parts[4]) # Unit: Tenths of °C

                        # --- UNIT CONVERSION ---
                        time_sec = ts / 1000.0
                        
                        # Current: µA -> Amps. Subtract Baseline (0.10A approx).
                        current_A = max(0, (abs(curr_raw) / 1000000.0) - 0.10)
                        
                        # Voltage: mV -> Volts
                        voltage_V = volt_mV / 1000.0

                        # Temp: Tenths -> Degrees
                        temp_C_val = temp_raw / 10.0  
                        
                        power_W = current_A * voltage_V

                        currents_A.append(current_A)
                        voltages_V.append(voltage_V)
                        power_readings.append(power_W)
                        capacities_pct.append(cap_pct)
                        temps_c.append(temp_C_val)

                        # Trapezoidal Integration
                        if prev_time is not None:
                            dt = time_sec - prev_time
                            if dt > 0:
                                avg_p = (power_W + prev_power) / 2
                                total_energy_joules += avg_p * dt
                        
                        prev_time = time_sec
                        prev_power = power_W
                        
                    except (ValueError, IndexError):
                        continue

    except FileNotFoundError:
        print(f"Error: File '{battery_log_path}' not found.")
        return {}

    # --- 4. Aggregate Results ---
    # These must be calculated outside the loop
    avg_current = statistics.mean(currents_A) if currents_A else 0
    avg_voltage = statistics.mean(voltages_V) if voltages_V else 0
    avg_power = statistics.mean(power_readings) if power_readings else 0
    avg_capacity = statistics.mean(capacities_pct) if capacities_pct else 0
    min_battery = min(capacities_pct) if capacities_pct else 0
    max_battery = max(capacities_pct) if capacities_pct else 0
    avg_temp = statistics.mean(temps_c) if temps_c else 0
    min_temp = min(temps_c) if temps_c else 0
    max_temp = max(temps_c) if temps_c else 0

    # Energy Per Token
    gen_tokens = llama_metrics.get('output_token_count', 0)
    energy_per_token = total_energy_joules / gen_tokens if gen_tokens > 0 else 0

    # Return Combined Data
    return {
        # Energy & Device Stats
        'avg_current': round(avg_current, 6), # Fixed missing comma
        'avg_voltage': round(avg_voltage, 4),
        'avg_power': round(avg_power, 4),
        'total_energy_consumption': round(total_energy_joules, 4),
        'energy_per_token': round(energy_per_token, 4),
        'battery_capacity': round(avg_capacity, 2),
        'min_battery_capacity': round(min_battery, 2),
        'max_battery_capacity': round(max_battery, 2),
        'average_temperature': round(avg_temp, 2),
        'min_temperature': round(min_temp, 2),
        'max_temperature': round(max_temp, 2)
    }

# Usage Example
if __name__ == "__main__":
    battery_log_path = "run_logcat.txt"
    # Mock metrics for testing
    metrics = {'output_token_count': 100} 
    result = battery_parser(battery_log_path, metrics)
    print(result)