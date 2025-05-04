#%%
import numpy as np
import time
import threading
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from influxdb import InfluxDBClient

# Step 1: Connect to InfluxDB
client = InfluxDBClient(host='localhost', port=8086, username='saran', password='root', database='power_grid')
#%%
# Step 2: Generate Synthetic 3-Phase Data
def generate_3phase_data():
    """Simulates real-time 3-phase power system data."""
    phase_voltage = [np.random.uniform(220, 250) for _ in range(3)]
    phase_current = [np.random.uniform(0, 50) for _ in range(3)]
    neutral_current = abs(phase_current[0] - phase_current[1] + phase_current[2]) / 3
    frequency = np.random.uniform(49.5, 50.5)
    return phase_voltage + phase_current + [neutral_current, frequency]

# Step 3: Generate Training Data
def generate_3phase_fault_data(num_samples=1000):
    """Generates labeled fault data."""
    x_train, y_train = [], []
    for _ in range(num_samples):
        fault_type = np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])  # More normal data
        phase_voltage = [np.random.uniform(220, 250) for _ in range(3)]
        phase_current = [np.random.uniform(0, 50) for _ in range(3)]

        if fault_type == 1:  # LG Fault
            phase_voltage[0] = np.random.uniform(0, 50)
        elif fault_type == 2:  # LL Fault
            phase_voltage[0] = phase_voltage[1] = np.random.uniform(0, 50)
        elif fault_type == 3:  # LLLG Fault
            phase_voltage = [np.random.uniform(0, 50) for _ in range(3)]

        neutral_current = abs(phase_current[0] - phase_current[1] + phase_current[2]) / 3
        frequency = np.random.uniform(49.5, 50.5)
        x_train.append(phase_voltage + phase_current + [neutral_current, frequency])
        y_train.append(fault_type)

    return np.array(x_train), np.array(y_train)
#%%
# Step 4: Train ANN Model
x_train, y_train = generate_3phase_fault_data()
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
y_train = to_categorical(y_train, num_classes=4)

def create_fault_classifier():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')  # Multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_fault_classifier()
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
#%%
# Step 5: Save Data to InfluxDB Without Blocking Classification
def save_to_influxdb(data, fault_type):
    """Thread-safe function to store data in InfluxDB."""
    json_body = [{
        "measurement": "grid_data",
        "tags": {"fault_type": str(fault_type)},
        "fields": {
            "voltage_R": data[0], "voltage_Y": data[1], "voltage_B": data[2],
            "current_R": data[3], "current_Y": data[4], "current_B": data[5],
            "neutral_current": data[6], "frequency": data[7]
        }
    }]
    threading.Thread(target=lambda: client.write_points(json_body)).start()

# Step 6: Real-Time Fault Classification
def classify_faults_in_real_time():
    """Continuously classify faults while storing data."""
    try:
        while True:
            data = generate_3phase_data()
            data_normalized = scaler.transform(np.expand_dims(data, axis=0))
            prediction = model.predict(data_normalized)[0]
            fault_type = np.argmax(prediction)

            # Print Fault Type
            fault_messages = {
                0: "✅ Normal Condition",
                1: "⚠️ Line-to-Ground (LG) Fault Detected!",
                2: "⚠️ Line-to-Line (LL) Fault Detected!",
                3: "⚠️ Three-Phase (LLLG) Fault Detected!"
            }
            print(fault_messages[fault_type])
            print(f"Real-Time Data: {data}")

            # Save data asynchronously to InfluxDB
            save_to_influxdb(data, fault_type)

            time.sleep(1)  # Simulate real-time processing

    except KeyboardInterrupt:
        print("\n🚨 Real-time fault detection terminated.")

# Start real-time fault classification
classify_faults_in_real_time()
#%%
import gym
from gym import spaces

class GridFaultEnv(gym.Env):
    """Custom RL environment for grid fault detection."""
    def __init__(self):
        super(GridFaultEnv, self).__init__()
        
        # Action space: Adjust fault detection thresholds
        self.action_space = spaces.Box(low=0.1, high=1.0, shape=(4,), dtype=np.float32)  # Thresholds for [Normal, LG, LL, LLLG]

        # Observation space: Real-time grid data
        self.observation_space = spaces.Box(low=0, high=300, shape=(8,), dtype=np.float32)  # 8 features: voltages, currents, etc.

        # Initialize thresholds
        self.thresholds = [0.5, 0.5, 0.5, 0.5]  # Default thresholds

    def step(self, action):
        """Adjust thresholds and calculate reward."""
        self.thresholds = action
        # Fetch real-time grid data from InfluxDB
        results = client.query("SELECT * FROM grid_data ORDER BY time DESC LIMIT 1")
        latest_data = list(results.get_points())[0]
        observation = [
            latest_data["voltage_R"], latest_data["voltage_Y"], latest_data["voltage_B"],
            latest_data["current_R"], latest_data["current_Y"], latest_data["current_B"],
            latest_data["neutral_current"], latest_data["frequency"]
        ]

        # Calculate reward based on fault detection accuracy
        fault_type = np.argmax(model.predict(np.expand_dims(observation, axis=0))[0])
        reward = 1.0 if fault_type == self.calculate_true_fault(observation) else -1.0

        # Done condition (optional)
        done = False  # Continue indefinitely

        return np.array(observation), reward, done, {}

    def reset(self):
        """Reset environment and return initial observation."""
        return np.zeros(8)  # Reset to zero state

    def calculate_true_fault(self, observation):
        """Simulate true fault type based on raw data."""
        # Example logic for true fault (replace with actual ground truth)
        if observation[0] < 50 and observation[1] > 210 and observation[2] > 210:
            return 1  # LG Fault
        elif observation[0] < 50 and observation[1] < 50 and observation[2] > 210:
            return 2  # LL Fault
        elif observation[0] < 50 and observation[1] < 50 and observation[2] < 50:
            return 3  # LLLG Fault
        else:
            return 0  # Normal Condition