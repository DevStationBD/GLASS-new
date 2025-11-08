# config.py
"""
Configuration file for Fabric Inspector
"""

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Training Settings
MIN_IMAGES_FOR_TRAINING = 100  # Minimum images before training is allowed
TRAINING_EPOCHS = 10  # Number of training epochs (adjust for your model)
SIMULATION_TRAIN_TIME = 12  # Seconds to simulate training

# Dataset Settings
DATASETS_DIR = "raw_data"
INFERENCE_DIR = "inference"
IMAGE_FORMAT = "png"
SERVER_DATASETS_DIR = "/root/TrainingHost/GLASS-new/raw-data"
SERVER_SSH_KEY = "/root/.ssh/id_rsa"
SERVER_IP = "139.59.4.173"
SERVER_TRAINING_API = "http://139.59.4.173:5000/api/start-training"
SERVER_MODELS_API = "http://139.59.4.173:5000/api/models"
SERVER_RESULTS_DIR = "/root/TrainingHost/GLASS-new/results"
MODEL_SYNC_INTERVAL = 300  # Sync models every 5 minutes (300 seconds)
# Inference Settings
INFERENCE_FPS = 10  # Frames per second during inference
AUTO_DETECT_SIMILARITY_THRESHOLD = 0.6  # Threshold for model auto-detection

# Serial Port Settings for Custom Input Device
SERIAL_ENABLED = False  # Set to True to enable serial port listening
SERIAL_PORT = "/dev/ttyUSB0"  # Serial port device (Linux: /dev/ttyUSB0, Windows: COM3)
SERIAL_BAUDRATE = 9600  # Baud rate
SERIAL_TIMEOUT = 1  # Timeout in seconds
SERIAL_DELIMITER = b'\n'  # Command delimiter (usually \n or \r\n)

# Serial Commands (multi-character strings)
# Commands are sent as strings followed by delimiter (e.g., "capture\n")
SERIAL_CMD_CAPTURE = "capture"     # Capture image command
SERIAL_CMD_TRAIN = "train"         # Start training command
SERIAL_CMD_INFERENCE = "inference" # Start inference command
SERIAL_CMD_MENU = "menu"           # Return to menu command
SERIAL_CMD_QUIT = "quit"           # Quit application command

# You can also use custom function names
# SERIAL_CMD_CAPTURE = "func1"
# SERIAL_CMD_TRAIN = "func2"
# etc.

# Key Bindings File
KEY_BINDINGS_FILE = "keys.json"

# UI Settings
WINDOW_THEME = "default"  # Options: "default", "dark", "light"
SHOW_FPS = True
ENABLE_AUDIO_ALERTS = False

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "fabric_inspector.log"
ENABLE_CLOUD_LOGGING = False

# Security
CREDENTIALS_FILE = ".credentials"  # Store encrypted credentials here
ENABLE_SSL = True
API_KEY_REQUIRED = False
