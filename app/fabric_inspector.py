#!/usr/bin/env python3
"""
Fabric Inspector - Industrial defect detection application
OpenCV GUI with configurable key bindings and serial port support
"""

import cv2
import numpy as np
import json
import time
import threading
import sys
import os
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import config

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fabric_inspector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add GLASS inference path
GLASS_INFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inference')
GLASS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models', 'backbone_0')

logger.info(f"Fabric Inspector starting...")
logger.debug(f"Current file: {os.path.abspath(__file__)}")
logger.debug(f"GLASS_INFERENCE_PATH: {GLASS_INFERENCE_PATH}")
logger.debug(f"GLASS_MODEL_PATH: {GLASS_MODEL_PATH}")
logger.debug(f"GLASS_INFERENCE_PATH exists: {os.path.exists(GLASS_INFERENCE_PATH)}")
logger.debug(f"GLASS_MODEL_PATH exists: {os.path.exists(GLASS_MODEL_PATH)}")

if os.path.exists(GLASS_INFERENCE_PATH):
    sys.path.insert(0, GLASS_INFERENCE_PATH)
    GLASS_AVAILABLE = True
    logger.info("GLASS inference found and loaded")
else:
    GLASS_AVAILABLE = False
    logger.warning("GLASS inference not found. Using simulation mode.")
    print("Warning: GLASS inference not found. Using simulation mode.")


class KeyBindings:
    """Load and manage key bindings from JSON file"""
    
    def __init__(self, json_file="keys.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.keys = data['keys']
        except FileNotFoundError:
            print(f"Warning: {json_file} not found, using defaults")
            self.keys = {
                'menu': {'training_mode': 't', 'inference_mode': 'i', 'quit': 'q'},
                'training': {'capture': 'c', 'submit': 's', 'back_to_menu': 'm'},
                'inference': {'reselect_model': 'r', 'back_to_menu': 'm'},
                'review': {'next': 'n', 'previous': 'p', 'discard': 'd', 'confirm': 's', 'back': 'b'}
            }
    
    def get(self, mode, action):
        """Get key for mode and action"""
        return self.keys.get(mode, {}).get(action, '').lower()


class SerialPortListener:
    """Listen for commands on serial port"""
    
    def __init__(self, port, baudrate, callback):
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.running = False
        self.thread = None
        
        try:
            import serial
            self.serial = serial
            self.enabled = True
        except ImportError:
            print("Warning: pyserial not installed. Serial port disabled.")
            self.enabled = False
    
    def start(self):
        """Start listening thread"""
        if not self.enabled or self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"Serial port listener started on {self.port}")
    
    def stop(self):
        """Stop listening thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            ser = self.serial.Serial(
                self.port,
                self.baudrate,
                timeout=config.SERIAL_TIMEOUT
            )
            
            buffer = b''
            while self.running:
                if ser.in_waiting > 0:
                    # Read available data
                    chunk = ser.read(ser.in_waiting)
                    buffer += chunk
                    
                    # Check for delimiter
                    if config.SERIAL_DELIMITER in buffer:
                        # Split by delimiter
                        parts = buffer.split(config.SERIAL_DELIMITER)
                        # Process all complete commands
                        for command in parts[:-1]:
                            if command:
                                try:
                                    command_str = command.decode('utf-8').strip()
                                    if command_str and self.callback:
                                        self.callback(command_str)
                                except UnicodeDecodeError:
                                    print(f"Warning: Invalid serial data: {command}")
                        # Keep incomplete part in buffer
                        buffer = parts[-1]
                
                time.sleep(0.01)
            
            ser.close()
        except Exception as e:
            print(f"Serial port error: {e}")
            self.running = False


class FabricInspector:
    """Main application class"""
    
    def __init__(self):
        self.camera = None
        self.current_frame = None
        self.mode = "menu"  # menu, train, test, review, model_select
        
        # Load key bindings
        self.keys = KeyBindings(config.KEY_BINDINGS_FILE)
        
        # Training state
        self.captured_images = []
        self.capture_count = 0
        self.batch_id = None
        self.training_in_progress = False
        self.training_logs = []
        
        # Upload state
        self.upload_in_progress = False
        self.upload_status = ""
        self.upload_complete = False
        
        # Review state
        self.review_index = 0
        self.images_to_discard = set()
        
        # Testing state
        self.selected_model = None
        self.available_models = []
        self.model_previews = {}
        self.model_selection_index = 0
        
        # Model sync state
        self.model_sync_thread = None
        self.model_sync_running = True
        self.last_model_sync = 0
        
        # UI state
        self.window_name = "Fabric Inspector"
        self.screen_width = 1024  # Default, will be updated in start()
        self.screen_height = 768  # Default, will be updated in start()
        
        # Serial port
        self.serial_listener = None
        if config.SERIAL_ENABLED:
            self.serial_listener = SerialPortListener(
                config.SERIAL_PORT,
                config.SERIAL_BAUDRATE,
                self.handle_serial_command
            )
    
    def start(self):
        """Start the application"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Get screen resolution and maximize window
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Set window to full screen size
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.moveWindow(self.window_name, 0, 0)
        
        print("=" * 60)
        print("Fabric Inspector")
        print("=" * 60)
        print("\nKeyboard Controls:")
        print(f"  {self.keys.get('menu', 'training_mode').upper()} - Training Mode")
        print(f"  {self.keys.get('menu', 'inference_mode').upper()} - Inference Mode")
        print(f"  {self.keys.get('menu', 'quit').upper()} - Quit")
        
        if config.SERIAL_ENABLED:
            print("\nSerial Port Commands:")
            print(f"  '{config.SERIAL_CMD_CAPTURE}' - Capture Image")
            print(f"  '{config.SERIAL_CMD_TRAIN}' - Training Mode")
            print(f"  '{config.SERIAL_CMD_INFERENCE}' - Inference Mode")
            print(f"  '{config.SERIAL_CMD_MENU}' - Menu")
            print(f"  '{config.SERIAL_CMD_QUIT}' - Quit")
        
        print("=" * 60)
        
        # Start serial listener if enabled
        if self.serial_listener:
            self.serial_listener.start()
        
        # Start model sync thread
        self.model_sync_running = True
        self.model_sync_thread = threading.Thread(target=self.sync_models_loop, daemon=True)
        self.model_sync_thread.start()
        print("Model sync thread started (5-minute interval)")
        
        self.run_main_loop()
    
    def handle_serial_command(self, command):
        """
        Handle command from serial port.
        
        Args:
            command: String command received from serial port (e.g., "capture", "func1")
        """
        print(f"Serial command received: '{command}'")
        
        if command == config.SERIAL_CMD_CAPTURE:
            if self.mode == "train":
                self.capture_image()
            else:
                print(f"  Warning: Capture command only works in training mode")
        elif command == config.SERIAL_CMD_TRAIN:
            self.mode = "train"
            self.start_training_mode()
        elif command == config.SERIAL_CMD_INFERENCE:
            self.mode = "test"
            self.start_testing_mode()
        elif command == config.SERIAL_CMD_MENU:
            self.mode = "menu"
            self.cleanup_mode()
        elif command == config.SERIAL_CMD_QUIT:
            self.cleanup()
        else:
            print(f"  Warning: Unknown serial command '{command}'")
    
    def run_main_loop(self):
        """Main application loop"""
        try:
            while True:
                # Get current frame based on mode
                if self.mode == "menu":
                    frame = self.draw_menu()
                elif self.mode == "training":
                    frame = self.draw_training()
                elif self.mode == "testing":
                    frame = self.draw_testing()
                elif self.mode == "model_selection":
                    frame = self.draw_model_selection()
                elif self.mode == "review":
                    frame = self.draw_review()
                elif self.mode == "upload":
                    frame = self.draw_upload_status()
                else:
                    frame = self.draw_menu()
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                
                if not self.handle_keyboard_input(key):
                    break
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        finally:
            self.cleanup()
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input based on current mode"""
        if key == ord('q') or key == 27:  # 'q' or ESC key
            return False  # Exit main loop
        
        key_char = chr(key) if key < 256 else None
        
        if self.mode == "menu":
            if key_char == 't' or key_char == 'T':
                self.mode = "training"
                self.start_training_mode()
            elif key_char == 'i' or key_char == 'I':
                self.mode = "model_selection"
                self.load_available_models()
            elif key_char == 'q' or key_char == 'Q':
                return False
        
        elif self.mode == "training":
            if key_char == 'c' or key_char == 'C':
                self.capture_image()
            elif key_char == 's' or key_char == 'S':
                if self.capture_count >= config.MIN_IMAGES_FOR_TRAINING:
                    self.submit_training()
            elif key_char == 'r' or key_char == 'R':
                self.mode = "review"
            elif key_char == 'm' or key_char == 'M':
                self.mode = "menu"
                self.cleanup_mode()
        
        elif self.mode == "testing":
            if key_char == 'r' or key_char == 'R':
                if hasattr(self, 'selected_model_class'):
                    self.load_available_models()
            elif key_char == 'm' or key_char == 'M':
                self.mode = "menu"
                self.cleanup_mode()
        
        elif self.mode == "model_selection":
            if key_char == 'n' or key_char == 'N':
                self.model_selection_next()
            elif key_char == 'p' or key_char == 'P':
                self.model_selection_previous()
            elif key_char == 's' or key_char == 'S':
                self.confirm_model_selection()
            elif key_char == 'b' or key_char == 'B':
                self.mode = "menu"
        
        elif self.mode == "review":
            if key_char == 'n' or key_char == 'N':
                self.review_next_image()
            elif key_char == 'p' or key_char == 'P':
                self.review_previous_image()
            elif key_char == 'd' or key_char == 'D':
                self.review_discard_current()
            elif key_char == 's' or key_char == 'S':
                self.review_confirm_submit()
            elif key_char == 'b' or key_char == 'B':
                self.mode = "training"
                self.images_to_discard.clear()
        
        elif self.mode == "upload":
            if key_char == 'm' or key_char == 'M':
                self.mode = "menu"
                self.upload_in_progress = False
                self.upload_complete = False
                self.upload_status = ""
        
        return True  # Continue main loop
    
    def draw_menu(self):
        """Draw main menu"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Title - centered
        title_text = "FABRIC INSPECTOR"
        title_x = (self.screen_width - len(title_text) * 25) // 2  # Approximate centering
        cv2.putText(frame, title_text, (title_x, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        
        # Options
        y = 250
        training_key = self.keys.get('menu', 'training_mode').upper()
        inference_key = self.keys.get('menu', 'inference_mode').upper()
        quit_key = self.keys.get('menu', 'quit').upper()
        
        # Get actual model count from GLASS model path
        models_dir = Path(GLASS_MODEL_PATH)
        model_count = 0
        if models_dir.exists():
            model_dirs = [p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
            new_model_count = len(model_dirs)
            if new_model_count != model_count:
                logger.debug(f"Menu: Found {new_model_count} models in {models_dir}: {[p.name for p in model_dirs]}")
            model_count = new_model_count
        else:
            logger.warning(f"Menu: Models directory does not exist: {models_dir}")
        
        options = [
            f"Press '{training_key}' - Training Mode (Capture & Train)",
            f"Press '{inference_key}' - Inference Mode (Test Models)",
            f"Press '{quit_key}' - Quit Application",
            "",
            f"Datasets: {len(list(Path(config.DATASETS_DIR).glob('batch_*')))} available",
            f"Models: {model_count} available",
        ]
        
        for text in options:
            if text:
                color = (100, 255, 100) if "Press" in text else (200, 200, 200)
                cv2.putText(frame, text, (100, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y += 50
        
        # Footer
        footer_text = "OpenCV GUI"
        if config.SERIAL_ENABLED:
            footer_text += f" | Serial: {config.SERIAL_PORT}"
        cv2.putText(frame, footer_text, (300, 700), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        return frame
    
    def configure_v4l2_settings(self, camera_index=None):
        """Configure camera using v4l2-ctl for high quality image capture"""
        try:
            if camera_index is None:
                camera_index = config.CAMERA_INDEX
            
            device = f"/dev/video{camera_index}"
            logger.info(f"🔧 Configuring camera via v4l2-ctl on {device}")
            
            # High-quality v4l2 settings optimized for fabric inspection
            v4l2_settings = {
                'auto_exposure': 1,              # Manual exposure mode
                'exposure_time_absolute': 5,     # Balanced shutter for sharp images
                'gain': 255,                     # Maximum gain for fast shutter
                'brightness': 200,               # High brightness compensation
                'contrast': 200,                 # Maximum contrast for detail
                'sharpness': 255,               # Maximum sharpness for fabric detail
                'saturation': 100,              # Standard saturation
                'focus_automatic_continuous': 0, # Manual focus for consistency
                'focus_absolute': 50,            # Medium focus distance
                'white_balance_automatic': 0,    # Manual white balance
                'white_balance_temperature': 4000, # Neutral white balance
                'power_line_frequency': 1,       # 60Hz (avoid flicker)
                'backlight_compensation': 0      # Disable backlight compensation
            }
            
            # Apply each setting
            for setting, value in v4l2_settings.items():
                try:
                    cmd = ['v4l2-ctl', '-d', device, '--set-ctrl', f'{setting}={value}']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        logger.debug(f"  ✅ {setting}: {value}")
                    else:
                        logger.warning(f"  ⚠️  {setting}: {value} - {result.stderr.strip()}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"  ❌ {setting}: timeout")
                except FileNotFoundError:
                    logger.error("v4l2-ctl not found - install with: sudo apt install v4l-utils")
                    return False
                except Exception as e:
                    logger.error(f"  ❌ {setting}: {e}")
            
            logger.info("✅ V4L2 camera configuration completed")
            return True
            
        except Exception as e:
            logger.error(f"V4L2 configuration failed: {e}")
            return False
    
    def initialize_high_quality_camera(self, camera_index=None):
        """Initialize camera with high-quality settings for fabric inspection"""
        try:
            if camera_index is None:
                camera_index = config.CAMERA_INDEX
            
            # Configure camera using v4l2-ctl first for precise control
            if not self.configure_v4l2_settings(camera_index):
                logger.warning("V4L2 configuration failed, using OpenCV settings only")
            
            # Use V4L2 backend for best performance
            camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            
            if not camera.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return None
            
            # Optimizations for high-quality capture
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            
            # Set high resolution and format
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for sharper motion capture
            
            # Verify actual settings
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = camera.get(cv2.CAP_PROP_FPS)
            
            logger.info("✅ High-quality camera initialized:")
            logger.info(f"  Resolution: {actual_width}x{actual_height}")
            logger.info(f"  FPS: {actual_fps}")
            logger.info(f"  Using v4l2-ctl settings for optimal image quality")
            
            return camera
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return None
    
    def start_training_mode(self):
        """Initialize training mode with high-quality camera settings"""
        logger.info("🎯 Starting training mode with high-quality camera settings...")
        
        # Use high-quality camera initialization
        self.camera = self.initialize_high_quality_camera()
        
        if self.camera is None:
            print("Error: Could not open camera with high-quality settings")
            logger.error("Failed to initialize high-quality camera for training")
            self.mode = "menu"
            return
        
        self.capture_count = 0
        self.captured_images = []
        self.batch_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        self.training_logs = ["Training mode started", f"Batch ID: {self.batch_id}"]
        
        print(f"\nTraining Mode Active - Batch: {self.batch_id}")
        print(f"Minimum images required: {config.MIN_IMAGES_FOR_TRAINING}")
    
    def draw_training(self):
        """Draw training interface"""
        if self.camera is None or not self.camera.isOpened():
            return self.draw_menu()
        
        ret, frame = self.camera.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.current_frame = frame.copy()
            
            # Draw overlay
            overlay = frame.copy()
            
            # Top bar
            cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "TRAINING MODE", (20, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Captured: {self.capture_count} / {config.MIN_IMAGES_FOR_TRAINING}", 
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Progress bar
            progress = min(self.capture_count / config.MIN_IMAGES_FOR_TRAINING, 1.0)
            bar_width = int(620 * progress)
            cv2.rectangle(frame, (10, 460), (630, 470), (50, 50, 50), -1)
            if bar_width > 0:
                color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
                cv2.rectangle(frame, (10, 460), (10 + bar_width, 470), color, -1)
            
            # Instructions
            y = 400
            capture_key = self.keys.get('training', 'capture').upper()
            submit_key = self.keys.get('training', 'submit').upper()
            menu_key = self.keys.get('training', 'back_to_menu').upper()
            
            instructions = [
                f"Press '{capture_key}' - Capture Image",
                f"Press '{submit_key}' - Submit & Train" if self.capture_count >= config.MIN_IMAGES_FOR_TRAINING else "",
                f"Press '{menu_key}' - Back to Menu",
            ]
            
            for text in instructions:
                if text:
                    cv2.putText(frame, text, (10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 25
        
        # Resize to full window
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        
        # Add training logs if any
        if self.training_logs:
            log_y = 680
            for log in self.training_logs[-3:]:
                cv2.putText(frame, log[:80], (20, log_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
                log_y += 25
        
        return frame
    
    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is None:
            return
        
        # Create dataset directory
        dataset_dir = Path(config.DATASETS_DIR) / f"batch_{self.batch_id}/good-images"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image with high quality settings
        img_path = dataset_dir / f"img_{self.capture_count + 1:04d}.{config.IMAGE_FORMAT}"
        
        # High-quality JPEG settings (95% quality, same as capture_fhd_old.py)
        jpeg_quality = 95
        success = cv2.imwrite(str(img_path), self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        
        if success:
            # Get file size for logging
            file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
            logger.info(f"💾 High-quality image saved: {img_path.name} ({file_size:.1f} MB, {jpeg_quality}% quality)")
        else:
            logger.error(f"❌ Failed to save image: {img_path}")
            return
        
        self.capture_count += 1
        self.captured_images.append(str(img_path))
        
        log = f"Captured image {self.capture_count}"
        self.training_logs.append(log)
        print(f"  {log}")
        
        if self.capture_count >= config.MIN_IMAGES_FOR_TRAINING:
            ready_key = self.keys.get('training', 'submit').upper()
            self.training_logs.append(f"Ready to train! Press '{ready_key}' to submit")
    
    def submit_training(self):
        """Enter review mode to review captured images"""
        if self.training_in_progress:
            return
        
        print(f"\nEntering review mode - {self.capture_count} images captured")
        self.training_logs.append("Entering review mode...")
        
        # Initialize review state
        self.review_index = 0
        self.images_to_discard.clear()
        
        # Switch to review mode
        self.mode = "review"
    
    def run_training(self):
        """Run training job"""
        dataset_dir = Path(config.DATASETS_DIR) / f"batch_{self.batch_id}"
        inference_dir = Path(config.INFERENCE_DIR) / f"batch_{self.batch_id}"
        
        def progress_callback(message):
            self.training_logs.append(message)
            print(f"  [Training] {message}")
        
        try:
            # Train locally
            success = train_model(
                dataset_dir=str(dataset_dir),
                output_dir=str(inference_dir),
                progress_callback=progress_callback
            )
            
            if success:
                self.training_logs.append("Training complete!")
                print("\n✓ Training complete!")
            else:
                self.training_logs.append("Training failed")
                print("\n✗ Training failed")
        
        except Exception as e:
            self.training_logs.append(f"Error: {str(e)}")
            print(f"\n✗ Error: {e}")
        
        finally:
            self.training_in_progress = False
    
    def draw_review(self):
        """Draw review interface for captured images"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Title bar
        cv2.putText(frame, "REVIEW CAPTURED IMAGES", (250, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)
        
        # Check if we have images to review
        if not self.captured_images:
            cv2.putText(frame, "No images to review", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        # Clamp review index
        self.review_index = max(0, min(self.review_index, len(self.captured_images) - 1))
        
        # Load and display current image
        current_img_path = self.captured_images[self.review_index]
        img = cv2.imread(current_img_path)
        
        if img is not None:
            # Resize image to fit in display area (center area)
            display_height = 500
            display_width = 700
            img_resized = cv2.resize(img, (display_width, display_height))
            
            # Center the image
            x_offset = (self.screen_width - display_width) // 2
            y_offset = 100
            
            # Check if image is marked for discard
            is_discarded = self.review_index in self.images_to_discard
            
            if is_discarded:
                # Apply red tint for discarded images
                red_overlay = np.zeros_like(img_resized)
                red_overlay[:, :] = (0, 0, 255)
                img_resized = cv2.addWeighted(img_resized, 0.5, red_overlay, 0.5, 0)
                
                # Add "DISCARDED" text
                cv2.putText(img_resized, "DISCARDED", (display_width // 2 - 150, display_height // 2), 
                            cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
            
            # Place image on frame
            frame[y_offset:y_offset + display_height, x_offset:x_offset + display_width] = img_resized
            
            # Draw border around image
            border_color = (0, 0, 255) if is_discarded else (0, 255, 0)
            cv2.rectangle(frame, (x_offset - 2, y_offset - 2), 
                          (x_offset + display_width + 2, y_offset + display_height + 2), 
                          border_color, 3)
        else:
            cv2.putText(frame, "Error loading image", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Image counter
        total_images = len(self.captured_images)
        discarded_count = len(self.images_to_discard)
        remaining_count = total_images - discarded_count
        
        counter_text = f"Image {self.review_index + 1} / {total_images}"
        cv2.putText(frame, counter_text, (420, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status bar
        status_text = f"Remaining: {remaining_count}  |  Discarded: {discarded_count}"
        cv2.putText(frame, status_text, (320, 630), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Instructions
        y = 670
        next_key = self.keys.get('review', 'next').upper()
        prev_key = self.keys.get('review', 'previous').upper()
        discard_key = self.keys.get('review', 'discard').upper()
        confirm_key = self.keys.get('review', 'confirm').upper()
        back_key = self.keys.get('review', 'back').upper()
        
        instructions = [
            f"'{prev_key}' Previous  |  '{next_key}' Next  |  '{discard_key}' Discard/Restore  |  '{confirm_key}' Confirm & Train  |  '{back_key}' Back",
        ]
        
        for text in instructions:
            cv2.putText(frame, text, (50, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            y += 30
        
        # Warning if too few images after discard
        if remaining_count < config.MIN_IMAGES_FOR_TRAINING:
            warning_text = f"WARNING: Need at least {config.MIN_IMAGES_FOR_TRAINING} images for training!"
            cv2.putText(frame, warning_text, (200, 720), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def draw_upload_status(self):
        """Draw upload and training status screen"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Title
        title_color = (0, 255, 0) if self.upload_complete else (0, 255, 255)
        title = "UPLOAD COMPLETE. TRAINING Started..." if self.upload_complete else "UPLOADING DATASET & TRAINING"
        cv2.putText(frame, title, (150, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, title_color, 3)
        
        # Status display area
        y = 250
        
        # Main status message
        status_color = (0, 255, 0) if "✅" in self.upload_status else (0, 0, 255) if "❌" in self.upload_status else (0, 255, 255)
        cv2.putText(frame, self.upload_status, (200, y), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2)
        
        y += 100
        
        # Batch ID
        batch_text = f"Batch ID: {self.batch_id}"
        cv2.putText(frame, batch_text, (250, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        y += 60
        
        # Images count
        images_text = f"Images Processed: {self.capture_count}"
        cv2.putText(frame, images_text, (250, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Progress indicator
        if self.upload_in_progress:
            y = 500
            # Animated progress indicator
            import time
            progress_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            char_index = int(time.time() * 10) % len(progress_chars)
            progress_text = f"{progress_chars[char_index]} Processing..."
            cv2.putText(frame, progress_text, (420, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Recent logs
        y = 600
        cv2.putText(frame, "Recent Activity:", (50, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        y += 40
        log_color = (100, 255, 100)
        for log_msg in self.training_logs[-4:]:
            cv2.putText(frame, log_msg[:80], (70, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, log_color, 1)
            y += 30
        
        # Instructions
        if self.upload_complete:
            instructions = "Press 'M' to return to Menu"
            instructions_color = (0, 255, 0)
        else:
            instructions = "Processing in progress... Press 'M' to cancel and return to Menu"
            instructions_color = (0, 255, 255)
        
        cv2.putText(frame, instructions, (300, self.screen_height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, instructions_color, 2)
        
        return frame
    
    def review_next_image(self):
        """Navigate to next image in review"""
        if self.review_index < len(self.captured_images) - 1:
            self.review_index += 1
        else:
            # Wrap around to first image
            self.review_index = 0
    
    def review_previous_image(self):
        """Navigate to previous image in review"""
        if self.review_index > 0:
            self.review_index -= 1
        else:
            # Wrap around to last image
            self.review_index = len(self.captured_images) - 1
    
    def review_discard_current(self):
        """Toggle discard status of current image"""
        if self.review_index in self.images_to_discard:
            # Restore the image
            self.images_to_discard.remove(self.review_index)
            print(f"Image {self.review_index + 1} restored")
        else:
            # Mark for discard
            self.images_to_discard.add(self.review_index)
            print(f"Image {self.review_index + 1} marked for discard")
    
    def review_confirm_submit(self):
        """Confirm review and start training with remaining images"""
        total_images = len(self.captured_images)
        discarded_count = len(self.images_to_discard)
        remaining_count = total_images - discarded_count
        
        # Check if we have enough images
        if remaining_count < config.MIN_IMAGES_FOR_TRAINING:
            print(f"\nError: Not enough images! Need at least {config.MIN_IMAGES_FOR_TRAINING}, have {remaining_count}")
            self.training_logs.append(f"Error: Not enough images ({remaining_count}/{config.MIN_IMAGES_FOR_TRAINING})")
            return
        
        # Delete discarded images
        for idx in sorted(self.images_to_discard, reverse=True):
            img_path = self.captured_images[idx]
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")
            
            # Remove from list
            del self.captured_images[idx]
        
        # Update capture count
        self.capture_count = len(self.captured_images)
        
        print(f"\nStarting dataset upload with {self.capture_count} images (discarded {discarded_count})...")
        self.training_logs.append(f"Uploading dataset with {self.capture_count} images...")
        
        # Clear review state
        self.images_to_discard.clear()
        self.review_index = 0
        
        # Switch to upload mode
        self.mode = "upload"
        self.upload_in_progress = True
        self.upload_complete = False
        self.upload_status = "Initializing upload..."
        
        # Run upload in thread
        thread = threading.Thread(target=self.run_upload_and_training, daemon=True)
        thread.start()
    
    def run_upload_and_training(self):
        """Upload dataset to server and then trigger training via API"""
        try:
            import requests
            
            # First, upload to server
            self.upload_status = "Uploading dataset to server..."
            local_dataset_dir = str(Path(config.DATASETS_DIR) / f"batch_{self.batch_id}")
            remote_path = f"root@{config.SERVER_IP}:{config.SERVER_DATASETS_DIR}/batch_{self.batch_id}"
            ssh_key = config.SERVER_SSH_KEY
            
            rsync_cmd = [
                "rsync", "-avz", "-e",
                f"ssh -i {ssh_key}",
                local_dataset_dir + "/",  # trailing slash to copy contents
                remote_path
            ]
            
            print(f"\n📤 Uploading dataset to server: {remote_path}")
            self.training_logs.append(f"Uploading to: {remote_path}")
            
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300000)
            if result.returncode == 0:
                self.upload_status = "✅ Dataset uploaded successfully!"
                print("✅ Dataset uploaded to server successfully.")
                self.training_logs.append("✅ Dataset uploaded to server.")
            else:
                self.upload_status = f"❌ Upload failed: {result.stderr[:100]}"
                print(f"❌ Rsync failed: {result.stderr}")
                self.training_logs.append(f"❌ Upload failed: {result.stderr}")
                self.upload_complete = True
                self.upload_in_progress = False
                return
            
            # Now trigger training on server via API
            self.upload_status = "Starting training on server..."
            self.training_logs.append("Triggering server training...")
            print(f"\n🚀 Triggering training on server...")
            
            api_url = config.SERVER_TRAINING_API
            payload = {
                "class_name": f"batch_{self.batch_id}"
            }
            headers = {
                "Content-Type": "application/json"
            }
            
            print(f"POST {api_url}")
            print(f"Payload: {payload}")
            
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                self.upload_status = "✅ Training started on server!"
                print(f"✅ Training API response: {response.status_code}")
                self.training_logs.append(f"✅ Training started on server - API responded with {response.status_code}")
                print(f"Response: {response.text}")
            else:
                self.upload_status = f"❌ Training API error: {response.status_code}"
                print(f"❌ Training API error: {response.status_code}")
                self.training_logs.append(f"❌ Training API error: {response.status_code}")
                print(f"Response: {response.text}")
        
        except subprocess.TimeoutExpired:
            self.upload_status = "❌ Upload timeout (> 5 minutes)"
            self.training_logs.append("Upload timeout")
            print("❌ Upload timeout")
        except requests.exceptions.RequestException as e:
            self.upload_status = f"❌ API Error: {str(e)[:80]}"
            self.training_logs.append(f"API Error: {str(e)}")
            print(f"❌ API Error: {e}")
        except Exception as e:
            self.upload_status = f"❌ Error: {str(e)[:80]}"
            self.training_logs.append(f"Error: {str(e)}")
            print(f"❌ Error: {e}")
        
        finally:
            self.upload_in_progress = False
            self.upload_complete = True
            print("\nUpload and training trigger complete.")
    
    def start_testing_mode(self):
        """Initialize testing mode - start with model selection"""
        # Load available models
        self.load_available_models()
        
        if not self.available_models:
            print("\nNo models found! Please train a model first.")
            self.mode = "menu"
            return
        
        # Go to model selection mode
        self.mode = "model_select"
        self.model_selection_index = 0
        print(f"\nFound {len(self.available_models)} available models. Please select one.")
    
    def auto_detect_model(self):
        """Auto-detect best matching model"""
        inference_dirs = list(Path(config.INFERENCE_DIR).glob("batch_*"))
        
        if not inference_dirs:
            print("No trained models found")
            return None
        
        # Get current frame
        ret, frame = self.camera.read()
        if not ret:
            return str(inference_dirs[-1])  # Latest
        
        # Compare with samples
        best_match = None
        best_score = -1
        
        for model_dir in inference_dirs:
            sample_path = model_dir / "sample.png"
            if not sample_path.exists():
                continue
            
            sample = cv2.imread(str(sample_path))
            if sample is None:
                continue
            
            # Simple histogram comparison
            hist_frame = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_sample = cv2.calcHist([sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            hist_frame = cv2.normalize(hist_frame, hist_frame).flatten()
            hist_sample = cv2.normalize(hist_sample, hist_sample).flatten()
            
            score = cv2.compareHist(hist_frame, hist_sample, cv2.HISTCMP_CORREL)
            
            if score > best_score:
                best_score = score
                best_match = str(model_dir)
        
        return best_match if best_match else str(inference_dirs[-1])
    
    def draw_testing(self):
        """Draw testing interface"""
        if self.camera is None or not self.camera.isOpened():
            return self.draw_menu()
        
        ret, frame = self.camera.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Run inference (simulation)
            result = self.run_inference(frame)
            
            # Draw overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "INFERENCE MODE", (20, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
            
            if self.selected_model:
                model_name = Path(self.selected_model).name
                cv2.putText(frame, f"Model: {model_name}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw result
            if result:
                label = result['label']
                confidence = result['confidence']
                
                color = (0, 255, 0) if label == "OK" else (0, 0, 255)
                cv2.putText(frame, f"{label}: {confidence:.1%}", (20, 90), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
            # Instructions
            y = 420
            reselect_key = self.keys.get('inference', 'reselect_model').upper()
            menu_key = self.keys.get('inference', 'back_to_menu').upper()
            
            instructions = [
                f"Press '{menu_key}' - Back to Menu",
                f"Press '{reselect_key}' - Reselect Model",
            ]
            
            for text in instructions:
                cv2.putText(frame, text, (10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 25
        
        # Resize to full window
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        
        return frame
    
    def load_available_models(self):
        """Load list of available models and their preview images"""
        logger.info("Loading available models...")
        
        # Models are inside the backbone_0 directory
        models_dir = Path(GLASS_MODEL_PATH)
        # Preview images are in results/judge/avg/0
        judge_dir = Path("results") / "judge" / "avg" / "0"
        
        logger.debug(f"GLASS_MODEL_PATH: {GLASS_MODEL_PATH}")
        logger.debug(f"Models directory: {models_dir}")
        logger.debug(f"Models directory exists: {models_dir.exists()}")
        logger.debug(f"Judge directory: {judge_dir}")
        logger.debug(f"Judge directory exists: {judge_dir.exists()}")
        
        self.available_models = []
        self.model_previews = {}
        
        # Find all model directories inside backbone_0
        if models_dir.exists():
            logger.debug(f"Scanning models directory: {models_dir}")
            for model_path in models_dir.iterdir():
                logger.debug(f"Found item: {model_path.name} (is_dir: {model_path.is_dir()}, starts_with_dot: {model_path.name.startswith('.')})")
                if model_path.is_dir() and not model_path.name.startswith('.'):
                    model_name = model_path.name
                    logger.info(f"Found model: {model_name}")
                    self.available_models.append(model_name)
                    
                    # Look for corresponding preview image in results/judge/avg/0
                    preview_image = None
                    
                    if judge_dir.exists():
                        # Look for image file with model name
                        preview_path = judge_dir / f"{model_name}.png"
                        if preview_path.exists():
                            preview_image = str(preview_path)
                            logger.debug(f"Found preview image: {preview_image}")
                        else:
                            logger.debug(f"No preview image found for model: {model_name} at {preview_path}")
                    else:
                        logger.debug(f"Judge directory not found: {judge_dir}")
                    
                    self.model_previews[model_name] = preview_image
        else:
            logger.warning(f"Models directory does not exist: {models_dir}")
        
        logger.info(f"Found {len(self.available_models)} models: {self.available_models}")
        print(f"Found models: {self.available_models}")
    
    def draw_model_selection(self):
        """Draw model selection interface"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(frame, "SELECT MODEL FOR INFERENCE", (200, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)
        
        if not self.available_models:
            cv2.putText(frame, "No models available", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        # Clamp selection index
        self.model_selection_index = max(0, min(self.model_selection_index, len(self.available_models) - 1))
        
        current_model = self.available_models[self.model_selection_index]
        preview_path = self.model_previews.get(current_model)
        
        # Display current model info
        cv2.putText(frame, f"Model: {current_model}", (50, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Model {self.model_selection_index + 1} of {len(self.available_models)}", (50, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Display preview image if available
        if preview_path and os.path.exists(preview_path):
            try:
                preview_img = cv2.imread(preview_path)
                if preview_img is not None:
                    # Resize preview to fit display area
                    display_height = 400
                    display_width = 600
                    preview_resized = cv2.resize(preview_img, (display_width, display_height))
                    
                    # Center the preview
                    x_offset = (self.screen_width - display_width) // 2
                    y_offset = 200
                    
                    # Place preview on frame
                    frame[y_offset:y_offset + display_height, x_offset:x_offset + display_width] = preview_resized
                    
                    # Draw border around preview
                    cv2.rectangle(frame, (x_offset - 2, y_offset - 2), 
                                  (x_offset + display_width + 2, y_offset + display_height + 2), 
                                  (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Preview image not available", (300, 400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            except Exception as e:
                cv2.putText(frame, f"Error loading preview: {str(e)[:50]}", (200, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No preview available", (350, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        
        # Instructions
        y = 650
        instructions = [
            "'N' Next Model  |  'P' Previous Model  |  'S' Select & Start Inference  |  'B' Back to Menu",
        ]
        
        for text in instructions:
            cv2.putText(frame, text, (50, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            y += 30
        
        return frame
    
    def model_selection_next(self):
        """Navigate to next model"""
        if self.model_selection_index < len(self.available_models) - 1:
            self.model_selection_index += 1
        else:
            self.model_selection_index = 0  # Wrap around
    
    def model_selection_previous(self):
        """Navigate to previous model"""
        if self.model_selection_index > 0:
            self.model_selection_index -= 1
        else:
            self.model_selection_index = len(self.available_models) - 1  # Wrap around
    
    def confirm_model_selection(self):
        """Confirm model selection and start inference"""
        if not self.available_models:
            return
        
        selected_model_name = self.available_models[self.model_selection_index]
        # The selected model path should point to the backbone_0 directory
        # since GLASS expects the backbone structure
        self.selected_model = GLASS_MODEL_PATH
        self.selected_model_class = selected_model_name
        
        print(f"\nSelected model: {selected_model_name}")
        
        # Check if GLASS inference is available
        if GLASS_AVAILABLE:
            print("Starting GLASS Inference Mode...")
            self.run_glass_inference()
            # After GLASS inference completes, return to menu
            self.mode = "menu"
            
            # Ensure main window is properly displayed
            cv2.destroyAllWindows()  # Clean up any leftover windows
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
            return
        
        # Fallback to OpenCV mode with high-quality settings
        logger.info("🎯 Starting testing mode with high-quality camera settings...")
        self.camera = self.initialize_high_quality_camera()
        
        if self.camera is None:
            print("Error: Could not open camera with high-quality settings")
            logger.error("Failed to initialize high-quality camera for testing")
            self.mode = "menu"
            return
        
        print(f"Testing Mode Active with model: {selected_model_name}")
        self.mode = "test"
    
    def run_glass_inference(self):
        """Run GLASS inference with video file using orchestrator"""
        try:
            # Import GLASS inference orchestrator
            sys.path.insert(0, GLASS_INFERENCE_PATH)
            from inference_orchestrator import GLASSInferenceOrchestrator

            # Get the selected model class name
            if hasattr(self, 'selected_model_class') and self.selected_model_class:
                class_name = self.selected_model_class
                print(f"Using selected GLASS model class: {class_name}")
            else:
                print("Error: No model class selected")
                return

            # Video path for inference
            video_path = "test-video/custom/grid/grid_combined.mp4"

            # Check if video exists
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found: {video_path}")
                print("Please ensure the video file exists before running inference.")
                return

            # Initialize GLASS inference orchestrator with proper GPU detection
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🎯 Initializing GLASS inference with device: {device}")

            orchestrator = GLASSInferenceOrchestrator(
                models_base_path=GLASS_MODEL_PATH,
                device=device,
                image_size=384,
                sample_frames=30
            )

            print("\n" + "="*60)
            print("GLASS Video Inference Active")
            print(f"Video: {video_path}")
            print(f"Model: {class_name}")
            print("="*60 + "\n")

            # Run video inference with selected model class
            results = orchestrator.run_inference_with_best_model(
                video_path=video_path,
                output_path=None,  # Will use organized output
                manual_class_name=class_name
            )

            print("\n" + "="*60)
            print("GLASS Inference Complete")
            print(f"Selected Model: {results['orchestrator_info']['selected_model']}")
            if 'unique_defects' in results['inference_results']:
                print(f"Unique defects found: {results['inference_results']['unique_defects']}")
            if 'fps_processing' in results['inference_results']:
                print(f"Processing FPS: {results['inference_results']['fps_processing']:.1f}")
            print("="*60 + "\n")

            # Generate PDF report immediately
            print("📄 Generating report...")
            self.generate_pdf_report(results)

        except ImportError as e:
            print(f"Error importing GLASS orchestrator: {e}")
            print("Make sure GLASS dependencies are installed")
        except Exception as e:
            print(f"Error during GLASS inference: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_pdf_report(self, results: dict):
        """Generate PDF report from inference results"""
        try:
            from pdf_report_generator import GLASSReportGenerator
            
            # Find the tracking report JSON file in the output directory structure
            # Expected path: output/mvtec_mvt2/20251025_192359/report/mvtec_mvt2_tracking_report.json
            
            # Look in the main output directory for the latest session
            output_base = Path("output")
            if not output_base.exists():
                print("⚠️  Output directory not found")
                return
            
            # Find the model class directory (e.g., mvtec_mvt2)
            model_class = self.selected_model_class if hasattr(self, 'selected_model_class') else None
            if not model_class:
                print("⚠️  No model class selected")
                return
            
            model_output_dir = output_base / model_class
            if not model_output_dir.exists():
                print(f"⚠️  Model output directory not found: {model_output_dir}")
                return
            
            # Find the most recent session directory (sorted by name, which includes timestamp)
            session_dirs = [d for d in model_output_dir.iterdir() if d.is_dir()]
            if not session_dirs:
                print(f"⚠️  No session directories found in: {model_output_dir}")
                return
            
            # Get the most recent session (latest timestamp)
            latest_session = sorted(session_dirs, key=lambda x: x.name)[-1]
            report_dir = latest_session / 'report'
            
            if not report_dir.exists():
                print(f"⚠️  Report directory not found: {report_dir}")
                return
            
            # Look for tracking report JSON
            tracking_report_files = list(report_dir.glob('*_tracking_report.json'))
            
            if tracking_report_files:
                json_path = str(tracking_report_files[0])
                print(f"📄 Found tracking report: {json_path}")
                print(f"📄 Generating PDF report...")
                
                # Generate PDF report
                generator = GLASSReportGenerator()
                pdf_path = generator.generate_report(json_path, open_after=False)
                
                print(f"✅ PDF report generated and saved to Documents folder!")
                print(f"📁 PDF location: {pdf_path}")
                
                # Store PDF path for orchestrator (don't try to open from OpenCV context)
                print(f"ORCHESTRATOR_PDF_PATH:{pdf_path}")  # Special marker for orchestrator
                
            else:
                print(f"⚠️  No tracking report JSON found in: {report_dir}")
                # List available files for debugging
                available_files = list(report_dir.glob('*.json'))
                if available_files:
                    print(f"Available JSON files: {[f.name for f in available_files]}")
                
        except ImportError as e:
            print(f"⚠️  Could not generate PDF report: {e}")
            print("💡 Install reportlab with: pip install reportlab")
        except Exception as e:
            print(f"⚠️  Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()
    
    def open_pdf_file(self, pdf_path):
        """Simple PDF opening with error reporting"""
        import subprocess
        
        # Try xdg-open exactly like the working manual script
        try:
            result = subprocess.run(['xdg-open', pdf_path], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL,
                                  timeout=5)
            if result.returncode == 0:
                print(f"📖 PDF opened successfully")
                return
            else:
                print(f"❌ xdg-open failed with return code: {result}")
        except Exception as e:
            print(f"❌ xdg-open error: {e}")
        
        # If xdg-open failed, just show the path
        print(f"💡 Manually open with: xdg-open '{pdf_path}'")
    
    def run_inference(self, frame):
        """Run inference on frame (simulation)"""
        # Simple simulation based on brightness
        # Replace this with your actual model inference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if avg_brightness < 50:
            return {'label': 'ALERT', 'confidence': 0.85}
        elif std_brightness > 60:
            return {'label': 'DEFECT', 'confidence': 0.75}
        else:
            return {'label': 'OK', 'confidence': 0.92}
    
    def cleanup_mode(self):
        """Cleanup current mode"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        self.current_frame = None
    
    def cleanup(self):
        """Clean up resources"""
        # Stop model sync thread
        self.model_sync_running = False
        if self.model_sync_thread:
            self.model_sync_thread.join(timeout=2)
        
        if self.camera and self.camera.isOpened():
            self.camera.release()
        
        if self.serial_listener:
            self.serial_listener.stop()
        
        cv2.destroyAllWindows()
        print("\nFabric Inspector closed. Goodbye!")
    
    def sync_models_loop(self):
        """Background thread to sync models every 5 minutes"""
        import time
        
        while self.model_sync_running:
            try:
                current_time = time.time()
                if current_time - self.last_model_sync >= config.MODEL_SYNC_INTERVAL:
                    self.sync_models_from_server()
                    self.last_model_sync = current_time
            except Exception as e:
                logger.error(f"Error in model sync loop: {e}")
            
            # Sleep in small intervals to allow quick shutdown
            time.sleep(1)
    
    def sync_models_from_server(self):
        """Fetch available models from server and rsync results folder back"""
        try:
            import requests
            
            logger.info("📡 Syncing models from server...")
            
            # Call the models API
            api_url = config.SERVER_MODELS_API
            response = requests.get(api_url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch models: {response.status_code}")
                return
            
            models_data = response.json()
            model_count = models_data.get('count', 0)
            models = models_data.get('models', [])
            
            logger.debug(f"Server has {model_count} models available")
            
            if model_count == 0:
                logger.debug("No models available on server")
                return
            
            # Rsync results folder back to local
            logger.info(f"📥 Syncing results folder from server...")
            local_results_dir = Path("results")
            local_results_dir.mkdir(exist_ok=True)
            
            remote_path = f"root@{config.SERVER_IP}:{config.SERVER_RESULTS_DIR}/"
            ssh_key = config.SERVER_SSH_KEY
            
            rsync_cmd = [
                "rsync", "-avz", "--delete", "-e",
                f"ssh -i {ssh_key}",
                remote_path,
                str(local_results_dir) + "/"
            ]
            
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"✅ Models synced successfully ({model_count} models available)")
                logger.debug(f"Models: {[m['name'] for m in models]}")
            else:
                logger.warning(f"⚠️  Rsync results failed: {result.stderr}")
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to models API: {e}")
        except subprocess.TimeoutExpired:
            logger.warning("Model sync rsync timeout")
        except Exception as e:
            logger.error(f"Error syncing models: {e}")


def main():
    """Main entry point"""
    # Create directories
    Path(config.DATASETS_DIR).mkdir(exist_ok=True)
    Path(config.INFERENCE_DIR).mkdir(exist_ok=True)
    
    # Start application
    app = FabricInspector()
    app.start()


if __name__ == "__main__":
    main()
