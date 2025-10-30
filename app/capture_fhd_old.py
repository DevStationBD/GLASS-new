#!/usr/bin/env python3
"""
1080p Command-Based Image Capture
Captures 1080p images on command via serial listener
"""

import cv2
import time
import threading
import signal
import sys
import os
import serial
from datetime import datetime
from collections import deque
import logging
import subprocess
import re

class CommandBasedCapture1080p:
    def __init__(self):
        # Fixed configuration - no user input
        self.camera_index = 0
        self.target_width, self.target_height = 1920, 1080  # 1080p resolution
        self.width, self.height = 1920, 1080
        self.resolution_name = "1080p"
        self.jpeg_quality = 95
        self.base_storage_path = "collection"
        
        # Serial port configuration
        self.devices = self.find_devices()
        self.serial_ports = {}
        self.monitoring = False
        self.command_stats = {'received': 0, 'processed': 0, 'errors': 0}
        self.last_command_time = 0
        
        # Camera and capture state
        self.cap = None
        self.running = False
        self.frame_times = deque(maxlen=60)
        self.frame_count = 0
        self.saved_images = 0
        self.start_time = None
        
        # Video recording - removed for command-based capture
        self.session_folder = None
        
        # Preview settings
        self.show_preview = False
        self.preview_width = 0
        self.preview_height = 0
        self.screen_detected = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Create storage structure
        self.create_session_folder()
        
        # Initialize screen detection and preview
        self.detect_screen()
    
    def find_devices(self):
        """Find all available input devices"""
        device_candidates = [
            '/dev/input-device',      # Hardware device
            '/tmp/input-virtual'      # Mock device for testing
        ]
        
        found_devices = []
        for device_path in device_candidates:
            if os.path.exists(device_path):
                found_devices.append(device_path)
        
        return found_devices
    
    def signal_handler(self, sig, frame):
        self.logger.info("Stopping capture...")
        self.stop_capture()
        sys.exit(0)
    
    def create_session_folder(self):
        """Create session folder with timestamp"""
        try:
            # Create base collection folder
            if not os.path.exists(self.base_storage_path):
                os.makedirs(self.base_storage_path)
            
            # Create session subfolder: yyyy-mm-dd-hh-mm
            session_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.session_folder = os.path.join(self.base_storage_path, session_timestamp)
            
            if not os.path.exists(self.session_folder):
                os.makedirs(self.session_folder)
                self.logger.info(f"Created session folder: {self.session_folder}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to create session folder: {e}")
            return False
    
    def setup_server_socket(self):
        """Setup serial port monitoring for command listening"""
        if not self.devices:
            self.logger.warning("No input devices found")
            self.logger.warning("Connect ESP32/NodeMCU or run setup_hardware.sh for hardware button support.")
            return False
        
        self.logger.info(f"Found {len(self.devices)} input device(s): {self.devices}")
        return True
    
    def start_monitoring(self):
        """Start monitoring all available devices"""
        if not self.setup_server_socket():
            return False
            
        self.monitoring = True
        self.logger.info("🔍 Starting serial device monitoring...")
        self.logger.info(f"📡 Listening for commands on devices: {self.devices}")
        self.logger.info("📋 Supported commands: func1 (capture image), quit/stop (exit)")
        
        # Start monitoring each device in its own thread
        monitor_threads = []
        for device_path in self.devices:
            thread = threading.Thread(
                target=self.monitor_single_device, 
                args=(device_path,), 
                daemon=True
            )
            thread.start()
            monitor_threads.append(thread)
            self.logger.info(f"🔧 Started monitoring thread for {device_path}")
        
        self.logger.info("✅ Serial monitoring started successfully")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring all devices"""
        self.logger.info("Stopping serial monitoring...")
        self.monitoring = False
        
        # Close all serial ports
        for device_path, ser in list(self.serial_ports.items()):
            try:
                ser.close()
                self.logger.debug(f"Closed serial port: {device_path}")
            except Exception as e:
                self.logger.error(f"Error closing {device_path}: {e}")
        
        self.serial_ports.clear()
        self.logger.info("Serial monitoring stopped")
    
    def monitor_single_device(self, device_path):
        """Monitor a single device in its own thread"""
        retry_count = 0
        max_retries = 5
        
        self.logger.debug(f"Starting monitor thread for {device_path}")
        
        while self.monitoring and retry_count < max_retries:
            try:
                self.logger.info(f"Opening serial connection: {device_path}")
                
                with serial.Serial(
                    port=device_path,
                    baudrate=115200,
                    timeout=0.1,  # Reduced timeout for faster response
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                ) as ser:
                    
                    self.serial_ports[device_path] = ser
                    self.logger.info(f"Connected to {device_path}")
                    
                    # Log buffer settings for debugging
                    self.logger.debug(f"Serial settings for {device_path}:")
                    self.logger.debug(f"  Baudrate: {ser.baudrate}")
                    self.logger.debug(f"  Timeout: {ser.timeout}s")
                    self.logger.debug(f"  Buffer size: {ser.in_waiting} (initial)")
                    
                    # Reset retry counter on successful connection
                    retry_count = 0
                    
                    while self.monitoring:
                        try:
                            # Process all available data in buffer
                            while ser.in_waiting > 0:
                                self.logger.debug(f"📊 Buffer status on {device_path}: {ser.in_waiting} bytes waiting")
                                
                                # Read all available data
                                try:
                                    raw_data = ser.readline()
                                    if not raw_data:
                                        break
                                        
                                    self.logger.info(f"📥 Raw data received from {device_path}: {raw_data}")
                                    command = raw_data.decode('utf-8', errors='ignore').strip().lower()
                                    self.logger.info(f"📋 Decoded command from {device_path}: '{command}'")
                                    
                                    if command:
                                        self.logger.info(f"✅ Processing command: '{command}' from {device_path}")
                                        self.handle_command(command, device_path)
                                    else:
                                        self.logger.warning(f"⚠️  Empty command received from {device_path}")
                                        
                                except UnicodeDecodeError as e:
                                    self.logger.warning(f"⚠️  Failed to decode data from {device_path}: {e}")
                                except Exception as e:
                                    self.logger.error(f"❌ Error processing data from {device_path}: {e}")
                                    break
                            
                            # Shorter sleep for more responsive monitoring
                            time.sleep(0.01)  # 10ms sleep instead of 100ms
                            
                            # Periodic status check (every 10 seconds)
                            if not hasattr(self, '_last_status_log'):
                                self._last_status_log = time.time()
                            elif time.time() - self._last_status_log > 10:
                                self.logger.debug(f"📊 Status {device_path}: Commands={self.command_stats['received']}, Buffer={ser.in_waiting}b")
                                self._last_status_log = time.time()
                            
                        except serial.SerialException as e:
                            self.logger.error(f"Serial read error on {device_path}: {e}")
                            break
                        except Exception as e:
                            self.logger.error(f"Unexpected error in monitoring loop for {device_path}: {e}")
                            break
                            
            except serial.SerialException as e:
                retry_count += 1
                self.logger.error(f"Failed to open {device_path}: {e}")
                if retry_count < max_retries and self.monitoring:
                    wait_time = retry_count * 2  # Exponential backoff
                    self.logger.info(f"Retrying {device_path} in {wait_time} seconds... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    if self.monitoring:
                        self.logger.error(f"Max retries reached for {device_path}")
                    break
            except Exception as e:
                self.logger.error(f"Unexpected error on {device_path}: {e}")
                break
            finally:
                # Clean up serial port reference
                if device_path in self.serial_ports:
                    del self.serial_ports[device_path]
        
        self.logger.info(f"Monitoring stopped for {device_path}")
    
    def handle_command(self, command, device_source=None):
        """Handle serial commands"""
        device_info = f" (from {device_source})" if device_source else ""
        current_time = time.time()
        
        # Update statistics
        self.command_stats['received'] += 1
        time_since_last = current_time - self.last_command_time if self.last_command_time > 0 else 0
        self.last_command_time = current_time
        
        self.logger.info(f"🎯 Command #{self.command_stats['received']}: '{command.upper()}'{device_info}")
        if time_since_last > 0:
            self.logger.debug(f"⏱️  Time since last command: {time_since_last:.3f}s")
        
        try:
            if command == "func1":
                self.logger.info(f"📸 FUNC1 command triggered - capturing image{device_info}")
                success = self.capture_single_image()
                if success:
                    self.command_stats['processed'] += 1
                    self.logger.info(f"✅ Image capture completed successfully{device_info}")
                else:
                    self.command_stats['errors'] += 1
                    self.logger.error(f"❌ Image capture failed{device_info}")
            elif command == "quit" or command == "stop":
                self.logger.info(f"🛑 Stop command received{device_info}")
                self.command_stats['processed'] += 1
                self.stop_capture()
            else:
                self.logger.warning(f"❓ Unknown command: '{command}'{device_info}")
                self.logger.info("📋 Available commands: func1, quit, stop")
                self.command_stats['errors'] += 1
                
        except Exception as e:
            self.command_stats['errors'] += 1
            self.logger.error(f"❌ Error handling command '{command}': {e}")
    
    def capture_single_image(self):
        """Capture and save a single image"""
        self.logger.info("📷 Starting image capture...")
        
        if not self.cap or not self.cap.isOpened():
            self.logger.error("❌ Camera not initialized")
            return False
            
        try:
            self.logger.debug("📹 Reading frame from camera...")
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("❌ Failed to capture frame from camera")
                return False
            
            self.logger.debug(f"✅ Frame captured successfully - size: {frame.shape}")
            self.frame_count += 1
            current_time = time.time()
            self.frame_times.append(current_time)
            
            # Save the frame
            self.logger.info("💾 Saving captured frame...")
            success = self.save_frame(frame)
            
            if success:
                self.logger.info(f"✅ Image #{self.saved_images} saved successfully")
            else:
                self.logger.error("❌ Failed to save image")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error capturing image: {e}")
            return False
    
    def detect_screen(self):
        """Detect screen resolution and enable preview if screen available"""
        try:
            # Check if DISPLAY environment variable is set (X11/Wayland)
            display_env = os.environ.get('DISPLAY')
            if not display_env:
                self.logger.info("No display environment detected - preview disabled")
                return
            
            # Try to get screen resolution using xrandr (X11)
            try:
                result = subprocess.run(['xrandr', '--current'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse primary display resolution
                    for line in result.stdout.split('\n'):
                        if ' connected primary ' in line or ' connected ' in line:
                            # Look for resolution pattern like "1920x1080"
                            match = re.search(r'(\d+)x(\d+)\+\d+\+\d+', line)
                            if match:
                                self.preview_width = int(match.group(1))
                                self.preview_height = int(match.group(2))
                                self.screen_detected = True
                                self.show_preview = True
                                break
                    
                    if self.screen_detected:
                        self.logger.info(f"📺 Screen detected: {self.preview_width}x{self.preview_height}")
                        self.logger.info("  Preview enabled - press 'q' to quit preview")
                        return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback: Try with xdpyinfo (X11)
            try:
                result = subprocess.run(['xdpyinfo'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Look for dimensions line
                    for line in result.stdout.split('\n'):
                        if 'dimensions:' in line:
                            match = re.search(r'(\d+)x(\d+) pixels', line)
                            if match:
                                self.preview_width = int(match.group(1))
                                self.preview_height = int(match.group(2))
                                self.screen_detected = True
                                self.show_preview = True
                                self.logger.info(f"📺 Screen detected: {self.preview_width}x{self.preview_height}")
                                self.logger.info("  Preview enabled - press 'q' to quit preview")
                                return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback: Use default preview size if we have a display but can't detect resolution
            if display_env:
                self.preview_width = 1920
                self.preview_height = 1080
                self.screen_detected = True
                self.show_preview = True
                self.logger.info("📺 Display detected, using default preview size: 1920x1080")
                self.logger.info("  Preview enabled - press 'q' to quit preview")
            else:
                self.logger.info("No screen detected - preview disabled")
                
        except Exception as e:
            self.logger.warning(f"Screen detection failed: {e}")
            self.logger.info("Preview disabled")
    
    def configure_v4l2_settings(self):
        """Configure camera using v4l2-ctl for precise control"""
        try:
            device = f"/dev/video{self.camera_index}"
            self.logger.info(f"🔧 Configuring MX Brio via v4l2-ctl on {device}")
            
            # Dictionary of v4l2 settings for balanced image capture
            v4l2_settings = {
                'auto_exposure': 1,              # Manual exposure mode
                'exposure_time_absolute': 5,    # Balanced shutter
                'gain': 255,                     # Maximum gain for fast shutter
                'brightness': 200,               # High brightness compensation
                'contrast': 200,                 # Maximum contrast
                'sharpness': 255,               # Maximum sharpness for detail
                'saturation': 100,              # Standard saturation
                'focus_automatic_continuous': 0, # Manual focus
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
                        self.logger.info(f"  ✅ {setting}: {value}")
                    else:
                        self.logger.warning(f"  ⚠️  {setting}: {value} - {result.stderr.strip()}")
                        
                except subprocess.TimeoutExpired:
                    self.logger.error(f"  ❌ {setting}: timeout")
                except FileNotFoundError:
                    self.logger.error("v4l2-ctl not found - install with: sudo apt install v4l-utils")
                    return False
                except Exception as e:
                    self.logger.error(f"  ❌ {setting}: {e}")
            
            # Verify critical settings were applied
            self.verify_v4l2_settings(device)
            return True
            
        except Exception as e:
            self.logger.error(f"V4L2 configuration failed: {e}")
            return False
    
    def verify_v4l2_settings(self, device):
        """Verify critical v4l2 settings were applied"""
        try:
            critical_settings = ['auto_exposure', 'exposure_time_absolute', 'gain']
            
            cmd = ['v4l2-ctl', '-d', device, '--get-ctrl'] + [','.join(critical_settings)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.logger.info("🔍 V4L2 Settings Verification:")
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        self.logger.info(f"  {line.strip()}")
            
        except Exception as e:
            self.logger.warning(f"V4L2 verification failed: {e}")

    def log_all_camera_settings(self):
        """Log all camera settings for verification"""
        try:
            self.logger.info("📋 CURRENT CAMERA SETTINGS:")
            
            # Core settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Image quality settings
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = self.cap.get(cv2.CAP_PROP_GAIN)
            brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
            sharpness = self.cap.get(cv2.CAP_PROP_SHARPNESS)
            saturation = self.cap.get(cv2.CAP_PROP_SATURATION)
            
            # Control settings
            autofocus = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
            auto_wb = self.cap.get(cv2.CAP_PROP_AUTO_WB)
            buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            
            self.logger.info(f"  Resolution: {width}x{height}")
            self.logger.info(f"  FPS: {fps}")
            self.logger.info(f"  Format: {fourcc_str}")
            self.logger.info(f"  Buffer Size: {buffer_size}")
            self.logger.info("  --- IMAGE QUALITY SETTINGS ---")
            self.logger.info(f"  Exposure: {exposure} (lower = faster shutter)")
            self.logger.info(f"  Gain: {gain}")
            self.logger.info(f"  Brightness: {brightness}")
            self.logger.info(f"  Contrast: {contrast}")
            self.logger.info(f"  Sharpness: {sharpness}")
            self.logger.info(f"  Saturation: {saturation}")
            self.logger.info("  --- CONTROL SETTINGS ---")
            self.logger.info(f"  Autofocus: {autofocus} (0=manual, 1=auto)")
            self.logger.info(f"  Auto White Balance: {auto_wb} (0=manual, 1=auto)")
            
            # Calculate approximate shutter speed from exposure
            if exposure != -1:  # -1 means auto
                approx_shutter = 1.0 / (2 ** abs(exposure))
                self.logger.info(f"  Estimated shutter speed: ~1/{int(1/approx_shutter)}s")
            
        except Exception as e:
            self.logger.error(f"Failed to read camera settings: {e}")
    
    def get_image_filename(self):
        """Generate timestamped filename for image"""
        timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]  # Include milliseconds
        return f"{self.resolution_name}_{timestamp}.jpg"
    
    def save_frame(self, frame):
        """Save frame to session folder"""
        try:
            filename = self.get_image_filename()
            filepath = os.path.join(self.session_folder, filename)
            
            self.logger.debug(f"💾 Saving image to: {filepath}")
            
            # Save image with high quality
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            if success:
                self.saved_images += 1
                # Get file size for logging
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                self.logger.info(f"✅ Image saved: {filename} ({file_size:.1f} MB) - Total captures: {self.saved_images}")
                return True
            else:
                self.logger.error(f"❌ Failed to save image: {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving frame: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera with 1080p resolution and anti-ghosting settings"""
        try:
            # Configure camera using v4l2-ctl first for precise control
            if not self.configure_v4l2_settings():
                self.logger.warning("V4L2 configuration failed, using OpenCV settings only")
            
            # Use V4L2 backend for best performance
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Jetson Nano optimizations (only non-conflicting settings)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            
            # Set 1080p resolution and format
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for sharper motion capture
            
            # NOTE: Image quality settings (exposure, gain, brightness, etc.) are controlled via v4l2-ctl
            # OpenCV settings would override the precise v4l2 configuration
            self.logger.info("📋 Using v4l2-ctl settings for image quality (not OpenCV overrides)")
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            actual_gain = self.cap.get(cv2.CAP_PROP_GAIN)
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Update instance variables
            self.width, self.height = actual_width, actual_height
            
            # Log final settings
            self.logger.info("✅ 1080p Camera initialized for command-based capture:")
            self.logger.info(f"  Resolution: {actual_width}x{actual_height} ({self.resolution_name})")
            self.logger.info(f"  FPS: {actual_fps}")
            self.logger.info(f"  Format: {fourcc_str}")
            self.logger.info(f"  Exposure: {actual_exposure} (fast shutter for sharp motion)")
            self.logger.info(f"  Gain: {actual_gain} (compensates for fast exposure)")
            self.logger.info(f"  JPEG Quality: {self.jpeg_quality}%")
            self.logger.info(f"  Session folder: {self.session_folder}")
            
            # Display all current camera settings for verification
            self.log_all_camera_settings()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    
    def scale_frame_for_preview(self, frame):
        """Scale frame to fit screen resolution maintaining aspect ratio"""
        if not self.show_preview:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scaling to fit within screen resolution
        # Leave some margin for window decorations (80% of screen)
        max_width = int(self.preview_width * 0.8)
        max_height = int(self.preview_height * 0.8)
        
        # Calculate scale factors
        scale_width = max_width / frame_width
        scale_height = max_height / frame_height
        
        # Use the smaller scale factor to maintain aspect ratio
        scale = min(scale_width, scale_height, 1.0)  # Don't upscale
        
        # Calculate new dimensions
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize the frame
        if scale < 1.0:
            preview_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            preview_frame = frame.copy()
        
        return preview_frame
    
    def calculate_fps(self):
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        time_span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / time_span if time_span > 0 else 0.0
    
    def log_stats(self):
        """Log statistics periodically"""
        while self.running:
            time.sleep(5)  # Log every 5 seconds
            if self.frame_count > 0:
                current_fps = self.calculate_fps()
                elapsed = time.time() - self.start_time if self.start_time else 0
                
                self.logger.info(f"{self.resolution_name} Capture - Frames: {self.frame_count:5d} | "
                               f"FPS: {current_fps:4.1f} | "
                               f"Command captures: {self.saved_images:4d} | "
                               f"Runtime: {elapsed:.0f}s")
    
    def capture_and_record(self):
        """Main capture loop with command-based image capture"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        self.logger.info(f"Starting {self.resolution_name} command-based capture...")
        self.logger.info("Send 'func1' via serial to capture images")
        self.logger.info("Press Ctrl+C to stop")
        
        # Start serial monitoring
        if not self.start_monitoring():
            self.logger.error("Failed to start serial monitoring")
            return False
        
        # Start logging thread
        log_thread = threading.Thread(target=self.log_stats, daemon=True)
        log_thread.start()
        
        try:
            # Main loop - just keep camera active and handle preview
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    break
                
                current_time = time.time()
                self.frame_times.append(current_time)
                self.frame_count += 1
                
                # Display preview if screen detected
                if self.show_preview:
                    # Scale frame to fit screen resolution
                    preview_frame = self.scale_frame_for_preview(frame)
                    
                    # Use static window name to prevent multiple windows
                    window_name = "Command Capture Preview"
                    cv2.imshow(window_name, preview_frame)
                    
                    # Update window title with capture count (if possible)
                    try:
                        cv2.setWindowTitle(window_name, f"{self.resolution_name} Command Capture - {self.saved_images} captures")
                    except:
                        pass  # Ignore if setWindowTitle not available
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Preview window closed - stopping capture")
                        break
                else:
                    # No preview - just small delay to prevent CPU spinning
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
        
        finally:
            self.stop_capture()
        
        return True
    
    def stop_capture(self):
        """Stop capture and cleanup"""
        self.running = False
        
        # Stop serial monitoring
        self.stop_monitoring()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Final statistics
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("SESSION SUMMARY:")
            self.logger.info(f"  Duration: {elapsed:.1f} seconds")
            self.logger.info(f"  Total frames processed: {self.frame_count}")
            self.logger.info(f"  Command captures: {self.saved_images}")
            self.logger.info(f"  Average FPS: {avg_fps:.1f}")
            self.logger.info(f"  Session folder: {os.path.abspath(self.session_folder)}")
            
            # Command statistics
            self.logger.info("  --- COMMAND STATISTICS ---")
            self.logger.info(f"  Commands received: {self.command_stats['received']}")
            self.logger.info(f"  Commands processed: {self.command_stats['processed']}")
            self.logger.info(f"  Command errors: {self.command_stats['errors']}")
            if self.command_stats['received'] > 0:
                success_rate = (self.command_stats['processed'] / self.command_stats['received']) * 100
                self.logger.info(f"  Success rate: {success_rate:.1f}%")
            
            # Calculate image storage
            if self.saved_images > 0:
                # Resolution-aware storage estimation
                avg_image_size = 0.25  # MB per image for 1080p
                estimated_image_storage = self.saved_images * avg_image_size
                self.logger.info(f"  Image storage: ~{estimated_image_storage:.1f} MB")
            
            self.logger.info("=" * 60)

def main():
    """Main function - command-based capture via serial"""
    print("🎥 1080p Command-Based Image Capture")
    print("====================================")
    print("Configuration:")
    print("  • Resolution: 1080p (1920x1080)")
    print("  • Mode: Command-based capture")
    print("  • Trigger: 'func1' via serial port")
    print("  • Storage: collection/yyyy-mm-dd-hh-mm/")
    print("  • Device: /dev/input-device or /tmp/input-virtual")
    print("")
    print("📡 Enhanced Logging & Reliability:")
    print("  • 📥 Shows all received data")
    print("  • 📋 Logs command processing with timing")
    print("  • 📸 Detailed capture workflow")
    print("  • 💾 File save confirmation")
    print("  • 🔄 Processes all buffered commands")
    print("  • ⏱️  Reduced timeouts (100ms → 10ms)")
    print("  • 📊 Command success rate tracking")
    print("")
    
    # Create and run capture
    capture = CommandBasedCapture1080p()
    success = capture.capture_and_record()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)