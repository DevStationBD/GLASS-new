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
from datetime import datetime
from pathlib import Path

import config

# Add GLASS inference path
GLASS_INFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inference')
GLASS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models', 'backbone_0')

if os.path.exists(GLASS_INFERENCE_PATH):
    sys.path.insert(0, GLASS_INFERENCE_PATH)
    GLASS_AVAILABLE = True
else:
    GLASS_AVAILABLE = False
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
        
        # Review state
        self.review_index = 0
        self.images_to_discard = set()
        
        # Testing state
        self.selected_model = None
        self.available_models = []
        self.model_previews = {}
        self.model_selection_index = 0
        
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
        """Main event loop"""
        while True:
            if self.mode == "menu":
                frame = self.draw_menu()
            elif self.mode == "train":
                frame = self.draw_training()
            elif self.mode == "test":
                frame = self.draw_testing()
            elif self.mode == "review":
                frame = self.draw_review()
            elif self.mode == "model_select":
                frame = self.draw_model_selection()
            else:
                frame = self.draw_menu()
            
            cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Handle keyboard input
            if key != 255:
                key_char = chr(key).lower()
                
                if self.mode == "menu":
                    if key_char == self.keys.get('menu', 'quit'):
                        break
                    elif key_char == self.keys.get('menu', 'training_mode'):
                        self.mode = "train"
                        self.start_training_mode()
                    elif key_char == self.keys.get('menu', 'inference_mode'):
                        self.mode = "test"
                        self.start_testing_mode()
                
                elif self.mode == "train":
                    if key_char == self.keys.get('training', 'capture'):
                        self.capture_image()
                    elif key_char == self.keys.get('training', 'submit'):
                        if self.capture_count >= config.MIN_IMAGES_FOR_TRAINING:
                            self.submit_training()
                    elif key_char == self.keys.get('training', 'back_to_menu'):
                        self.mode = "menu"
                        self.cleanup_mode()
                
                elif self.mode == "test":
                    if key_char == self.keys.get('inference', 'reselect_model'):
                        self.selected_model = self.auto_detect_model()
                    elif key_char == self.keys.get('inference', 'back_to_menu'):
                        self.mode = "menu"
                        self.cleanup_mode()
                
                elif self.mode == "review":
                    if key_char == self.keys.get('review', 'next'):
                        self.review_next_image()
                    elif key_char == self.keys.get('review', 'previous'):
                        self.review_previous_image()
                    elif key_char == self.keys.get('review', 'discard'):
                        self.review_discard_current()
                    elif key_char == self.keys.get('review', 'confirm'):
                        self.review_confirm_submit()
                    elif key_char == self.keys.get('review', 'back'):
                        self.mode = "train"
                        self.images_to_discard.clear()
                
                elif self.mode == "model_select":
                    if key_char == self.keys.get('review', 'next') or key_char == 'n':
                        self.model_selection_next()
                    elif key_char == self.keys.get('review', 'previous') or key_char == 'p':
                        self.model_selection_previous()
                    elif key_char == self.keys.get('review', 'confirm') or key_char == 's':
                        self.confirm_model_selection()
                    elif key_char == self.keys.get('review', 'back') or key_char == 'b':
                        self.mode = "menu"
        
        self.cleanup()
    
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
        
        options = [
            f"Press '{training_key}' - Training Mode (Capture & Train)",
            f"Press '{inference_key}' - Inference Mode (Test Models)",
            f"Press '{quit_key}' - Quit Application",
            "",
            f"Datasets: {len(list(Path(config.DATASETS_DIR).glob('batch_*')))} available",
            f"Models: {len(list(Path(config.INFERENCE_DIR).glob('batch_*')))} available",
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
    
    def start_training_mode(self):
        """Initialize training mode"""
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not self.camera.isOpened():
            print("Error: Could not open camera")
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
        dataset_dir = Path(config.DATASETS_DIR) / f"batch_{self.batch_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        img_path = dataset_dir / f"img_{self.capture_count + 1:04d}.{config.IMAGE_FORMAT}"
        cv2.imwrite(str(img_path), self.current_frame)
        
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
        
        print(f"\nStarting training with {self.capture_count} images (discarded {discarded_count})...")
        self.training_logs.append(f"Training with {self.capture_count} images...")
        self.training_in_progress = True
        
        # Clear review state
        self.images_to_discard.clear()
        self.review_index = 0
        
        # Return to training mode display during training
        self.mode = "train"
        
        # Run training in thread
        thread = threading.Thread(target=self.run_training, daemon=True)
        thread.start()
    
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
        # Models are inside the backbone_0 directory
        models_dir = Path(GLASS_MODEL_PATH)
        eval_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'results' / 'eval'
        
        self.available_models = []
        self.model_previews = {}
        
        # Find all model directories inside backbone_0
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir() and not model_path.name.startswith('.'):
                    model_name = model_path.name
                    self.available_models.append(model_name)
                    
                    # Look for corresponding preview image in eval folder
                    eval_model_dir = eval_dir / model_name
                    preview_image = None
                    
                    if eval_model_dir.exists():
                        # Find first image file in the eval directory
                        for img_file in eval_model_dir.iterdir():
                            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                                preview_image = str(img_file)
                                break
                    
                    self.model_previews[model_name] = preview_image
        
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
            return
        
        # Fallback to simple OpenCV mode
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        if not self.camera.isOpened():
            print("Error: Could not open camera")
            self.mode = "menu"
            return
        
        print(f"Testing Mode Active with model: {selected_model_name}")
        self.mode = "test"
    
    def run_glass_inference(self):
        """Run GLASS real-time inference with camera using orchestrator"""
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
            
            # Initialize GLASS inference orchestrator
            orchestrator = GLASSInferenceOrchestrator(
                models_base_path=GLASS_MODEL_PATH,
                device='cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
                image_size=384,
                sample_frames=30
            )
            
            print("\n" + "="*60)
            print("GLASS Real-time Inference Active")
            print("Press 'q' in the preview window to stop")
            print("="*60 + "\n")
            
            # Run camera inference with selected model class
            results = orchestrator.run_inference_with_camera(
                camera_id=config.CAMERA_INDEX,
                camera_fps=config.CAMERA_FPS,
                duration_seconds=None,  # Continuous until stopped
                skip_model_selection=False,
                manual_class_name=class_name,
                output_path=None
            )
            
            print("\n" + "="*60)
            print("GLASS Inference Complete")
            print(f"Selected Model: {results['orchestrator_info']['selected_model']}")
            if 'unique_defects' in results['inference_results']:
                print(f"Unique defects found: {results['inference_results']['unique_defects']}")
            if 'fps_processing' in results['inference_results']:
                print(f"Processing FPS: {results['inference_results']['fps_processing']:.1f}")
            print("="*60 + "\n")
            
            # Generate PDF report
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
            
            # Find the tracking report JSON file
            inference_results = results.get('inference_results', {})
            
            # Look for tracking report in the output directory
            # The orchestrator should have created organized output
            if 'output_path' in inference_results:
                output_dir = Path(inference_results['output_path']).parent
                report_dir = output_dir / 'report'
                
                # Look for tracking report JSON
                tracking_report_files = list(report_dir.glob('*_tracking_report.json'))
                
                if tracking_report_files:
                    json_path = str(tracking_report_files[0])
                    print(f"📄 Generating PDF report from: {json_path}")
                    
                    # Generate PDF report
                    generator = GLASSReportGenerator()
                    pdf_path = generator.generate_report(json_path, open_after=True)
                    
                    print(f"✅ PDF report saved to Documents folder: {pdf_path}")
                    
                else:
                    print("⚠️  No tracking report JSON found for PDF generation")
            else:
                print("⚠️  No output path found in inference results")
                
        except ImportError as e:
            print(f"⚠️  Could not generate PDF report: {e}")
            print("Install reportlab with: pip install reportlab")
        except Exception as e:
            print(f"⚠️  Error generating PDF report: {e}")
    
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
        """Cleanup and exit"""
        self.cleanup_mode()
        
        if self.serial_listener:
            self.serial_listener.stop()
        
        cv2.destroyAllWindows()
        print("\nFabric Inspector closed. Goodbye!")


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
