#!/usr/bin/env python3
"""
GLASS Video Inference with Defect Tracking

Enhanced video inference that integrates temporal defect tracking capabilities
for continuous fabric inspection with comprehensive reporting.
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime
from tqdm import tqdm
import time
import logging
import gc
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backbones
import glass
import common
from torchvision import transforms
from size_analyzer import DefectSizeAnalyzer

# Import our new tracking system
from defect_tracking import GLASSDefectTracker, FabricMotionEstimator, MotionEstimationMethod

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoInferenceWithTracking:
    """Enhanced video inference with defect tracking capabilities"""
    
    def __init__(self, 
                 model_path: str,
                 class_name: str,
                 device: str = 'cuda:0',
                 image_size: int = 384,
                 pixel_size: float = 0.1,
                 physical_unit: str = "mm",
                 fabric_speed_estimate: float = 5.0,
                 save_defect_frames: bool = True,
                 use_organized_output: bool = True,
                 show_preview: bool = True):
        """Initialize tracking-enabled video inference"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.physical_unit = physical_unit
        self.save_defect_frames = save_defect_frames
        self.use_organized_output = use_organized_output
        self.show_preview = show_preview
        
        # Initialize organized output directories
        self.output_base_dir = None
        self.defect_frames_dir = None
        
        # Memory-efficient defect tracking
        self.saved_track_frames = set()
        self.track_frame_counts = {}  # Track ID -> list of frame numbers
        self.track_middle_frames = {}  # Track ID -> current middle frame data
        self.max_stored_tracks = 50   # Maximum concurrent tracks to store frames for
        self.completed_tracks = set()  # Track IDs that have been completed and saved
        
        # Load GLASS model
        self.glass_model = self._load_model()
        
        # Initialize defect tracker
        self.defect_tracker = GLASSDefectTracker(
            pixel_size_mm=pixel_size,
            initial_fabric_speed=fabric_speed_estimate,
            high_conf_threshold=0.7,
            low_conf_threshold=0.3,
            min_defect_area=10,
            detection_threshold=0.5
        )
        
        # Initialize motion estimator
        self.motion_estimator = FabricMotionEstimator(
            method=MotionEstimationMethod.OPTICAL_FLOW_SPARSE
        )
        
        # Image preprocessing to match training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Statistics
        self.total_frames_processed = 0
        self.processing_start_time = None
        
        logger.info(f"VideoInferenceWithTracking initialized for class: {class_name}")
    
    def _load_model(self):
        """Load GLASS model"""
        glass_model = glass.GLASS(self.device)
        backbone = backbones.load('wideresnet50')
        
        glass_model.load(
            backbone=backbone,
            layers_to_extract_from=['layer2', 'layer3'],
            device=self.device,
            input_shape=(3, self.image_size, self.image_size),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=10,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
        )
        
        # Load checkpoint
        import glob
        ckpt_path = glob.glob(os.path.join(self.model_path, f'{self.class_name}/ckpt_best*'))
        if len(ckpt_path) == 0:
            ckpt_path = glob.glob(os.path.join(self.model_path, f'{self.class_name}/ckpt.pth'))
        
        if len(ckpt_path) > 0:
            logger.info(f"Loading checkpoint: {ckpt_path[0]}")
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            
            if 'discriminator' in state_dict:
                glass_model.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    glass_model.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                glass_model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"No checkpoint found for {self.class_name}")
        
        glass_model.eval()
        return glass_model
    
    def predict_frame(self, frame: np.ndarray):
        """Run GLASS inference on single frame"""
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use the GLASS model's _predict method which handles the full pipeline
            image_scores, masks = self.glass_model._predict(input_tensor)
            
            # Get the first (and only) result from the batch
            anomaly_map = masks[0]
            anomaly_score = float(image_scores[0])
            
            # Resize anomaly map to match frame dimensions
            if isinstance(anomaly_map, torch.Tensor):
                anomaly_map = anomaly_map.cpu().numpy()
            
            # Ensure the anomaly map is in the right format and size
            if anomaly_map.shape != (frame.shape[0], frame.shape[1]):
                anomaly_map = cv2.resize(anomaly_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return anomaly_map, anomaly_score
    
    def _save_defect_frames(self, frame: np.ndarray, active_tracks: list, frame_count: int, anomaly_map: np.ndarray):
        """Memory-efficient defect frame tracking - only stores current middle frames"""
        if not self.save_defect_frames or not active_tracks:
            return
            
        for track in active_tracks:
            track_id = track.track_id
            
            # Track frame numbers for each defect
            if track_id not in self.track_frame_counts:
                self.track_frame_counts[track_id] = []
            self.track_frame_counts[track_id].append(frame_count)
            
            # Calculate current middle frame index
            current_frames = self.track_frame_counts[track_id]
            middle_idx = len(current_frames) // 2
            middle_frame_num = current_frames[middle_idx]
            
            # Only store/update if this is the new middle frame
            if frame_count == middle_frame_num:
                # Clean up old middle frame data if exists
                if track_id in self.track_middle_frames:
                    old_data = self.track_middle_frames[track_id]
                    # Explicitly delete large arrays
                    if 'frame' in old_data:
                        del old_data['frame']
                    if 'anomaly_map' in old_data:
                        del old_data['anomaly_map']
                    del old_data
                
                # Store new middle frame (only one per track)
                self.track_middle_frames[track_id] = {
                    'frame': frame.copy(),
                    'track': track, 
                    'frame_count': frame_count,
                    'anomaly_map': anomaly_map.copy(),
                    'middle_frame_num': middle_frame_num
                }
            
        # Memory safety: enforce maximum stored tracks
        if len(self.track_middle_frames) > self.max_stored_tracks:
            logger.warning(f"Reached maximum stored tracks ({self.max_stored_tracks}). Cleaning up oldest tracks.")
            self._cleanup_oldest_stored_frames()
            
        # Clean up completed tracks immediately
        self._save_and_cleanup_completed_tracks()
    
    def _cleanup_oldest_stored_frames(self):
        """Remove oldest stored frames to free memory"""
        if len(self.track_middle_frames) <= self.max_stored_tracks:
            return
            
        # Sort by frame count (older frames first)
        sorted_tracks = sorted(self.track_middle_frames.items(), 
                             key=lambda x: x[1]['frame_count'])
        
        # Remove oldest frames until we're under the limit
        num_to_remove = len(self.track_middle_frames) - self.max_stored_tracks
        for i in range(num_to_remove):
            track_id, frame_data = sorted_tracks[i]
            
            # Clean up memory
            del frame_data['frame']
            del frame_data['anomaly_map']
            del self.track_middle_frames[track_id]
            
            logger.debug(f"Cleaned up stored frame for track {track_id} due to memory limit")
    
    def _save_and_cleanup_completed_tracks(self):
        """Save and immediately clean up completed tracks"""
        if not self.save_defect_frames:
            return
            
        # Get current active track IDs
        current_active_tracks = set()
        defect_summaries = self.defect_tracker.get_tracked_defect_summaries()
        for summary in defect_summaries:
            if summary.track_id in self.track_middle_frames:
                current_active_tracks.add(summary.track_id)
        
        # Find completed tracks that have stored frames but are no longer active
        completed_tracks = []
        for track_id in list(self.track_middle_frames.keys()):
            # If we have a stored frame but the track is not in summaries, it's completed
            if track_id not in current_active_tracks and track_id not in self.completed_tracks:
                completed_tracks.append(track_id)
        
        # Save and cleanup completed tracks immediately
        for track_id in completed_tracks:
            if track_id in self.track_middle_frames:
                frame_data = self.track_middle_frames[track_id]
                
                # Create and save annotated frame
                annotated_frame = self._create_defect_frame_annotation(
                    frame_data['frame'], 
                    frame_data['track'], 
                    frame_data['frame_count'], 
                    frame_data['anomaly_map']
                )
                
                # Save with track ID in filename
                filename = f"track_{track_id:03d}_middle_frame_{frame_data['middle_frame_num']:06d}.jpg"
                output_path = os.path.join(self.defect_frames_dir, filename)
                
                success = cv2.imwrite(output_path, annotated_frame)
                if success:
                    logger.info(f"💾 Saved completed track frame: {filename}")
                    self.completed_tracks.add(track_id)
                    
                    # Clean up memory immediately  
                    del frame_data['frame']
                    del frame_data['anomaly_map']
                    del self.track_middle_frames[track_id]
                else:
                    logger.warning(f"Failed to save defect frame: {filename}")
    
    def _save_middle_frames_for_completed_tracks(self):
        """Save any remaining middle frames that weren't saved during processing"""
        if not self.save_defect_frames:
            return
            
        # Save any remaining middle frames that are still in memory
        remaining_tracks = list(self.track_middle_frames.keys())
        for track_id in remaining_tracks:
            if track_id not in self.completed_tracks:
                frame_data = self.track_middle_frames[track_id]
                
                # Create and save annotated frame
                annotated_frame = self._create_defect_frame_annotation(
                    frame_data['frame'], 
                    frame_data['track'], 
                    frame_data['frame_count'], 
                    frame_data['anomaly_map']
                )
                
                # Save with track ID in filename
                filename = f"track_{track_id:03d}_middle_frame_{frame_data['middle_frame_num']:06d}.jpg"
                output_path = os.path.join(self.defect_frames_dir, filename)
                
                success = cv2.imwrite(output_path, annotated_frame)
                if success:
                    logger.info(f"💾 Saved final track frame: {filename}")
                    self.completed_tracks.add(track_id)
                    
                    # Clean up memory
                    del frame_data['frame']
                    del frame_data['anomaly_map'] 
                    del self.track_middle_frames[track_id]
                else:
                    logger.warning(f"Failed to save final defect frame: {filename}")
    
    def _create_defect_frame_annotation(self, frame: np.ndarray, track, frame_count: int, anomaly_map: np.ndarray):
        """Create annotated frame with track information"""
        annotated_frame = frame.copy()
        
        # Get track information
        track_id = track.track_id
        confidence = track.confidence if hasattr(track, 'confidence') else 0.0
        area_mm2 = track.area_physical if hasattr(track, 'area_physical') else 0.0
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        color = (0, 255, 255)  # Yellow color for visibility
        bg_color = (0, 0, 0, 120)   # Black background with transparency
        
        # Prepare text lines
        text_lines = [
            f"Track ID: {track_id}",
            f"Frame: {frame_count}",
            # f"Confidence: {confidence:.3f}",
            # f"Area: {area_mm2:.1f}{self.physical_unit}²"
        ]
        
        # Calculate text dimensions
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
        max_width = max(size[0] for size in text_sizes)
        total_height = sum(size[1] for size in text_sizes) + len(text_lines) * 10 + 20
        
        # Create semi-transparent overlay for text background
        padding = 15
        overlay = annotated_frame.copy()
        rect_start = (padding, padding)
        rect_end = (max_width + 2*padding, total_height + padding)
        cv2.rectangle(overlay, rect_start, rect_end, bg_color[:3], -1)
        cv2.rectangle(overlay, rect_start, rect_end, color, 2)
        
        # Apply transparency (0.7 = 70% original, 30% overlay)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        # Add text lines
        y_offset = padding + 30
        for i, (line, text_size) in enumerate(zip(text_lines, text_sizes)):
            text_pos = (padding + 10, y_offset + i * (text_size[1] + 15))
            cv2.putText(annotated_frame, line, text_pos, font, font_scale, color, thickness)
        
        # Visualize defect area from anomaly map
        threshold = 0.5
        defect_mask = (anomaly_map > threshold).astype(np.uint8) * 255
        
        # Find contours of defect areas
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw defect boundaries and add track ID
        if contours:
            # Find the largest contour (main defect)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw defect boundary
            cv2.drawContours(annotated_frame, [largest_contour], -1, color, 3)

            # Get bounding box for track ID placement
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add track ID inside the defect area
            track_text = f"#{track_id}"
            text_size = cv2.getTextSize(track_text, font, 1.0, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            # Background padding
            bg_padding = 5
            (x1, y1) = (text_x - bg_padding, text_y - text_size[1] - bg_padding)
            (x2, y2) = (text_x + text_size[0] + bg_padding, text_y + bg_padding)

            # Make a copy of frame for overlay
            overlay = annotated_frame.copy()

            # Draw solid black rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # Blend with transparency (70% bg, 30% original frame)
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Finally draw the text (keeps it crisp)
            cv2.putText(annotated_frame, track_text, (text_x, text_y),
                        font, 1.0, color, 2)
        
        # Fallback: use track bbox if available
        elif hasattr(track, 'bbox') and track.bbox is not None:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Add track ID inside the defect boundary
            track_text = f"#{track_id}"
            text_size = cv2.getTextSize(track_text, font, 1.0, 2)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            
            # Add background for track ID
            bg_padding = 5
            cv2.rectangle(annotated_frame, 
                         (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                         (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                         (0, 0, 0), -1)
            cv2.putText(annotated_frame, track_text, (text_x, text_y), 
                       font, 1.0, color, 2)
        
        return annotated_frame
    
    def _setup_organized_output(self, output_path: str = None) -> tuple:
        """Setup organized output directory structure"""
        if self.use_organized_output:
            # Create timestamp-based directory structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_base_dir = f"output/{self.class_name}/{timestamp}"
            
            # Create subdirectories
            video_dir = os.path.join(self.output_base_dir, "output-video")
            self.defect_frames_dir = os.path.join(self.output_base_dir, "defects")
            report_dir = os.path.join(self.output_base_dir, "report")
            
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            if self.save_defect_frames:
                os.makedirs(self.defect_frames_dir, exist_ok=True)
                logger.info(f"Defect frames will be saved to: {self.defect_frames_dir}")
            
            # Generate organized output path
            if output_path is None:
                output_filename = f"{self.class_name}_tracked_inference.mp4"
            else:
                output_filename = os.path.basename(output_path)
            
            organized_output_path = os.path.join(video_dir, output_filename)
            logger.info(f"Organized output directory: {self.output_base_dir}")
            
            return organized_output_path, self.defect_frames_dir
        else:
            # Use provided paths or defaults
            if output_path is None:
                output_path = "output_tracked.mp4"
            
            if self.save_defect_frames:
                if self.defect_frames_dir is None:
                    self.defect_frames_dir = "defect_frames"
                os.makedirs(self.defect_frames_dir, exist_ok=True)
                logger.info(f"Defect frames will be saved to: {self.defect_frames_dir}")
            
            return output_path, self.defect_frames_dir
    
    def process_video(self, video_path: str, output_path: str = None, **kwargs):
        """Process video with defect tracking"""
        self.processing_start_time = time.time()
        
        # Setup organized output structure
        organized_output_path, organized_defect_dir = self._setup_organized_output(output_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(organized_output_path, fourcc, fps, (width * 2, height))
        
        self.defect_tracker.reset()
        self.motion_estimator.reset()
        
        frame_count = 0
        active_tracks_history = []
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        # Setup preview window if enabled
        if self.show_preview:
            window_name = f"GLASS Real-time Inference - {self.class_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1920, width * 2), min(1080, height))
            logger.info(f"📺 Live preview enabled: Press 'q' to quit, 's' to skip preview")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                motion_estimate = self.motion_estimator.estimate_motion(frame)
                fabric_motion = motion_estimate.displacement
                
                anomaly_map, anomaly_score = self.predict_frame(frame)
                
                active_tracks = self.defect_tracker.process_frame(frame, anomaly_map, fabric_motion)
                active_tracks_history.append(len(active_tracks))
                
                # Save defect frames if enabled
                self._save_defect_frames(frame, active_tracks, frame_count, anomaly_map)
                
                vis_frame = self._create_visualization(frame, anomaly_map, active_tracks, motion_estimate, frame_count)
                out.write(vis_frame)
                
                # Display real-time preview
                if self.show_preview:
                    # Create resized preview for display (to avoid window being too large)
                    preview_height = 600
                    preview_width = int(vis_frame.shape[1] * preview_height / vis_frame.shape[0])
                    preview_frame = cv2.resize(vis_frame, (preview_width, preview_height))
                    
                    cv2.imshow(window_name, preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("🛑 User requested quit via 'q' key")
                        break
                    elif key == ord('s'):
                        logger.info("⏭️ Skipping preview display")
                        self.show_preview = False
                        cv2.destroyWindow(window_name)
                
                self.total_frames_processed += 1
                pbar.update(1)
                
                # Periodic memory cleanup and monitoring  
                if frame_count % 100 == 0:
                    logger.debug(f"Frame {frame_count}: {len(active_tracks)} active tracks")
                    
                    # Memory monitoring and cleanup
                    memory_info = psutil.virtual_memory()
                    stored_tracks = len(self.track_middle_frames)
                    
                    logger.debug(f"Memory: {memory_info.percent:.1f}% used, "
                               f"stored tracks: {stored_tracks}/{self.max_stored_tracks}")
                    
                    # Force cleanup every 100 frames to prevent memory buildup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Emergency cleanup if memory usage is high
                    if memory_info.percent > 85:
                        logger.warning(f"High memory usage: {memory_info.percent:.1f}% "
                                     f"({memory_info.used / (1024**3):.1f}GB used)")
                        
                        # Additional emergency cleanup
                        if memory_info.percent > 90 and stored_tracks > 20:
                            logger.warning("Emergency memory cleanup: reducing stored tracks")
                            self.max_stored_tracks = min(20, self.max_stored_tracks)
                            self._cleanup_oldest_stored_frames()
                            gc.collect()
                
                # Explicit cleanup of large variables each frame
                del anomaly_map
                if 'motion_estimate' in locals():
                    del motion_estimate
        
        finally:
            cap.release()
            out.release()
            pbar.close()
            
            # Cleanup preview window
            if self.show_preview:
                cv2.destroyAllWindows()
        
        processing_time = time.time() - self.processing_start_time
        
        # Save middle frames for all completed tracks
        self._save_middle_frames_for_completed_tracks()
        
        results = self._generate_report(video_path, organized_output_path, processing_time, active_tracks_history)
        
        logger.info(f"Processing complete! Processed {self.total_frames_processed} frames in {processing_time:.2f}s")
        logger.info(f"Found {results['unique_defects']} unique defects")
        
        return results
    
    def _create_visualization(self, frame, anomaly_map, active_tracks, motion_estimate, frame_count):
        """Create side-by-side visualization"""
        anomaly_colored = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.4, anomaly_colored, 0.6, 0)
        overlay_with_tracks = self.defect_tracker.visualize_tracks_on_frame(overlay, active_tracks)
        
        info_text = [
            f"Frame: {frame_count}",
            f"Active Tracks: {len(active_tracks)}",
            f"Motion: ({motion_estimate.displacement[0]:.1f}, {motion_estimate.displacement[1]:.1f})",
            f"Speed: {motion_estimate.speed_pixels_per_frame:.1f} px/frame",
            f"Confidence: {motion_estimate.confidence:.2f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(overlay_with_tracks, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay_with_tracks, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return np.hstack([frame, overlay_with_tracks])
    
    def _generate_report(self, video_path, output_path, processing_time, active_tracks_history):
        """Generate comprehensive tracking report"""
        defect_summaries = self.defect_tracker.get_tracked_defect_summaries()
        tracking_stats = self.defect_tracker.get_tracking_statistics()
        
        report = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(video_path),
            'output_file': os.path.basename(output_path),
            'frames_processed': self.total_frames_processed,
            'processing_time_seconds': processing_time,
            'fps_processing': self.total_frames_processed / processing_time,
            'unique_defects': len(defect_summaries),
            'peak_concurrent_defects': max(active_tracks_history) if active_tracks_history else 0,
            'tracking_statistics': tracking_stats,
            'tracked_defects': [
                {
                    'track_id': s.track_id,
                    'defect_type': s.defect_type,
                    'first_frame': s.first_frame,
                    'last_frame': s.last_frame,
                    'duration_frames': s.duration_frames,
                    'max_confidence': s.max_confidence,
                    'max_area_mm2': s.max_area_physical,
                    'trajectory_length': len(s.centroid_trajectory)
                } for s in defect_summaries
            ]
        }
        
        # Save tracking report in organized structure
        if self.use_organized_output and self.output_base_dir:
            report_dir = os.path.join(self.output_base_dir, "report")
            report_path = os.path.join(report_dir, f"{self.class_name}_tracking_report.json")
        else:
            report_path = output_path.replace('.mp4', '_tracking_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Tracking report saved to: {report_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description='GLASS Video Inference with Defect Tracking')
    parser.add_argument('--model_path', type=str, required=True, help='Path to GLASS model')
    parser.add_argument('--class_name', type=str, required=True, help='Model class name')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, help='Output video path (optional with organized output)')
    parser.add_argument('--pixel_size', type=float, default=0.1, help='Pixel size in mm')
    parser.add_argument('--fabric_speed', type=float, default=5.0, help='Initial fabric speed estimate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--image_size', type=int, default=384, help='Model input image size')
    parser.add_argument('--no_save_defect_frames', action='store_true', help='Disable saving individual defect frames')
    parser.add_argument('--no_organized_output', action='store_true', help='Disable organized output structure')
    parser.add_argument('--no_preview', action='store_true', help='Disable real-time preview window')
    
    args = parser.parse_args()
    
    inference_system = VideoInferenceWithTracking(
        model_path=args.model_path,
        class_name=args.class_name,
        device=args.device,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        fabric_speed_estimate=args.fabric_speed,
        save_defect_frames=not args.no_save_defect_frames,
        use_organized_output=not args.no_organized_output,
        show_preview=not args.no_preview
    )
    
    results = inference_system.process_video(
        video_path=args.video_path,
        output_path=args.output_path
    )
    
    print("\n=== GLASS Defect Tracking Results ===")
    print(f"Unique defects found: {results['unique_defects']}")
    print(f"Processing time: {results['processing_time_seconds']:.2f}s")
    print(f"Processing FPS: {results['fps_processing']:.1f}")


if __name__ == '__main__':
    main()