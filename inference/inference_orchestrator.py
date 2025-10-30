#!/usr/bin/env python3
"""
GLASS Inference Orchestrator

Automatically selects the best model and class for a given video by testing
all available models and choosing the one with the lowest average anomaly score.
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
import time
import logging
from typing import List, Dict, Tuple, Optional
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backbones
import glass
import common
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCandidate:
    """Represents a model candidate with its performance metrics"""
    
    def __init__(self, model_path: str, class_name: str, model_dir: str):
        self.model_path = model_path
        self.class_name = class_name
        self.model_dir = model_dir
        self.avg_anomaly_score = None
        self.frame_scores = []
        self.is_loaded = False
        self.glass_model = None
        
    def __str__(self):
        return f"ModelCandidate(class={self.class_name}, avg_score={self.avg_anomaly_score:.4f if self.avg_anomaly_score else 'N/A'})"

class GLASSInferenceOrchestrator:
    """Orchestrator that automatically selects the best model for a video"""
    
    def _log_gpu_info(self):
        """Log comprehensive GPU information"""
        logger.info("=" * 60)
        logger.info("🔍 GPU DETECTION AND SETUP")
        logger.info("=" * 60)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU Count: {gpu_count}")
            
            # Log all available GPUs
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("No CUDA GPUs detected")
            
        # Log PyTorch versions
        logger.info(f"PyTorch Version: {torch.__version__}")
        if cuda_available:
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
            if torch.backends.cudnn.is_available():
                logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        logger.info("=" * 60)
    
    def __init__(self, 
                 models_base_path: str = "results/models/backbone_0",
                 device: str = 'cuda:0',
                 image_size: int = 384,
                 sample_frames: int = 30):
        """Initialize the orchestrator"""
        self.models_base_path = models_base_path
        
        # GPU Detection and Device Setup
        self._log_gpu_info()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            logger.info(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"🔥 CUDA Version: {torch.version.cuda}")
            logger.info(f"⚡ cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            logger.warning("⚠️  Running on CPU - GPU not available or not detected")
        
        self.image_size = image_size
        self.sample_frames = sample_frames
        
        # Image preprocessing to match training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Discover available models
        self.available_models = self._discover_models()
        logger.info(f"Found {len(self.available_models)} available models")
        
    def _discover_models(self) -> List[ModelCandidate]:
        """Discover all available trained models"""
        models = []
        
        if not os.path.exists(self.models_base_path):
            logger.warning(f"Models base path does not exist: {self.models_base_path}")
            return models
            
        # Scan for class directories with checkpoints
        for class_dir in os.listdir(self.models_base_path):
            class_path = os.path.join(self.models_base_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Check for checkpoint files
            ckpt_best = glob.glob(os.path.join(class_path, "ckpt_best*.pth"))
            ckpt_regular = glob.glob(os.path.join(class_path, "ckpt.pth"))
            
            if ckpt_best or ckpt_regular:
                model = ModelCandidate(
                    model_path=self.models_base_path,
                    class_name=class_dir,
                    model_dir=class_path
                )
                models.append(model)
                logger.debug(f"Found model: {class_dir}")
                
        return sorted(models, key=lambda x: x.class_name)

    def get_model_by_class_name(self, class_name: str) -> Optional[ModelCandidate]:
        """Manually get a specific model by class name"""
        for model in self.available_models:
            if model.class_name == class_name:
                logger.info(f"✅ Manually selected model: {class_name}")
                return model

        # Model not found, show available models
        available_classes = [m.class_name for m in self.available_models]
        logger.error(f"❌ Model '{class_name}' not found. Available models: {', '.join(available_classes)}")
        return None

    def _load_glass_model(self, model_candidate: ModelCandidate):
        """Load GLASS model for a specific candidate"""
        if model_candidate.is_loaded:
            return model_candidate.glass_model
            
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
        
        # Load checkpoint (prefer best, fallback to regular)
        ckpt_best = glob.glob(os.path.join(model_candidate.model_dir, "ckpt_best*.pth"))
        ckpt_regular = glob.glob(os.path.join(model_candidate.model_dir, "ckpt.pth"))
        
        ckpt_path = ckpt_best[0] if ckpt_best else (ckpt_regular[0] if ckpt_regular else None)
        
        if ckpt_path:
            logger.debug(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            
            if 'discriminator' in state_dict:
                glass_model.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    glass_model.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                glass_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No checkpoint found for {model_candidate.class_name}")
        
        glass_model.eval()
        model_candidate.glass_model = glass_model
        model_candidate.is_loaded = True
        
        # Log GPU memory usage after model loading
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            logger.info(f"🧠 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        return glass_model
    
    def _predict_frame(self, frame: np.ndarray, model_candidate: ModelCandidate) -> float:
        """Run inference on a single frame with a specific model"""
        glass_model = self._load_glass_model(model_candidate)

        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Log GPU tensor transfer
        if self.device.type == 'cuda':
            logger.debug(f"⚡ Tensor moved to GPU: {input_tensor.shape} -> {self.device}")
        
        with torch.no_grad():
            start_time = time.time()
            image_scores, masks = glass_model._predict(input_tensor)
            inference_time = time.time() - start_time
            anomaly_score = float(image_scores[0])
            
            # Log inference performance
            if self.device.type == 'cuda':
                logger.debug(f"🚀 GPU Inference: {inference_time*1000:.1f}ms, Score: {anomaly_score:.4f}")
            else:
                logger.debug(f"🐌 CPU Inference: {inference_time*1000:.1f}ms, Score: {anomaly_score:.4f}")
            
        return anomaly_score
    
    def _sample_camera_frames(self, camera_id: int, camera_fps: int = 30) -> List[np.ndarray]:
        """Sample frames from camera for model evaluation"""
        logger.info(f"📹 Sampling {self.sample_frames} frames from camera {camera_id} for model selection")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}. Please check camera connection.")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FPS, camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set reasonable resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frames = []
        warmup_frames = 10  # Skip first few frames to let camera adjust
        frame_interval = max(1, camera_fps // 2)  # Sample every 0.5 seconds
        
        try:
            # Warmup camera
            for _ in range(warmup_frames):
                ret, _ = cap.read()
                if not ret:
                    raise ValueError("Camera stream ended during warmup")
            
            logger.info("🔄 Camera warmed up, collecting sample frames...")
            frame_count = 0
            collected_frames = 0
            
            while collected_frames < self.sample_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera stream ended unexpectedly")
                    break
                
                # Sample frame at intervals
                if frame_count % frame_interval == 0:
                    frames.append(frame.copy())
                    collected_frames += 1
                    
                    if collected_frames % 5 == 0:
                        logger.info(f"  Collected {collected_frames}/{self.sample_frames} frames")
                
                frame_count += 1
                
                # Safety timeout (max 30 seconds of sampling)
                if frame_count > camera_fps * 30:
                    logger.warning("Camera sampling timeout reached")
                    break
                    
        finally:
            cap.release()
        
        if len(frames) < 5:
            raise ValueError(f"Could not collect enough frames from camera (got {len(frames)}, need at least 5)")
            
        logger.info(f"✅ Collected {len(frames)} frames from camera for model selection")
        return frames
    
    def _sample_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Sample frames from video for model evaluation"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample evenly
        if total_frames <= self.sample_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // self.sample_frames
            frame_indices = [i * step for i in range(self.sample_frames)]
        
        frames = []
        current_frame = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if current_frame in frame_indices:
                    frames.append(frame.copy())
                    
                current_frame += 1
                
                if len(frames) >= self.sample_frames:
                    break
                    
        finally:
            cap.release()
            
        logger.info(f"Sampled {len(frames)} frames from video ({total_frames} total frames)")
        return frames
    
    def _evaluate_model_on_frames(self, frames: List[np.ndarray], model_candidate: ModelCandidate) -> Dict:
        """Evaluate a model on sampled frames"""
        logger.info(f"Evaluating model: {model_candidate.class_name}")
        
        start_time = time.time()
        scores = []
        
        for i, frame in enumerate(frames):
            try:
                anomaly_score = self._predict_frame(frame, model_candidate)
                scores.append(anomaly_score)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Frame {i+1}/{len(frames)}: score={anomaly_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Failed to process frame {i} with model {model_candidate.class_name}: {e}")
                continue
        
        evaluation_time = time.time() - start_time
        
        if not scores:
            logger.error(f"No valid scores for model {model_candidate.class_name}")
            return None
            
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        model_candidate.avg_anomaly_score = avg_score
        model_candidate.frame_scores = scores
        
        result = {
            'class_name': model_candidate.class_name,
            'avg_anomaly_score': avg_score,
            'std_anomaly_score': std_score,
            'min_anomaly_score': min_score,
            'max_anomaly_score': max_score,
            'frames_evaluated': len(scores),
            'evaluation_time_seconds': evaluation_time,
            'model_dir': model_candidate.model_dir
        }
        
        logger.info(f"  Model {model_candidate.class_name}: avg_score={avg_score:.4f} (±{std_score:.4f})")
        return result
    
    def select_best_model_from_camera(self, camera_id: int, camera_fps: int = 30) -> Tuple[ModelCandidate, Dict]:
        """Select the best model for camera input"""
        logger.info(f"🎯 Starting model selection for camera {camera_id}")
        
        # Sample frames from camera
        frames = self._sample_camera_frames(camera_id, camera_fps)
        if not frames:
            raise ValueError("No frames could be extracted from camera")
        
        # Evaluate each model
        evaluation_results = []
        best_model = None
        best_score = float('inf')
        
        for model_candidate in self.available_models:
            try:
                result = self._evaluate_model_on_frames(frames, model_candidate)
                if result is None:
                    continue
                    
                evaluation_results.append(result)
                
                # Track best model (lowest anomaly score = best fit)
                if result['avg_anomaly_score'] < best_score:
                    best_score = result['avg_anomaly_score']
                    best_model = model_candidate
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate model {model_candidate.class_name}: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("No models could be successfully evaluated")
        
        # Generate selection report
        selection_report = {
            'timestamp': datetime.now().isoformat(),
            'input_source': f'camera_{camera_id}',
            'camera_fps': camera_fps,
            'selected_model': {
                'class_name': best_model.class_name,
                'avg_anomaly_score': best_model.avg_anomaly_score,
                'model_dir': best_model.model_dir
            },
            'all_evaluations': evaluation_results,
            'selection_criteria': 'lowest_average_anomaly_score',
            'frames_sampled': len(frames)
        }
        
        logger.info(f"✅ Selected best model: {best_model.class_name} (score: {best_score:.4f})")
        return best_model, selection_report

    def select_best_model(self, video_path: str) -> Tuple[ModelCandidate, Dict]:
        """Select the best model for the given video"""
        logger.info(f"Starting model selection for video: {video_path}")
        
        # Sample frames from video
        frames = self._sample_video_frames(video_path)
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Evaluate each model
        evaluation_results = []
        best_model = None
        best_score = float('inf')
        
        for model_candidate in self.available_models:
            try:
                result = self._evaluate_model_on_frames(frames, model_candidate)
                if result is None:
                    continue
                    
                evaluation_results.append(result)
                
                # Track best model (lowest anomaly score = best fit)
                if result['avg_anomaly_score'] < best_score:
                    best_score = result['avg_anomaly_score']
                    best_model = model_candidate
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate model {model_candidate.class_name}: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("No models could be successfully evaluated")
        
        # Generate selection report
        selection_report = {
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(video_path),
            'selected_model': {
                'class_name': best_model.class_name,
                'avg_anomaly_score': best_model.avg_anomaly_score,
                'model_dir': best_model.model_dir
            },
            'all_evaluations': evaluation_results,
            'selection_criteria': 'lowest_average_anomaly_score',
            'frames_sampled': len(frames)
        }
        
        logger.info(f"✅ Selected best model: {best_model.class_name} (score: {best_score:.4f})")
        return best_model, selection_report
    
    def run_inference_with_best_model(self, video_path: str, output_path: str = None, manual_class_name: str = None) -> Dict:
        """Run complete inference workflow with automatic or manual model selection"""
        logger.info("🚀 Starting GLASS Inference Orchestrator")

        # Step 1: Select model (manual or automatic)
        if manual_class_name:
            logger.info(f"🎯 Using manually specified model: {manual_class_name}")
            best_model = self.get_model_by_class_name(manual_class_name)
            if best_model is None:
                raise ValueError(f"Model '{manual_class_name}' not found")

            selection_report = {
                'timestamp': datetime.now().isoformat(),
                'video_file': os.path.basename(video_path),
                'selected_model': {
                    'class_name': best_model.class_name,
                    'model_dir': best_model.model_dir
                },
                'selection_criteria': 'manual_selection',
                'frames_sampled': 0
            }
        else:
            # Automatic selection
            best_model, selection_report = self.select_best_model(video_path)
        
        # Step 2: Run full inference with selected model
        logger.info(f"Running full inference with model: {best_model.class_name}")
        
        # Import video inference with tracking
        from video_inference_with_tracking import VideoInferenceWithTracking
        
        # Create inference system with selected model
        inference_system = VideoInferenceWithTracking(
            model_path=best_model.model_path,
            class_name=best_model.class_name,
            device=str(self.device),
            image_size=self.image_size,
            save_defect_frames=True,
            use_organized_output=True
        )
        
        # Run inference
        inference_results = inference_system.process_video(video_path, output_path)
        
        # Combine results
        orchestrator_results = {
            'orchestrator_info': {
                'timestamp': datetime.now().isoformat(),
                'video_file': os.path.basename(video_path),
                'models_evaluated': len(self.available_models) if not manual_class_name else 1,
                'selected_model': best_model.class_name,
                'selection_score': best_model.avg_anomaly_score,
                'selection_mode': 'manual' if manual_class_name else 'automatic'
            },
            'model_selection': selection_report,
            'inference_results': inference_results
        }
        
        # Save orchestrator report
        if hasattr(inference_system, 'output_base_dir') and inference_system.output_base_dir:
            report_dir = os.path.join(inference_system.output_base_dir, "report")
            orchestrator_report_path = os.path.join(report_dir, "orchestrator_selection_report.json")
            
            with open(orchestrator_report_path, 'w') as f:
                json.dump(orchestrator_results, f, indent=2)
            
            logger.info(f"Orchestrator report saved: {orchestrator_report_path}")
        
        return orchestrator_results
    
    def run_inference_with_camera(self, camera_id: int, camera_fps: int = 30,
                                 duration_seconds: Optional[int] = None,
                                 skip_model_selection: bool = False,
                                 manual_class_name: str = None,
                                 output_path: str = None) -> Dict:
        """Run complete inference workflow with camera input"""
        logger.info(f"🚀 Starting GLASS Camera Inference Orchestrator")
        logger.info(f"📹 Camera: {camera_id}, FPS: {camera_fps}, Duration: {duration_seconds or 'continuous'}")

        # Step 1: Model selection (manual, skip, or automatic)
        if manual_class_name:
            # Manual model selection
            logger.info(f"🎯 Using manually specified model: {manual_class_name}")
            best_model = self.get_model_by_class_name(manual_class_name)
            if best_model is None:
                raise ValueError(f"Model '{manual_class_name}' not found")

            selection_report = {
                'timestamp': datetime.now().isoformat(),
                'input_source': f'camera_{camera_id}',
                'camera_fps': camera_fps,
                'selected_model': {
                    'class_name': best_model.class_name,
                    'model_dir': best_model.model_dir
                },
                'selection_criteria': 'manual_selection',
                'frames_sampled': 0
            }
        elif skip_model_selection:
            if not self.available_models:
                raise RuntimeError("No models available")
            best_model = self.available_models[0]
            logger.info(f"⚡ Skipping model selection, using: {best_model.class_name}")
            
            selection_report = {
                'timestamp': datetime.now().isoformat(),
                'input_source': f'camera_{camera_id}',
                'camera_fps': camera_fps,
                'selected_model': {
                    'class_name': best_model.class_name,
                    'model_dir': best_model.model_dir
                },
                'selection_criteria': 'first_available_model',
                'frames_sampled': 0
            }
        else:
            # Select best model from camera
            best_model, selection_report = self.select_best_model_from_camera(camera_id, camera_fps)
        
        # Step 2: Run full inference with selected model
        logger.info(f"🎯 Running camera inference with model: {best_model.class_name}")
        
        # Import video inference with tracking
        from video_inference_with_tracking import VideoInferenceWithTracking
        
        # Create camera input path
        camera_input = f"camera:{camera_id}:{camera_fps}"
        if duration_seconds:
            camera_input += f":{duration_seconds}"
        
        # Create inference system with selected model
        inference_system = VideoInferenceWithTracking(
            model_path=best_model.model_path,
            class_name=best_model.class_name,
            device=str(self.device),
            image_size=self.image_size,
            save_defect_frames=True,
            use_organized_output=True
        )
        
        # Run camera inference
        inference_results = inference_system.process_camera(
            camera_id=camera_id,
            camera_fps=camera_fps,
            duration_seconds=duration_seconds,
            output_path=output_path
        )
        
        # Combine results
        selection_mode = 'manual' if manual_class_name else ('skip' if skip_model_selection else 'automatic')
        orchestrator_results = {
            'orchestrator_info': {
                'timestamp': datetime.now().isoformat(),
                'input_source': f'camera_{camera_id}',
                'camera_fps': camera_fps,
                'duration_seconds': duration_seconds,
                'models_evaluated': len(self.available_models) if not (skip_model_selection or manual_class_name) else 1,
                'selected_model': best_model.class_name,
                'selection_score': getattr(best_model, 'avg_anomaly_score', None),
                'selection_mode': selection_mode
            },
            'model_selection': selection_report,
            'inference_results': inference_results
        }
        
        # Save orchestrator report
        if hasattr(inference_system, 'output_base_dir') and inference_system.output_base_dir:
            report_dir = os.path.join(inference_system.output_base_dir, "report")
            orchestrator_report_path = os.path.join(report_dir, "orchestrator_camera_report.json")
            
            with open(orchestrator_report_path, 'w') as f:
                json.dump(orchestrator_results, f, indent=2)
            
            logger.info(f"Camera orchestrator report saved: {orchestrator_report_path}")
        
        return orchestrator_results
    
    def list_available_models(self) -> None:
        """List all available models"""
        print(f"\n📋 Available Models ({len(self.available_models)}):")
        print("-" * 50)
        
        for i, model in enumerate(self.available_models, 1):
            # Check for best checkpoint
            ckpt_best = glob.glob(os.path.join(model.model_dir, "ckpt_best*.pth"))
            ckpt_regular = glob.glob(os.path.join(model.model_dir, "ckpt.pth"))
            
            checkpoint_info = ""
            if ckpt_best:
                epoch = ckpt_best[0].split("_best_")[1].split(".pth")[0]
                checkpoint_info = f"(best epoch: {epoch})"
            elif ckpt_regular:
                checkpoint_info = "(regular checkpoint)"
            
            print(f"{i:2d}. {model.class_name:<25} {checkpoint_info}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='GLASS Inference Orchestrator - Automatic Model Selection')
    parser.add_argument('--video_path', type=str, help='Path to input video (required unless --camera or --list_models)')
    parser.add_argument('--camera', action='store_true', help='Use camera input instead of video file')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--camera_fps', type=int, default=30, help='Camera FPS for recording (default: 30)')
    parser.add_argument('--duration_seconds', type=int, help='Duration to record from camera (default: continuous)')
    parser.add_argument('--output_path', type=str, help='Output video path (optional with organized output)')
    parser.add_argument('--models_path', type=str, default='results/models/backbone_0',
                       help='Path to models directory')
    parser.add_argument('--class_name', type=str,
                       help='Manually specify model class name (bypasses automatic selection)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--image_size', type=int, default=384, help='Model input image size')
    parser.add_argument('--sample_frames', type=int, default=30,
                       help='Number of frames to sample for model evaluation')
    parser.add_argument('--skip_model_selection', action='store_true',
                       help='Skip automatic model selection and use first available model')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = GLASSInferenceOrchestrator(
        models_base_path=args.models_path,
        device=args.device,
        image_size=args.image_size,
        sample_frames=args.sample_frames
    )
    
    # List models if requested
    if args.list_models:
        orchestrator.list_available_models()
        return
    
    # Validate input source
    if args.camera and args.video_path:
        logger.error("Cannot specify both --camera and --video_path. Choose one input source.")
        return
    
    if not args.camera and not args.video_path:
        logger.error("Must specify either --camera or --video_path as input source.")
        return
    
    # Validate video file if specified
    if args.video_path and not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Test camera connection if specified
    if args.camera:
        logger.info(f"🔍 Testing camera connection (ID: {args.camera_id})...")
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            logger.error(f"❌ Could not open camera {args.camera_id}. Please check camera connection.")
            return
        else:
            # Test frame capture
            ret, frame = cap.read()
            if not ret:
                logger.error(f"❌ Camera {args.camera_id} connected but cannot capture frames.")
                cap.release()
                return
            
            logger.info(f"✅ Camera {args.camera_id} connected successfully ({frame.shape[1]}x{frame.shape[0]})")
            cap.release()
    
    try:
        if args.camera:
            # Run camera inference
            logger.info("🎥 Starting camera-based inference...")
            results = orchestrator.run_inference_with_camera(
                camera_id=args.camera_id,
                camera_fps=args.camera_fps,
                duration_seconds=args.duration_seconds,
                skip_model_selection=args.skip_model_selection,
                manual_class_name=args.class_name,
                output_path=args.output_path
            )
            
            # Print camera summary
            print("\n" + "="*60)
            print("🎯 GLASS CAMERA INFERENCE RESULTS")
            print("="*60)
            print(f"📹 Camera: {args.camera_id} @ {args.camera_fps}fps")
            if args.duration_seconds:
                print(f"⏱️  Duration: {args.duration_seconds}s")
            else:
                print(f"⏱️  Duration: Continuous (until stopped)")
            print(f"🏆 Selected Model: {results['orchestrator_info']['selected_model']}")
            print(f"🎲 Selection Mode: {results['orchestrator_info']['selection_mode']}")
            if results['orchestrator_info']['selection_score']:
                print(f"📊 Selection Score: {results['orchestrator_info']['selection_score']:.4f}")
            print(f"🔍 Models Evaluated: {results['orchestrator_info']['models_evaluated']}")
            if 'unique_defects' in results['inference_results']:
                print(f"🎭 Unique Defects Found: {results['inference_results']['unique_defects']}")
            if 'fps_processing' in results['inference_results']:
                print(f"⚡ Processing FPS: {results['inference_results']['fps_processing']:.1f}")
            print("="*60)
            
        else:
            # Run video inference
            results = orchestrator.run_inference_with_best_model(
                video_path=args.video_path,
                output_path=args.output_path,
                manual_class_name=args.class_name
            )
            
            # Print video summary
            print("\n" + "="*60)
            print("🎯 GLASS VIDEO INFERENCE RESULTS")
            print("="*60)
            print(f"📹 Video: {os.path.basename(args.video_path)}")
            print(f"🏆 Selected Model: {results['orchestrator_info']['selected_model']}")
            print(f"🎲 Selection Mode: {results['orchestrator_info']['selection_mode']}")
            if results['orchestrator_info']['selection_score']:
                print(f"📊 Selection Score: {results['orchestrator_info']['selection_score']:.4f}")
            print(f"🔍 Models Evaluated: {results['orchestrator_info']['models_evaluated']}")
            print(f"🎭 Unique Defects Found: {results['inference_results']['unique_defects']}")
            print(f"⚡ Processing FPS: {results['inference_results']['fps_processing']:.1f}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        raise


if __name__ == '__main__':
    main()