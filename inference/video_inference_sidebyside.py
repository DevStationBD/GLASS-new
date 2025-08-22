#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backbones
import glass
import common
from torchvision import transforms
import glob
from tqdm import tqdm
import time
from size_analyzer import DefectSizeAnalyzer

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoInferenceSideBySide:
    def __init__(self, model_path, class_name, device='cuda:0', image_size=384, pixel_size=None, physical_unit="mm"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        self.image_size = image_size
        
        # Initialize defect size analyzer
        self.size_analyzer = DefectSizeAnalyzer(
            pixel_size=pixel_size,
            physical_unit=physical_unit,
            min_defect_area=5  # Lower threshold for video processing
        )
        
        # Load the GLASS model
        self.glass_model = self._load_model()
        
        # Image preprocessing - match training preprocessing with square resize
        # Training uses: Resize((imgsize, imgsize)) → ToTensor() → Normalize() (square)
        # Inference matches exactly: force square dimensions
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),  # Force square to match training
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _load_model(self):
        # Create GLASS model with default parameters
        glass_model = glass.GLASS(self.device)
        
        # Load backbone (WideResNet50)
        backbone = backbones.load('wideresnet50')
        
        # Initialize GLASS with parameters matching training configuration
        glass_model.load(
            backbone=backbone,
            layers_to_extract_from=['layer2', 'layer3'],
            device=self.device,
            input_shape=(3, self.image_size, self.image_size),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=10,  # Updated to match your training
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
        
        # Load the trained checkpoint
        ckpt_path = glob.glob(os.path.join(self.model_path, f'{self.class_name}/ckpt_best*'))
        if len(ckpt_path) == 0:
            ckpt_path = glob.glob(os.path.join(self.model_path, f'{self.class_name}/ckpt.pth'))
        
        if len(ckpt_path) > 0:
            print(f"Loading checkpoint: {ckpt_path[0]}")
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            
            if 'discriminator' in state_dict:
                glass_model.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    glass_model.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                glass_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: No checkpoint found for {self.class_name}")
            return None
        
        glass_model.eval()
        return glass_model
    
    def preprocess_frame(self, frame, return_transformed_image=False):
        """Preprocess a single frame for inference"""
        # Store original frame for display
        original_frame = frame.copy()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (this will resize to square for model inference)
        tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        if return_transformed_image:
            # Convert tensor back to displayable image for preview
            # Denormalize the tensor
            denorm_tensor = tensor[0].clone()
            for t, m, s in zip(denorm_tensor, IMAGENET_MEAN, IMAGENET_STD):
                t.mul_(s).add_(m)
            
            # Convert to numpy and BGR for OpenCV
            transformed_image = denorm_tensor.permute(1, 2, 0).numpy()
            transformed_image = np.clip(transformed_image * 255, 0, 255).astype(np.uint8)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            
            return tensor, original_frame, transformed_image
        
        return tensor, original_frame
    
    def predict_frame(self, frame_tensor, threshold=0.5):
        """Run inference and defect area analysis on a single frame"""
        if self.glass_model is None:
            return 0.0, np.zeros((self.image_size, self.image_size)), None
        
        with torch.no_grad():
            scores, masks = self.glass_model._predict(frame_tensor)
            
        score = scores[0]
        mask = masks[0]
        
        # Calculate defect area metrics
        metrics = self.size_analyzer.analyze_defects(mask, threshold=threshold, use_morphology=False)
        
        return score, mask, metrics
    
    def create_annotated_frame(self, original_frame, mask, score, metrics=None, threshold=0.8, inference_fps=0, overall_fps=0):
        """Create annotated frame with anomaly overlay and defect area information"""
        h, w = original_frame.shape[:2]
        
        # Resize mask to match frame size
        mask_resized = cv2.resize(mask, (w, h))
        
        # Normalize mask to 0-255 range
        mask_normalized = (mask_resized * 255).astype(np.uint8)
        
        # Apply colormap to mask (red for anomalies)
        mask_colored = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(original_frame, 0.7, mask_colored, 0.3, 0)
        
        # Add score text
        score_text = f'Anomaly Score: {score:.4f}'
        cv2.putText(overlay, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add threshold line
        threshold_text = f'Threshold: {threshold:.4f}'
        cv2.putText(overlay, threshold_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add status
        status = "ANOMALY" if score > threshold else "NORMAL"
        status_color = (0, 0, 255) if score > threshold else (0, 255, 0)
        cv2.putText(overlay, status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add defect area information
        if metrics is not None:
            # Defect count
            defect_count_text = f'Defects: {metrics.num_defects}'
            cv2.putText(overlay, defect_count_text, (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Defect area percentage
            area_percent_text = f'Area: {metrics.defect_percentage:.2f}%'
            cv2.putText(overlay, area_percent_text, (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Physical area if available
            if metrics.physical_unit and metrics.total_defect_area_physical is not None:
                physical_area_text = f'Physical: {metrics.total_defect_area_physical:.2f}{metrics.physical_unit}²'
                cv2.putText(overlay, physical_area_text, (10, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y_offset = 170
            else:
                y_offset = 150
        else:
            y_offset = 110
        
        # Add FPS information
        inference_fps_text = f'Inference FPS: {inference_fps:.1f}'
        cv2.putText(overlay, inference_fps_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        overall_fps_text = f'Processing FPS: {overall_fps:.1f}'
        cv2.putText(overlay, overall_fps_text, (10, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return overlay
    
    def process_video_sidebyside(self, video_path, output_path, threshold=0.9, show_live=True):
        """Process video and create side-by-side output with optional live display"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Frame size: {frame_width}x{frame_height}")
        
        # Create output video writer - we'll determine dimensions from first transformed frame
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None  # Initialize later when we know transformed frame dimensions
        
        # Setup live display window if requested
        if show_live:
            window_name = "GLASS Real-time Inference"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print(f"Live display: Press 'q' to quit, 'p' to pause/resume")
            print("Display window will be sized after first frame processing")
        
        # Use default threshold of 0.8 if none provided
        if threshold is None:
            threshold = 0.8
        
        print(f"Using threshold: {threshold:.4f}")
        scores = []
        
        # Second pass for processing
        frame_count = 0
        anomaly_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps_update_interval = 1.0  # Update FPS every second
        paused = False  # For pause/resume functionality
        
        with tqdm(total=total_frames, desc="Processing frames", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while True:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame and get both original and transformed images for display
                frame_tensor, original_frame, transformed_frame = self.preprocess_frame(frame, return_transformed_image=True)
                
                # Get prediction and defect area analysis
                inference_start = time.time()
                score, mask, metrics = self.predict_frame(frame_tensor, threshold)
                inference_time = time.time() - inference_start
                scores.append(score)
                
                # Calculate current FPS values
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                inference_fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Create annotated frame using original frame (preserving aspect ratio)
                annotated_frame = self.create_annotated_frame(original_frame, mask, score, metrics, threshold, inference_fps, current_fps)
                
                # Add labels to frames - show original vs detected
                original_labeled = original_frame.copy()
                cv2.putText(original_labeled, 'Original Input', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(annotated_frame, 'GLASS Detection', (10, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Create side-by-side frame using original input and annotated result
                sidebyside_frame = np.hstack([original_labeled, annotated_frame])
                
                # Initialize video writer on first frame (now we know dimensions)
                if out is None:
                    h, w = sidebyside_frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not out.isOpened():
                        raise ValueError(f"Could not create output video: {output_path}")
                    print(f"Output video dimensions: {w}x{h}")
                    
                    # Setup display window size now that we know frame dimensions
                    if show_live:
                        display_width = min(1200, w)
                        display_height = int(display_width * h / w)
                        cv2.resizeWindow(window_name, display_width, display_height)
                        print(f"Display window: {display_width}x{display_height}")
                
                # Write frame to output video
                out.write(sidebyside_frame)
                
                # Show live display if enabled
                if show_live:
                    cv2.imshow(window_name, sidebyside_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopping inference (user requested quit)...")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        if paused:
                            print("\nPaused - Press 'p' to resume or 'q' to quit")
                            while paused:
                                key = cv2.waitKey(30) & 0xFF
                                if key == ord('p'):
                                    paused = False
                                    print("Resumed processing...")
                                elif key == ord('q'):
                                    print("Stopping inference...")
                                    break
                        if key == ord('q'):
                            break
                
                if score > threshold:
                    anomaly_count += 1
                
                frame_count += 1
                frame_time = time.time() - frame_start_time
                
                # Update progress bar with FPS and defect area info
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    area_info = f'{metrics.defect_percentage:.1f}%' if metrics else '0%'
                    defects_info = f'{metrics.num_defects}' if metrics else '0'
                    pbar.set_postfix({
                        'FPS': f'{current_fps:.1f}',
                        'Score': f'{score:.3f}',
                        'Area': area_info,
                        'Defects': defects_info,
                        'Anomalies': anomaly_count
                    })
                    last_fps_update = current_time
                
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Calculate final timing statistics
        total_processing_time = time.time() - start_time
        average_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {total_processing_time:.2f} seconds")
        print(f"Average processing FPS: {average_fps:.2f}")
        print(f"Original video FPS: {fps:.1f}")
        print(f"Speed ratio: {average_fps/fps:.2f}x {'faster' if average_fps > fps else 'slower'} than real-time")
        print(f"Anomalous frames detected: {anomaly_count}")
        print(f"Anomaly percentage: {(anomaly_count/frame_count)*100:.2f}%")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return {
            'total_frames': frame_count,
            'anomaly_frames': anomaly_count,
            'scores': scores,
            'threshold': threshold,
            'fps': fps,
            'processing_fps': average_fps,
            'processing_time': total_processing_time
        }

def main():
    parser = argparse.ArgumentParser(description='GLASS Video Inference - Side by Side')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model directory (e.g., results/models/backbone_0/)')
    parser.add_argument('--class_name', type=str, required=True,
                       help='Class name for model (e.g., wfdd_yellow_cloth)')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path for output video file')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='Anomaly threshold (default: 0.9)')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Image size used during training (default: 384)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable live display window (default: show live display)')
    
    # Defect size analysis parameters
    parser.add_argument('--pixel_size', type=float, default=None,
                       help='Physical size per pixel (e.g., 0.1 for 0.1mm per pixel)')
    parser.add_argument('--physical_unit', type=str, default='mm',
                       help='Physical unit for measurements (mm, cm, inch)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create inference instance
    print(f"Initializing GLASS model for class: {args.class_name}")
    if args.pixel_size:
        print(f"Physical measurements enabled: {args.pixel_size} {args.physical_unit}/pixel")
    
    inference = VideoInferenceSideBySide(
        model_path=args.model_path,
        class_name=args.class_name,
        device=args.device,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        physical_unit=args.physical_unit
    )
    
    # Process video
    show_live = not args.no_display
    results = inference.process_video_sidebyside(
        video_path=args.video_path,
        output_path=args.output_path,
        threshold=args.threshold,
        show_live=show_live
    )
    
    print(f"\nDone! Check the output video: {args.output_path}")

if __name__ == '__main__':
    main()