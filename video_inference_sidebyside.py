#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import backbones
import glass
import common
from torchvision import transforms
import glob
from tqdm import tqdm
import time

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoInferenceSideBySide:
    def __init__(self, model_path, class_name, device='cuda:0', image_size=384):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        self.image_size = image_size
        
        # Load the GLASS model
        self.glass_model = self._load_model()
        
        # Image preprocessing - using the same size as training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
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
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for inference"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_frame(self, frame_tensor):
        """Run inference on a single frame"""
        if self.glass_model is None:
            return 0.0, np.zeros((self.image_size, self.image_size))
        
        with torch.no_grad():
            scores, masks = self.glass_model._predict(frame_tensor)
            
        return scores[0], masks[0]
    
    def create_annotated_frame(self, original_frame, mask, score, threshold=0.8, inference_fps=0, overall_fps=0):
        """Create annotated frame with anomaly overlay"""
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
        
        # Add FPS information
        inference_fps_text = f'Inference FPS: {inference_fps:.1f}'
        cv2.putText(overlay, inference_fps_text, (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        overall_fps_text = f'Processing FPS: {overall_fps:.1f}'
        cv2.putText(overlay, overall_fps_text, (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return overlay
    
    def process_video_sidebyside(self, video_path, output_path, threshold=0.8):
        """Process video and create side-by-side output"""
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
        
        # Create output video writer (side-by-side width)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
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
        
        with tqdm(total=total_frames, desc="Processing frames", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while True:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor = self.preprocess_frame(frame)
                
                # Get prediction
                inference_start = time.time()
                score, mask = self.predict_frame(frame_tensor)
                inference_time = time.time() - inference_start
                scores.append(score)
                
                # Calculate current FPS values
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                inference_fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Create annotated frame with FPS info
                annotated_frame = self.create_annotated_frame(frame, mask, score, threshold, inference_fps, current_fps)
                
                # Add labels to frames
                original_labeled = frame.copy()
                cv2.putText(original_labeled, 'Original', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(annotated_frame, 'GLASS Detection', (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Create side-by-side frame
                sidebyside_frame = np.hstack([original_labeled, annotated_frame])
                
                # Write frame
                out.write(sidebyside_frame)
                
                if score > threshold:
                    anomaly_count += 1
                
                frame_count += 1
                frame_time = time.time() - frame_start_time
                
                # Update progress bar with FPS info
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    pbar.set_postfix({
                        'FPS': f'{current_fps:.1f}',
                        'Inf_FPS': f'{inference_fps:.1f}',
                        'Score': f'{score:.3f}',
                        'Anomalies': anomaly_count
                    })
                    last_fps_update = current_time
                
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        
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
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Anomaly threshold (default: 0.8)')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Image size used during training (default: 384)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    
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
    inference = VideoInferenceSideBySide(
        model_path=args.model_path,
        class_name=args.class_name,
        device=args.device,
        image_size=args.image_size
    )
    
    # Process video
    results = inference.process_video_sidebyside(
        video_path=args.video_path,
        output_path=args.output_path,
        threshold=args.threshold
    )
    
    print(f"\nDone! Check the output video: {args.output_path}")

if __name__ == '__main__':
    main()