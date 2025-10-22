#!/usr/bin/env python3
"""
Clean GLASS Video Inference - No Info Text Overlays
Processes video with GLASS anomaly detection without any text overlays or annotations.
Outputs clean side-by-side comparison: original input vs anomaly detection visualization.
"""

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CleanVideoInference:
    def __init__(self, model_path, class_name, device='cuda:0', image_size=384):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        self.image_size = image_size
        
        # Load the GLASS model
        self.glass_model = self._load_model()
        
        # Image preprocessing - match training preprocessing exactly
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
        # Store original frame
        original_frame = frame.copy()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor, original_frame
    
    def predict_frame(self, frame_tensor):
        """Run inference on a single frame"""
        if self.glass_model is None:
            return 0.0, np.zeros((self.image_size, self.image_size))
        
        with torch.no_grad():
            scores, masks = self.glass_model._predict(frame_tensor)
            
        score = scores[0]
        mask = masks[0]
        
        return score, mask
    
    def create_clean_visualization(self, original_frame, mask, intensity=0.6):
        """Create clean anomaly visualization without any text overlays"""
        h, w = original_frame.shape[:2]
        
        # Resize mask to match frame size
        mask_resized = cv2.resize(mask, (w, h))
        
        # Normalize mask to 0-255 range
        mask_normalized = (mask_resized * 255).astype(np.uint8)
        
        # Apply colormap to mask (red for anomalies)
        mask_colored = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
        
        # Create overlay with specified intensity
        overlay = cv2.addWeighted(original_frame, 1.0 - intensity, mask_colored, intensity, 0)
        
        return overlay
    
    def process_video_clean(self, video_path, output_path, intensity=0.6, show_live=True):
        """Process video and create clean side-by-side output"""
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
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None  # Initialize later when we know dimensions
        
        # Setup live display window if requested
        if show_live:
            window_name = "GLASS Clean Inference"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print(f"Live display: Press 'q' to quit, 'p' to pause/resume")
        
        scores = []
        frame_count = 0
        start_time = time.time()
        paused = False
        
        with tqdm(total=total_frames, desc="Processing frames", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor, original_frame = self.preprocess_frame(frame)
                
                # Get prediction
                score, mask = self.predict_frame(frame_tensor)
                scores.append(score)
                
                # Create clean visualization (no text overlays)
                anomaly_frame = self.create_clean_visualization(original_frame, mask, intensity)
                
                # Create clean side-by-side frame
                sidebyside_frame = np.hstack([original_frame, anomaly_frame])
                
                # Initialize video writer on first frame
                if out is None:
                    h, w = sidebyside_frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not out.isOpened():
                        raise ValueError(f"Could not create output video: {output_path}")
                    print(f"Output video dimensions: {w}x{h}")
                    
                    # Setup display window size
                    if show_live:
                        display_width = min(1200, w)
                        display_height = int(display_width * h / w)
                        cv2.resizeWindow(window_name, display_width, display_height)
                
                # Write frame to output video
                out.write(sidebyside_frame)
                
                # Show live display if enabled
                if show_live:
                    cv2.imshow(window_name, sidebyside_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\\nStopping inference (user requested quit)...")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        if paused:
                            print("\\nPaused - Press 'p' to resume or 'q' to quit")
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
                
                frame_count += 1
                
                # Update progress bar
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({'FPS': f'{current_fps:.1f}', 'Score': f'{score:.3f}'})
                
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Calculate statistics
        total_processing_time = time.time() - start_time
        average_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        # Print summary
        print(f"\\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {total_processing_time:.2f} seconds")
        print(f"Average processing FPS: {average_fps:.2f}")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return {
            'total_frames': frame_count,
            'scores': scores,
            'fps': fps,
            'processing_fps': average_fps,
            'processing_time': total_processing_time
        }

def main():
    parser = argparse.ArgumentParser(description='GLASS Clean Video Inference - No Text Overlays')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model directory (e.g., results/models/backbone_0/)')
    parser.add_argument('--class_name', type=str, required=True,
                       help='Class name for model (e.g., wfdd_yellow_cloth)')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path for output video file')
    parser.add_argument('--intensity', type=float, default=0.6,
                       help='Overlay intensity (0.0-1.0, default: 0.6)')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Image size used during training (default: 384)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable live display window')
    
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
    
    inference = CleanVideoInference(
        model_path=args.model_path,
        class_name=args.class_name,
        device=args.device,
        image_size=args.image_size
    )
    
    # Process video
    show_live = not args.no_display
    results = inference.process_video_clean(
        video_path=args.video_path,
        output_path=args.output_path,
        intensity=args.intensity,
        show_live=show_live
    )
    
    print(f"\\nDone! Check the output video: {args.output_path}")

if __name__ == '__main__':
    main()