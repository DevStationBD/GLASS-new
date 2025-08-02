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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoInference:
    def __init__(self, model_path, class_name, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        
        # Load the GLASS model
        self.glass_model = self._load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 288)),
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
            input_shape=(3, 288, 288),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=640,
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
            return 0.0, np.zeros((288, 288))
        
        with torch.no_grad():
            scores, masks = self.glass_model._predict(frame_tensor)
            
        return scores[0], masks[0]
    
    def process_video(self, video_path, output_dir=None, save_frames=False):
        """Process entire video and return anomaly scores and masks"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        scores = []
        masks = []
        frame_count = 0
        
        # Create output directory if needed
        if save_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor = self.preprocess_frame(frame)
                
                # Get prediction
                score, mask = self.predict_frame(frame_tensor)
                scores.append(score)
                masks.append(mask)
                
                # Save frame with anomaly overlay if requested
                if save_frames and output_dir:
                    self._save_frame_with_overlay(frame, mask, score, 
                                                frame_count, output_dir)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        return {
            'scores': scores,
            'masks': masks,
            'frame_count': frame_count,
            'fps': fps,
            'video_path': video_path
        }
    
    def _save_frame_with_overlay(self, frame, mask, score, frame_idx, output_dir):
        """Save frame with anomaly mask overlay"""
        # Resize mask to match frame size
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        
        # Apply colormap to mask
        mask_colored = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), 
                                       cv2.COLORMAP_JET)
        
        # Overlay mask on frame
        overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
        
        # Add score text
        cv2.putText(overlay, f'Score: {score:.4f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(frame_path, overlay)

def create_summary_report(results, output_path):
    """Create a summary report of the inference results"""
    report = []
    report.append("GLASS Video Inference Summary Report")
    report.append("=" * 50)
    report.append("")
    
    for class_name, result in results.items():
        report.append(f"Class: {class_name}")
        report.append(f"Video: {os.path.basename(result['video_path'])}")
        report.append(f"Total Frames: {result['frame_count']}")
        report.append(f"FPS: {result['fps']:.2f}")
        
        scores = result['scores']
        if scores:
            report.append(f"Min Score: {min(scores):.4f}")
            report.append(f"Max Score: {max(scores):.4f}")
            report.append(f"Mean Score: {np.mean(scores):.4f}")
            report.append(f"Std Score: {np.std(scores):.4f}")
            
            # Find anomalous frames (above threshold)
            threshold = np.mean(scores) + 2 * np.std(scores)
            anomalous_frames = [i for i, score in enumerate(scores) if score > threshold]
            report.append(f"Anomalous Frames (threshold={threshold:.4f}): {len(anomalous_frames)}")
            if anomalous_frames:
                report.append(f"Anomalous Frame Indices: {anomalous_frames[:10]}...")  # Show first 10
        
        report.append("-" * 30)
        report.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='GLASS Video Inference')
    parser.add_argument('--video_dir', type=str, default='wfdd_videos/',
                       help='Directory containing videos to process')
    parser.add_argument('--model_dir', type=str, default='results/models/backbone_0/',
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--output_dir', type=str, default='video_inference_results/',
                       help='Output directory for results')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save frames with anomaly overlays')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map video files to model classes
    video_class_mapping = {
        'wfdd_good.mp4': 'wfdd_grey_cloth',
        'wfdd_contaminated.mp4': 'wfdd_grey_cloth',
        'wfdd_defect.mp4': 'wfdd_grid_cloth',
        'wfdd_flecked.mp4': 'wfdd_pink_flower',
        'wfdd_line.mp4': 'wfdd_yellow_cloth',
        'wfdd_string.mp4': 'wfdd_yellow_cloth',
    }
    
    results = {}
    
    # Process each video
    for video_file, class_name in video_class_mapping.items():
        video_path = os.path.join(args.video_dir, video_file)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        print(f"\nProcessing {video_file} with model {class_name}")
        
        # Create inference instance
        inference = VideoInference(args.model_dir, class_name, args.device)
        
        # Create output subdirectory for this video
        video_output_dir = os.path.join(args.output_dir, 
                                      os.path.splitext(video_file)[0])
        
        # Process video
        result = inference.process_video(video_path, video_output_dir, 
                                       args.save_frames)
        results[class_name] = result
        
        # Save individual results
        result_file = os.path.join(video_output_dir, 'results.npz')
        os.makedirs(video_output_dir, exist_ok=True)
        np.savez(result_file, 
                scores=result['scores'], 
                masks=result['masks'],
                frame_count=result['frame_count'],
                fps=result['fps'])
        
        print(f"Results saved to: {result_file}")
    
    # Create summary report
    summary_path = os.path.join(args.output_dir, 'summary_report.txt')
    create_summary_report(results, summary_path)

if __name__ == '__main__':
    main()