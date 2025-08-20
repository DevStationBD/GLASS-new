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
import json
from size_analyzer import DefectSizeAnalyzer, DefectMetrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoInferenceWithSize:
    def __init__(self, model_path, class_name, device='cuda:0', image_size=384, pixel_size=None, physical_unit="mm"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_name = class_name
        self.model_path = model_path
        self.image_size = image_size
        
        # Initialize defect size analyzer
        self.size_analyzer = DefectSizeAnalyzer(
            pixel_size=pixel_size,
            physical_unit=physical_unit,
            min_defect_area=10
        )
        
        # Load the GLASS model
        self.glass_model = self._load_model()
        
        # Image preprocessing - using the same size as training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Storage for batch analysis
        self.frame_metrics = []
        self.size_history = []
    
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
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_frame_with_size(self, frame_tensor, original_frame, threshold=0.8):
        """Run inference and size analysis on a single frame"""
        if self.glass_model is None:
            return 0.0, np.zeros((self.image_size, self.image_size)), None
        
        with torch.no_grad():
            scores, masks = self.glass_model._predict(frame_tensor)
            
        score = scores[0]
        mask = masks[0]
        
        # Analyze defect sizes
        metrics = self.size_analyzer.analyze_defects(mask, threshold=threshold)
        self.frame_metrics.append(metrics)
        
        # Track size history for trending
        self.size_history.append({
            'frame': len(self.size_history),
            'score': score,
            'num_defects': metrics.num_defects,
            'total_area': metrics.total_defect_pixels,
            'defect_percentage': metrics.defect_percentage,
            'largest_defect': metrics.largest_defect_area
        })
        
        return score, mask, metrics
    
    def create_enhanced_annotated_frame(self, original_frame, mask, score, metrics, 
                                      threshold=0.8, inference_fps=0, overall_fps=0):
        """Create annotated frame with anomaly overlay and defect size information"""
        h, w = original_frame.shape[:2]
        
        # Resize mask to match frame size
        mask_resized = cv2.resize(mask, (w, h))
        
        # Create defect visualization with size measurements
        vis_frame = self.size_analyzer.create_defect_visualization(
            original_frame, mask_resized, metrics, threshold)
        
        # Add score and status information (non-overlapping with size info)
        score_text = f'Anomaly Score: {score:.4f}'
        cv2.putText(vis_frame, score_text, (w - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add threshold line
        threshold_text = f'Threshold: {threshold:.4f}'
        cv2.putText(vis_frame, threshold_text, (w - 300, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add status
        status = "ANOMALY" if score > threshold else "NORMAL"
        status_color = (0, 0, 255) if score > threshold else (0, 255, 0)
        cv2.putText(vis_frame, status, (w - 300, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add FPS information
        inference_fps_text = f'Inf FPS: {inference_fps:.1f}'
        cv2.putText(vis_frame, inference_fps_text, (w - 300, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        overall_fps_text = f'Proc FPS: {overall_fps:.1f}'
        cv2.putText(vis_frame, overall_fps_text, (w - 300, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add defect trend information if we have history
        if len(self.size_history) > 1:
            recent_frames = self.size_history[-10:]  # Last 10 frames
            avg_defects = np.mean([f['num_defects'] for f in recent_frames])
            trend_text = f'Avg Defects (10f): {avg_defects:.1f}'
            cv2.putText(vis_frame, trend_text, (w - 300, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_video_with_size_analysis(self, video_path, output_path, 
                                       threshold=0.9, show_live=True,
                                       export_results=True, results_dir=None):
        """Process video with defect size analysis and enhanced visualization"""
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
        print(f"Size analyzer: pixel_size={self.size_analyzer.pixel_size}, unit={self.size_analyzer.physical_unit}")
        
        # Create output video writer (side-by-side width)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Setup results directory
        if export_results and results_dir:
            os.makedirs(results_dir, exist_ok=True)
        
        # Setup live display window if requested
        if show_live:
            window_name = "GLASS Real-time Inference with Size Analysis"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, frame_width * 2)
            display_height = int(display_width * frame_height / (frame_width * 2))
            cv2.resizeWindow(window_name, display_width, display_height)
            print(f"Live display: Press 'q' to quit, 'p' to pause/resume, 's' to save current frame analysis")
        
        # Use default threshold if none provided
        if threshold is None:
            threshold = 0.8
        
        print(f"Using threshold: {threshold:.4f}")
        
        # Reset tracking variables
        self.frame_metrics = []
        self.size_history = []
        scores = []
        
        # Processing loop
        frame_count = 0
        anomaly_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps_update_interval = 1.0
        paused = False
        
        with tqdm(total=total_frames, desc="Processing frames with size analysis", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while True:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor = self.preprocess_frame(frame)
                
                # Get prediction and size analysis
                inference_start = time.time()
                score, mask, metrics = self.predict_frame_with_size(frame_tensor, frame, threshold)
                inference_time = time.time() - inference_start
                scores.append(score)
                
                # Calculate current FPS values
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                inference_fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Create enhanced annotated frame with size info
                annotated_frame = self.create_enhanced_annotated_frame(
                    frame, mask, score, metrics, threshold, inference_fps, current_fps)
                
                # Add labels to frames
                original_labeled = frame.copy()
                cv2.putText(original_labeled, 'Original', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(annotated_frame, 'GLASS Detection + Size Analysis', (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Create side-by-side frame
                sidebyside_frame = np.hstack([original_labeled, annotated_frame])
                
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
                            print("\\nPaused - Press 'p' to resume, 'q' to quit, 's' to save analysis")
                            while paused:
                                key = cv2.waitKey(30) & 0xFF
                                if key == ord('p'):
                                    paused = False
                                    print("Resumed processing...")
                                elif key == ord('s') and export_results and results_dir:
                                    # Save current frame analysis
                                    frame_results_path = os.path.join(results_dir, f'frame_{frame_count}_analysis.json')
                                    self.size_analyzer.export_measurements(metrics, frame_results_path)
                                    print(f"Saved current frame analysis to: {frame_results_path}")
                                elif key == ord('q'):
                                    print("Stopping inference...")
                                    break
                        if key == ord('q'):
                            break
                    elif key == ord('s') and export_results and results_dir:
                        # Save current frame analysis
                        frame_results_path = os.path.join(results_dir, f'frame_{frame_count}_analysis.json')
                        self.size_analyzer.export_measurements(metrics, frame_results_path)
                        print(f"\\nSaved frame {frame_count} analysis to: {frame_results_path}")
                
                if score > threshold:
                    anomaly_count += 1
                
                frame_count += 1
                frame_time = time.time() - frame_start_time
                
                # Update progress bar with enhanced info
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    pbar.set_postfix({
                        'FPS': f'{current_fps:.1f}',
                        'Score': f'{score:.3f}',
                        'Defects': metrics.num_defects,
                        'Area%': f'{metrics.defect_percentage:.1f}',
                        'Anomalies': anomaly_count
                    })
                    last_fps_update = current_time
                
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Calculate final timing and size statistics
        total_processing_time = time.time() - start_time
        average_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        # Generate comprehensive results
        results = self._generate_comprehensive_results(
            scores, frame_count, total_processing_time, average_fps, fps, anomaly_count, threshold)
        
        # Export detailed results if requested
        if export_results and results_dir:
            self._export_detailed_results(results_dir, results)
        
        # Print enhanced summary
        self._print_enhanced_summary(results, output_path)
        
        return results
    
    def _generate_comprehensive_results(self, scores, frame_count, processing_time, 
                                      avg_fps, video_fps, anomaly_count, threshold):
        """Generate comprehensive analysis results"""
        # Basic video stats
        results = {
            'video_stats': {
                'total_frames': frame_count,
                'processing_time': processing_time,
                'processing_fps': avg_fps,
                'video_fps': video_fps,
                'speed_ratio': avg_fps / video_fps if video_fps > 0 else 0,
                'anomalous_frames': anomaly_count,
                'anomaly_percentage': (anomaly_count / frame_count) * 100 if frame_count > 0 else 0,
                'threshold': threshold
            },
            'score_stats': {
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'median_score': np.median(scores) if scores else 0
            }
        }
        
        # Size analysis statistics
        if self.frame_metrics:
            total_defects = sum(m.num_defects for m in self.frame_metrics)
            frames_with_defects = sum(1 for m in self.frame_metrics if m.num_defects > 0)
            
            defect_areas = []
            defect_percentages = []
            for m in self.frame_metrics:
                defect_areas.extend(m.defect_areas)
                defect_percentages.append(m.defect_percentage)
            
            results['size_stats'] = {
                'total_defects_detected': total_defects,
                'frames_with_defects': frames_with_defects,
                'avg_defects_per_frame': total_defects / frame_count if frame_count > 0 else 0,
                'avg_defects_per_anomalous_frame': total_defects / frames_with_defects if frames_with_defects > 0 else 0,
                'defect_size_stats': {
                    'min_area': min(defect_areas) if defect_areas else 0,
                    'max_area': max(defect_areas) if defect_areas else 0,
                    'mean_area': np.mean(defect_areas) if defect_areas else 0,
                    'median_area': np.median(defect_areas) if defect_areas else 0,
                    'std_area': np.std(defect_areas) if defect_areas else 0
                },
                'coverage_stats': {
                    'min_coverage': min(defect_percentages) if defect_percentages else 0,
                    'max_coverage': max(defect_percentages) if defect_percentages else 0,
                    'mean_coverage': np.mean(defect_percentages) if defect_percentages else 0
                }
            }
            
            # Physical measurements if available
            if self.size_analyzer.pixel_size:
                physical_areas = []
                for m in self.frame_metrics:
                    if m.defect_areas_physical:
                        physical_areas.extend(m.defect_areas_physical)
                
                results['physical_measurements'] = {
                    'unit': self.size_analyzer.physical_unit,
                    'pixel_size': self.size_analyzer.pixel_size,
                    'total_physical_area': sum(physical_areas) if physical_areas else 0,
                    'mean_defect_area': np.mean(physical_areas) if physical_areas else 0,
                    'largest_defect_area': max(physical_areas) if physical_areas else 0
                }
        
        # Trend analysis
        if len(self.size_history) > 10:
            recent_trend = self.size_history[-10:]
            early_trend = self.size_history[:10] if len(self.size_history) >= 20 else self.size_history[:len(self.size_history)//2]
            
            results['trend_analysis'] = {
                'defect_count_trend': np.mean([f['num_defects'] for f in recent_trend]) - np.mean([f['num_defects'] for f in early_trend]),
                'area_trend': np.mean([f['total_area'] for f in recent_trend]) - np.mean([f['total_area'] for f in early_trend]),
                'score_trend': np.mean([f['score'] for f in recent_trend]) - np.mean([f['score'] for f in early_trend])
            }
        
        return results
    
    def _export_detailed_results(self, results_dir, results):
        """Export detailed analysis results"""
        # Export summary JSON
        summary_path = os.path.join(results_dir, 'video_analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export frame-by-frame size history
        history_path = os.path.join(results_dir, 'frame_size_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.size_history, f, indent=2)
        
        # Export individual frame metrics (sample)
        if len(self.frame_metrics) > 0:
            # Save every 10th frame or frames with significant defects
            sample_metrics = []
            for i, metrics in enumerate(self.frame_metrics):
                if i % 10 == 0 or metrics.num_defects > 5:  # Sample or significant defects
                    metrics_path = os.path.join(results_dir, f'frame_{i:06d}_metrics.json')
                    self.size_analyzer.export_measurements(metrics, metrics_path, include_coordinates=True)
                    sample_metrics.append(i)
            
            # Save sample list
            sample_path = os.path.join(results_dir, 'detailed_frame_samples.json')
            with open(sample_path, 'w') as f:
                json.dump({'sampled_frames': sample_metrics}, f, indent=2)
        
        print(f"Detailed analysis results exported to: {results_dir}")
    
    def _print_enhanced_summary(self, results, output_path):
        """Print comprehensive summary with size analysis"""
        print(f"\\n" + "="*60)
        print(f"GLASS VIDEO ANALYSIS WITH SIZE MEASUREMENTS - COMPLETE")
        print(f"="*60)
        
        # Basic processing stats
        video_stats = results['video_stats']
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {video_stats['total_frames']}")
        print(f"Processing time: {video_stats['processing_time']:.2f} seconds")
        print(f"Average processing FPS: {video_stats['processing_fps']:.2f}")
        print(f"Original video FPS: {video_stats['video_fps']:.1f}")
        print(f"Speed ratio: {video_stats['speed_ratio']:.2f}x {'faster' if video_stats['speed_ratio'] > 1 else 'slower'} than real-time")
        
        # Anomaly detection stats
        print(f"\\nANOMALY DETECTION:")
        print(f"Anomalous frames detected: {video_stats['anomalous_frames']}")
        print(f"Anomaly percentage: {video_stats['anomaly_percentage']:.2f}%")
        
        score_stats = results['score_stats']
        print(f"Score range: {score_stats['min_score']:.4f} - {score_stats['max_score']:.4f}")
        print(f"Mean score: {score_stats['mean_score']:.4f} ± {score_stats['std_score']:.4f}")
        
        # Size analysis stats
        if 'size_stats' in results:
            size_stats = results['size_stats']
            print(f"\\nDEFECT SIZE ANALYSIS:")
            print(f"Total defects detected: {size_stats['total_defects_detected']}")
            print(f"Frames with defects: {size_stats['frames_with_defects']}")
            print(f"Average defects per frame: {size_stats['avg_defects_per_frame']:.2f}")
            print(f"Average defects per anomalous frame: {size_stats['avg_defects_per_anomalous_frame']:.2f}")
            
            defect_size = size_stats['defect_size_stats']
            print(f"\\nDEFECT SIZE STATISTICS (pixels):")
            print(f"Size range: {defect_size['min_area']} - {defect_size['max_area']} pixels")
            print(f"Mean defect size: {defect_size['mean_area']:.1f} ± {defect_size['std_area']:.1f} pixels")
            print(f"Median defect size: {defect_size['median_area']:.1f} pixels")
            
            coverage = size_stats['coverage_stats']
            print(f"\\nDEFECT COVERAGE STATISTICS:")
            print(f"Coverage range: {coverage['min_coverage']:.3f}% - {coverage['max_coverage']:.3f}%")
            print(f"Mean coverage per frame: {coverage['mean_coverage']:.3f}%")
            
            # Physical measurements
            if 'physical_measurements' in results:
                phys = results['physical_measurements']
                print(f"\\nPHYSICAL MEASUREMENTS ({phys['unit']}):")
                print(f"Pixel size: {phys['pixel_size']} {phys['unit'][:-1]}/pixel")
                print(f"Total defect area: {phys['total_physical_area']:.3f} {phys['unit']}")
                print(f"Mean defect size: {phys['mean_defect_area']:.3f} {phys['unit']}")
                print(f"Largest defect: {phys['largest_defect_area']:.3f} {phys['unit']}")
        
        # Trend analysis
        if 'trend_analysis' in results:
            trend = results['trend_analysis']
            print(f"\\nTREND ANALYSIS:")
            print(f"Defect count trend: {trend['defect_count_trend']:+.2f} (early vs recent frames)")
            print(f"Area trend: {trend['area_trend']:+.1f} pixels (early vs recent frames)")
            print(f"Score trend: {trend['score_trend']:+.4f} (early vs recent frames)")

def main():
    parser = argparse.ArgumentParser(description='GLASS Video Inference with Defect Size Analysis')
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
                       help='Disable live display window')
    
    # Size analysis parameters
    parser.add_argument('--pixel_size', type=float, default=None,
                       help='Physical size per pixel (e.g., 0.1 for 0.1mm per pixel)')
    parser.add_argument('--physical_unit', type=str, default='mm',
                       help='Physical unit for measurements (mm, cm, inch)')
    parser.add_argument('--export_results', action='store_true', default=True,
                       help='Export detailed analysis results (default: True)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory for detailed results (auto-generated if None)')
    
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
    
    # Setup results directory
    results_dir = args.results_dir
    if args.export_results and not results_dir:
        video_name = Path(args.video_path).stem
        results_dir = f"results/size_analysis/{args.class_name}/{video_name}"
    
    # Create inference instance
    print(f"Initializing GLASS model with size analysis for class: {args.class_name}")
    if args.pixel_size:
        print(f"Physical measurements enabled: {args.pixel_size} {args.physical_unit}/pixel")
    
    inference = VideoInferenceWithSize(
        model_path=args.model_path,
        class_name=args.class_name,
        device=args.device,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        physical_unit=args.physical_unit
    )
    
    # Process video
    show_live = not args.no_display
    results = inference.process_video_with_size_analysis(
        video_path=args.video_path,
        output_path=args.output_path,
        threshold=args.threshold,
        show_live=show_live,
        export_results=args.export_results,
        results_dir=results_dir
    )
    
    print(f"\\nAnalysis complete! Check the output video: {args.output_path}")
    if args.export_results and results_dir:
        print(f"Detailed results saved to: {results_dir}")

if __name__ == '__main__':
    main()