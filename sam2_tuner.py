# import numpy as np
# import cv2
# import torch
# import time
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# @dataclass
# class SAMTuningMetrics:
#     avg_mask_area: float
#     mask_count: int
#     overlap_ratio: float
#     processing_time: float
#     memory_usage: float
#     coverage_ratio: float
#     edge_complexity: float
#     completeness_score: float
#     boundary_confidence: float

# class SAM2Tuner:
#     def __init__(
#         self, 
#         model_name: str,
#         target_mask_count: Optional[int] = None,
#         min_object_size: Optional[int] = 100,
#         max_memory_gb: Optional[float] = None,
#         max_processing_time: Optional[float] = None,
#         target_coverage: Optional[float] = 0.85,
#         batch_size: int = 64,
#         min_completeness: float = 0.85
#     ):
#         self.model_name = model_name
#         self.target_mask_count = target_mask_count
#         self.min_object_size = min_object_size
#         self.max_memory_gb = max_memory_gb
#         self.max_processing_time = max_processing_time
#         self.target_coverage = target_coverage
#         self.batch_size = batch_size
#         self.min_completeness = min_completeness

#     def _calculate_overlap_ratio(self, masks: List[Dict]) -> float:
#         """Calculate average IoU between overlapping masks."""
#         if len(masks) < 2:
#             return 0.0
            
#         total_iou = 0
#         count = 0
        
#         for i in range(len(masks)):
#             for j in range(i + 1, len(masks)):
#                 mask1 = masks[i]['segmentation']
#                 mask2 = masks[j]['segmentation']
                
#                 intersection = np.logical_and(mask1, mask2).sum()
#                 union = np.logical_or(mask1, mask2).sum()
                
#                 if union > 0:
#                     iou = intersection / union
#                     total_iou += iou
#                     count += 1
        
#         return total_iou / count if count > 0 else 0.0

#     def _calculate_edge_complexity(self, masks: List[Dict]) -> float:
#         """Calculate average boundary complexity of masks."""
#         if not masks:
#             return 0.0
            
#         complexities = []
#         for mask in masks:
#             contours, _ = cv2.findContours(
#                 mask['segmentation'].astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )
            
#             if contours:
#                 for contour in contours:
#                     perimeter = cv2.arcLength(contour, True)
#                     area = cv2.contourArea(contour)
#                     if area > 0:
#                         complexity = perimeter * perimeter / (4 * np.pi * area)
#                         complexities.append(complexity)
                        
#         return np.mean(complexities) if complexities else 0.0

#     def _calculate_completeness_score(self, masks: List[Dict]) -> float:
#         """Calculate how well each mask represents a complete object."""
#         print("\nCalculating completeness scores...")
#         if not masks:
#             print("No masks to evaluate")
#             return 0.0
            
#         completeness_scores = []
#         for idx, mask in enumerate(masks):
#             try:
#                 # Calculate mask solidity (area / convex hull area)
#                 contours, _ = cv2.findContours(
#                     mask['segmentation'].astype(np.uint8),
#                     cv2.RETR_EXTERNAL,
#                     cv2.CHAIN_APPROX_SIMPLE
#                 )
                
#                 if contours:
#                     hull = cv2.convexHull(contours[0])
#                     hull_area = cv2.contourArea(hull)
#                     if hull_area > 0:
#                         solidity = mask['area'] / hull_area
#                         boundary_conf = mask.get('stability_score', 0.5)
#                         completeness = (solidity + boundary_conf) / 2
#                         completeness_scores.append(completeness)
                        
#                         print(f"Mask {idx}: solidity={solidity:.3f}, "
#                               f"boundary_conf={boundary_conf:.3f}, "
#                               f"completeness={completeness:.3f}")
#                     else:
#                         print(f"Mask {idx}: Invalid hull area")
#                 else:
#                     print(f"Mask {idx}: No contours found")
                
#             except Exception as e:
#                 print(f"Error processing mask {idx}: {str(e)}")
        
#         avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
#         print(f"Average completeness score: {avg_completeness:.3f}")
#         return avg_completeness

#     def _calculate_score(self, metrics: SAMTuningMetrics) -> float:
#         """Calculate overall score with emphasis on complete object segmentation."""
#         print("\nCalculating configuration score...")
#         score = 0.0
        
#         # Apply hard constraints
#         if self.max_memory_gb and metrics.memory_usage > self.max_memory_gb:
#             print(f"Memory usage {metrics.memory_usage:.2f}GB exceeds limit {self.max_memory_gb}GB")
#             return float('-inf')
#         if self.max_processing_time and metrics.processing_time > self.max_processing_time:
#             print(f"Processing time {metrics.processing_time:.2f}s exceeds limit {self.max_processing_time}s")
#             return float('-inf')
#         if metrics.completeness_score < self.min_completeness:
#             print(f"Completeness score {metrics.completeness_score:.3f} below threshold {self.min_completeness}")
#             return float('-inf')
            
#         # Calculate score components
#         completeness_component = metrics.completeness_score * 3.0
#         boundary_component = metrics.boundary_confidence * 2.0
#         overlap_penalty = metrics.overlap_ratio * 3.0
        
#         print("Score components:")
#         print(f"- Completeness (x3.0): {completeness_component:.3f}")
#         print(f"- Boundary confidence (x2.0): {boundary_component:.3f}")
#         print(f"- Overlap penalty (x3.0): -{overlap_penalty:.3f}")
        
#         score = completeness_component + boundary_component - overlap_penalty
        
#         # Size score
#         size_score = 0.0
#         if self.min_object_size:
#             normalized_area = metrics.avg_mask_area / (self.min_object_size * 4)
#             size_score = min(normalized_area, 1.0)
#             print(f"- Size score: {size_score:.3f} (avg_area={metrics.avg_mask_area:.1f})")
#         score += size_score
        
#         # Complexity penalty
#         complexity_penalty = max(0, metrics.edge_complexity - 1.2) * 0.5
#         print(f"- Complexity penalty (x0.5): -{complexity_penalty:.3f}")
#         score -= complexity_penalty
        
#         # Coverage penalty
#         coverage_penalty = abs(metrics.coverage_ratio - self.target_coverage) * 0.5
#         print(f"- Coverage penalty (x0.5): -{coverage_penalty:.3f}")
#         score -= coverage_penalty
        
#         print(f"Final score: {score:.3f}")
#         return score

#     def evaluate_configuration(
#         self,
#         image: np.ndarray,
#         config: Dict
#     ) -> Optional[SAMTuningMetrics]:
#         """Evaluate configuration with focus on complete objects."""
#         print("\nEvaluating configuration:")
#         for key, value in config.items():
#             print(f"- {key}: {value}")
        
#         try:
#             start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
#             start_time = time.time()
            
#             print("Generating masks...")
#             mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
#                 self.model_name,
#                 points_per_batch=self.batch_size,
#                 **config
#             )
            
#             masks = mask_generator.generate(image)
#             print(f"Generated {len(masks)} masks")
            
#             end_time = time.time()
#             end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
#             if not masks:
#                 print("No masks generated")
#                 return None
                
#             # Calculate metrics
#             areas = [mask['area'] for mask in masks]
#             avg_area = np.mean(areas) if areas else 0
#             print(f"Average mask area: {avg_area:.1f}")
            
#             print("Calculating coverage...")
#             total_coverage = np.zeros(image.shape[:2], dtype=bool)
#             for mask in masks:
#                 total_coverage |= mask['segmentation']
#             coverage_ratio = total_coverage.sum() / (image.shape[0] * image.shape[1])
#             print(f"Coverage ratio: {coverage_ratio:.2%}")
            
#             # Calculate additional metrics
#             completeness_score = self._calculate_completeness_score(masks)
#             boundary_confidence = np.mean([
#                 mask.get('stability_score', 0.5) for mask in masks
#             ]) if masks else 0.0
#             print(f"Average boundary confidence: {boundary_confidence:.3f}")
            
#             metrics = SAMTuningMetrics(
#                 avg_mask_area=avg_area,
#                 mask_count=len(masks),
#                 overlap_ratio=self._calculate_overlap_ratio(masks),
#                 processing_time=end_time - start_time,
#                 memory_usage=(end_mem - start_mem) / 1024**3,
#                 coverage_ratio=coverage_ratio,
#                 edge_complexity=self._calculate_edge_complexity(masks),
#                 completeness_score=completeness_score,
#                 boundary_confidence=boundary_confidence
#             )
            
#             print("\nMetrics summary:")
#             print(f"- Processing time: {metrics.processing_time:.2f}s")
#             print(f"- Memory usage: {metrics.memory_usage:.2f}GB")
#             print(f"- Mask count: {metrics.mask_count}")
#             print(f"- Average area: {metrics.avg_mask_area:.1f}")
#             print(f"- Coverage: {metrics.coverage_ratio:.2%}")
#             print(f"- Overlap ratio: {metrics.overlap_ratio:.3f}")
#             print(f"- Edge complexity: {metrics.edge_complexity:.3f}")
#             print(f"- Completeness: {metrics.completeness_score:.3f}")
#             print(f"- Boundary confidence: {metrics.boundary_confidence:.3f}")
            
#             return metrics
            
#         except Exception as e:
#             print(f"Configuration evaluation failed: {e}")
#             return None

#     def optimize_parameters(
#         self,
#         image: np.ndarray,
#         n_trials: int = 20,
#         use_bayesian: bool = True
#     ) -> Tuple[Dict, SAMTuningMetrics]:
#         """Find optimal parameters using Bayesian optimization or random search."""
#         if use_bayesian:
#             try:
#                 from skopt import gp_minimize
#                 from skopt.space import Real, Integer
                
#                 space = [
#                     Integer(16, 64, name='points_per_side'),
#                     Real(0.7, 0.95, name='pred_iou_thresh'),
#                     Real(0.7, 0.97, name='stability_score_thresh'),
#                     Integer(1, 3, name='crop_n_layers'),
#                     Real(0.5, 0.9, name='box_nms_thresh'),
#                     Real(50.0, 200.0, name='min_mask_region_area')
#                 ]
                
#                 def objective(params):
#                     config = {
#                         'points_per_side': int(params[0]),
#                         'pred_iou_thresh': params[1],
#                         'stability_score_thresh': params[2],
#                         'crop_n_layers': int(params[3]),
#                         'box_nms_thresh': params[4],
#                         'min_mask_region_area': params[5],
#                         'use_m2m': True
#                     }
                    
#                     metrics = self.evaluate_configuration(image, config)
#                     if metrics is None:
#                         return float('inf')
#                     return -self._calculate_score(metrics)
                
#                 result = gp_minimize(
#                     objective,
#                     space,
#                     n_calls=n_trials,
#                     noise=0.1,
#                     n_jobs=-1
#                 )
                
#                 best_config = {
#                     'points_per_side': int(result.x[0]),
#                     'pred_iou_thresh': result.x[1],
#                     'stability_score_thresh': result.x[2],
#                     'crop_n_layers': int(result.x[3]),
#                     'box_nms_thresh': result.x[4],
#                     'min_mask_region_area': result.x[5],
#                     'use_m2m': True
#                 }
                
#                 best_metrics = self.evaluate_configuration(image, best_config)
#                 return best_config, best_metrics
                
#             except ImportError:
#                 print("Scikit-optimize not found, falling back to random search")
#                 use_bayesian = False
        
#         return self._random_search(image, n_trials)

#     def _random_search(self, image: np.ndarray, n_trials: int) -> Tuple[Dict, SAMTuningMetrics]:
#         """Random search with parameters optimized for complete objects."""
#         print("\nStarting random search optimization...")
#         print(f"Number of trials: {n_trials}")
        
#         best_config = None
#         best_metrics = None
#         best_score = float('-inf')
        
#         param_ranges = {
#             'points_per_side': (24, 48),
#             'pred_iou_thresh': (0.85, 0.95),
#             'stability_score_thresh': (0.90, 0.97),
#             'crop_n_layers': (1, 2),
#             'box_nms_thresh': (0.6, 0.8),
#             'min_mask_region_area': (
#                 self.min_object_size if self.min_object_size 
#                 else 100.0,
#                 self.min_object_size * 2 if self.min_object_size 
#                 else 200.0
#             )
#         }
        
#         print("\nParameter ranges:")
#         for param, (min_val, max_val) in param_ranges.items():
#             print(f"- {param}: [{min_val}, {max_val}]")
        
#         for trial in range(n_trials):
#             print(f"\nTrial {trial + 1}/{n_trials}")
#             print("-" * 40)
            
#             config = {
#                 'points_per_side': int(np.random.uniform(*param_ranges['points_per_side'])),
#                 'pred_iou_thresh': np.random.uniform(*param_ranges['pred_iou_thresh']),
#                 'stability_score_thresh': np.random.uniform(*param_ranges['stability_score_thresh']),
#                 'crop_n_layers': int(np.random.uniform(*param_ranges['crop_n_layers'])),
#                 'box_nms_thresh': np.random.uniform(*param_ranges['box_nms_thresh']),
#                 'min_mask_region_area': np.random.uniform(*param_ranges['min_mask_region_area']),
#                 'use_m2m': True
#             }
            
#             metrics = self.evaluate_configuration(image, config)
#             if metrics is None:
#                 print("Trial failed, skipping...")
#                 continue
                
#             score = self._calculate_score(metrics)
            
#             if score > best_score:
#                 best_score = score
#                 best_config = config
#                 best_metrics = metrics
#                 print("\n🌟 New best configuration found!")
#                 print(f"Best score so far: {best_score:.3f}")
        
#         print("\nOptimization complete!")
#         print(f"Best score achieved: {best_score:.3f}")
#         print("\nBest configuration:")
#         for key, value in best_config.items():
#             print(f"- {key}: {value}")
        
#         return best_config, best_metrics 