import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from .interactive_segmenter import InteractiveSegmenter

logger = logging.getLogger(__name__)

class InteractiveSegmentationUI:
    def __init__(self, segmenter: InteractiveSegmenter):
        self.segmenter = segmenter
        self.original_image = self.segmenter.image.copy()  # Keep original image
        self.working_image = self.segmenter.image.copy()  # Image with masked objects removed
        self.points = []  # List of (x, y) coordinates
        self.labels = []  # List of 0/1 labels
        self.box = None   # Optional bounding box [x1, y1, x2, y2]
        self.drawing_box = False
        self.box_start = None
        self.current_label = 1  # 1 for foreground, 0 for background
        self.window_name = "Interactive Segmentation"
        self.overlay = None
        self.masks = []
        self.scores = []
        self.mask_opacity = 0.5  # Adjustable mask opacity
        self.show_scores = True  # Toggle score display
        self.multimask_output = True  # Whether to show multiple mask options
        self.saved_masks = []  # List to store all saved masks
        self.saved_scores = []  # List to store all saved scores
        
        # Calculate window size (max 800x800, maintaining aspect ratio)
        h, w = self.original_image.shape[:2]
        self.scale = min(800 / max(h, w), 1.0)
        self.window_size = (int(w * self.scale), int(h * self.scale))
        
    def _scale_point(self, x: int, y: int, inverse: bool = False) -> Tuple[int, int]:
        """Scale point coordinates between window and original image space"""
        if inverse:
            # Window to image space
            return (int(x / self.scale), int(y / self.scale))
        else:
            # Image to window space
            return (int(x * self.scale), int(y * self.scale))
            
    def mouse_callback(self, event, x, y, flags, param):
        # Convert window coordinates to image coordinates
        img_x, img_y = self._scale_point(x, y, inverse=True)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing_box:
                # Add point
                self.points.append([img_x, img_y])
                self.labels.append(self.current_label)
                self._update_segmentation()
            else:
                # Start drawing box
                self.box_start = (img_x, img_y)
                
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_box and self.box_start:
            # Update box preview
            self._update_display(preview_box=[
                self.box_start[0], self.box_start[1], 
                img_x, img_y
            ])
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing_box and self.box_start:
            # Finish drawing box
            self.box = [
                min(self.box_start[0], img_x),
                min(self.box_start[1], img_y),
                max(self.box_start[0], img_x),
                max(self.box_start[1], img_y)
            ]
            self.drawing_box = False
            self.box_start = None
            self._update_segmentation()

    def _update_segmentation(self):
        if not self.points:
            return
            
        points = np.array(self.points)
        labels = np.array(self.labels)
        box = np.array(self.box) if self.box is not None else None
        
        # Update segmenter's image to working image
        self.segmenter.set_image(self.working_image)
        
        masks, scores, _ = self.segmenter.predict_from_points(
            points=points,
            labels=labels,
            box=box,
            multimask_output=self.multimask_output
        )
        
        # If multiple masks returned, keep only the highest scoring one
        if not self.multimask_output and len(masks.shape) == 3:
            best_idx = np.argmax(scores)
            masks = masks[best_idx:best_idx+1]
            scores = scores[best_idx:best_idx+1]
        
        self.masks = masks
        self.scores = scores
        self._update_display()
        
    def _save_current_mask(self):
        """Save current mask and remove it from working image"""
        if self.masks is not None and len(self.masks) > 0:
            # Get best mask if multiple
            if len(self.masks) > 1:
                best_idx = np.argmax(self.scores)
                mask = self.masks[best_idx]
                score = self.scores[best_idx]
            else:
                mask = self.masks[0]
                score = self.scores[0]
            
            # Save mask and score
            self.saved_masks.append(mask)
            self.saved_scores.append(score)
            
            # Remove segmented object from working image
            mask_bool = mask.astype(bool)
            self.working_image[mask_bool] = 0
            
            # Clear current points and masks
            self.points = []
            self.labels = []
            self.masks = None
            self.scores = None
            
            logger.info(f"Saved mask {len(self.saved_masks)} with score {score:.3f}")
            return True
        return False
        
    def _update_display(self, preview_box=None):
        # Start with the working image
        display = self.working_image.copy()
        
        # Draw current masks
        if self.masks is not None:
            for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
                # Convert mask to boolean and create colored overlay
                mask_bool = mask.astype(bool)
                color = np.random.random(3) * 255
                mask_overlay = np.zeros_like(display)
                mask_overlay[mask_bool] = color
                cv2.addWeighted(mask_overlay, self.mask_opacity, display, 1, 0, display)
                
                # Show scores if enabled
                if self.show_scores:
                    score_text = f"Current mask {i}: {score:.2f}"
                    cv2.putText(display, score_text, 
                              (10, 180 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 255), 1)
        
        # Draw points with numbers
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            # Draw star marker
            size = 10
            center = tuple(map(int, point))  # Ensure integer coordinates
            for j in range(8):  # 8-point star
                angle = j * np.pi / 4
                end_x = int(point[0] + size * np.cos(angle))
                end_y = int(point[1] + size * np.sin(angle))
                cv2.line(display, center, (end_x, end_y), color, 2)
            # Add white outline
            cv2.circle(display, center, size//2, (255, 255, 255), 1)
            # Add point number
            cv2.putText(display, str(i), 
                       (point[0] + 12, point[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Draw box if present
        if self.box is not None:
            cv2.rectangle(display, 
                        (int(self.box[0]), int(self.box[1])),
                        (int(self.box[2]), int(self.box[3])),
                        (0, 255, 0), 2)
                        
        # Draw preview box
        if preview_box is not None:
            cv2.rectangle(display,
                        (int(preview_box[0]), int(preview_box[1])),
                        (int(preview_box[2]), int(preview_box[3])),
                        (0, 255, 0), 2)
                        
        # Show instructions (in a more compact format)
        instructions = [
            "Left click: Add point (green=fg, red=bg) | 'b': Toggle fg/bg | 'c': Clear",
            "'z': Undo | 'm': Toggle multi-mask | 'o': Opacity | 't': Scores",
            "'SPACE': Save current object | 's': Save all and exit | 'q': Quit",
            f"Mode: {'Foreground' if self.current_label == 1 else 'Background'} | "
            f"Objects saved: {len(self.saved_masks)}"
        ]
        
        # Add instructions with dark background for better visibility
        for i, text in enumerate(instructions):
            # Draw background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(display, 
                         (5, 15 + i*20), 
                         (10 + text_w, 15 + (i+1)*20), 
                         (0, 0, 0), -1)
            # Draw text
            cv2.putText(display, text, 
                       (8, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
        
        # Resize window
        display = cv2.resize(display, self.window_size)
        cv2.imshow(self.window_name, display)
        
    def run(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run the interactive segmentation session."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('b'):
                # Toggle foreground/background
                self.current_label = 1 - self.current_label
                self._update_display()
                
            elif key == ord('c'):
                # Clear all
                self.points = []
                self.labels = []
                self.box = None
                self.masks = None
                self.scores = None
                self._update_display()
                
            elif key == ord('z'):
                # Undo last point
                if self.points:
                    self.points.pop()
                    self.labels.pop()
                    self._update_segmentation()
                    
            elif key == ord('m'):
                # Toggle multimask output
                self.multimask_output = not self.multimask_output
                self._update_segmentation()
                
            elif key == ord('o'):
                # Adjust opacity
                self.mask_opacity = (self.mask_opacity + 0.2) % 1.1
                if self.mask_opacity < 0.1: self.mask_opacity = 0.1
                self._update_display()
                
            elif key == ord('t'):
                # Toggle score display
                self.show_scores = not self.show_scores
                self._update_display()
                
            elif key == ord(' '):  # Spacebar
                # Save current mask and continue
                if self._save_current_mask():
                    self._update_display()
                
            elif key == ord('s'):
                # Return all saved masks and scores
                cv2.destroyAllWindows()
                if self.saved_masks:
                    return np.array(self.saved_masks), np.array(self.saved_scores)
                break
                
        cv2.destroyAllWindows()
        return None, None 