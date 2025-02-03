# @TODO: Implement depth estimation using the Depth Anything model, need to download the model as a github submodule and build it first - can set up a fastapi server to serve the model later


# import torch
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from typing import Union, Optional
# import logging
# # from promptda.promptda import PromptDA
# # from promptda.utils.io_wrapper import load_image, load_depth, save_depth

# logger = logging.getLogger(__name__)

# class DepthEstimator:
#     """
#     A class to handle depth estimation using the Depth Anything model
#     """
    
#     def __init__(
#         self,
#         checkpoint_path: str = "model.ckpt",
#         device: str = "cuda"
#     ):
#         """
#         Initialize the depth estimator
        
#         Args:
#             checkpoint_path: Path to model checkpoint file
#             device: Device to run model on ("cuda" or "cpu")
#         """
#         self.device = device
#         self.checkpoint_path = checkpoint_path
        
#         logger.info(f"Loading Depth Anything model from checkpoint: {checkpoint_path}")
#         self.model = PromptDA()
#         state_dict = torch.load(checkpoint_path, map_location=device)
#         self.model.load_state_dict(state_dict)
#         self.model.to(device).eval()
#         logger.info("Model loaded successfully")
        
#     def estimate_depth(
#         self,
#         image: Union[str, Path, Image.Image, torch.Tensor],
#         prompt_depth: Optional[Union[str, Path, torch.Tensor]] = None
#     ) -> torch.Tensor:
#         """
#         Estimate depth from an image with optional prompt depth
        
#         Args:
#             image: Input image (path, PIL Image, or tensor)
#             prompt_depth: Optional prompt depth (path or tensor)
            
#         Returns:
#             Estimated depth map as tensor (HxW, in meters)
#         """
#         # Load and process image
#         if isinstance(image, (str, Path)):
#             image = load_image(str(image))
#         elif isinstance(image, Image.Image):
#             image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
            
#         image = image.to(self.device)
        
#         # Load and process prompt depth if provided
#         if prompt_depth is not None:
#             if isinstance(prompt_depth, (str, Path)):
#                 prompt_depth = load_depth(str(prompt_depth))
#             prompt_depth = prompt_depth.to(self.device)
        
#         # Run inference
#         with torch.no_grad():
#             depth = self.model.predict(image, prompt_depth)
            
#         return depth
        
#     def save_depth_visualization(
#         self,
#         depth: torch.Tensor,
#         output_path: Union[str, Path],
#         prompt_depth: Optional[torch.Tensor] = None,
#         image: Optional[torch.Tensor] = None
#     ) -> None:
#         """
#         Save depth visualization
        
#         Args:
#             depth: Estimated depth map
#             output_path: Path to save visualization
#             prompt_depth: Optional prompt depth for comparison
#             image: Optional input image for side-by-side visualization
#         """
#         save_depth(
#             depth,
#             str(output_path),
#             prompt_depth=prompt_depth,
#             image=image
#         )
#         logger.info(f"Saved depth visualization to {output_path}") 