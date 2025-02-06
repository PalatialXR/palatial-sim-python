from typing import Dict, Set, Optional
import trimesh
import logging

logger = logging.getLogger(__name__)

class CategoryManager:
    """Manages object categories and their mappings to LVIS categories."""
    
    def __init__(self):
        # Primary categories that map to multiple LVIS categories
        self.category_mappings = {
            'furniture': {
                'desk', 'table', 'chair', 'sofa', 'cabinet', 'shelf', 'bookcase',
                'bed', 'drawer', 'wardrobe', 'ottoman', 'stool'
            },
            'electronic': {
                'computer', 'monitor', 'laptop', 'keyboard', 'mouse', 'printer',
                'scanner', 'speaker', 'television', 'phone', 'camera', 'tablet'
            },
            'container': {
                'box', 'bin', 'basket', 'container', 'jar', 'vase', 'bowl',
                'pot', 'bottle', 'cup', 'glass'
            },
            'accessory': {
                'bag', 'backpack', 'purse', 'wallet', 'case', 'holder',
                'stand', 'hanger', 'organizer'
            },
            'lighting': {
                'lamp', 'light', 'chandelier', 'sconce', 'lantern',
                'flashlight', 'torch'
            },
            'tool': {
                'hammer', 'screwdriver', 'wrench', 'pliers', 'saw',
                'drill', 'measure', 'level', 'clamp'
            },
            'kitchenware': {
                'pan', 'pot', 'utensil', 'plate', 'bowl', 'cup',
                'glass', 'knife', 'fork', 'spoon'
            },
            'decoration': {
                'art', 'painting', 'sculpture', 'statue', 'vase',
                'frame', 'mirror', 'clock', 'plant'
            }
        }

        # Properties that should be analyzed for each object
        self.object_properties = {
            'size': ['small', 'medium', 'large'],
            'material': ['wood', 'metal', 'plastic', 'glass', 'fabric'],
            'shape': ['rectangular', 'circular', 'irregular', 'cylindrical'],
            'complexity': ['simple', 'moderate', 'complex'],
            'style': ['modern', 'traditional', 'industrial', 'minimalist']
        }

    def get_category_variations(self, category: str) -> Set[str]:
        """Generate variations of a category name for better matching."""
        variations = set()
        base_name = category.lower()
        
        # Add original
        variations.add(base_name)
        
        # Add with different separators
        variations.add(base_name.replace(' ', '_'))
        variations.add(base_name.replace('_', ' '))
        
        # Add plural/singular variations
        variations.add(base_name + 's')
        if base_name.endswith('s'):
            variations.add(base_name[:-1])
            
        # Add compound variations
        if '_' in base_name:
            parts = base_name.split('_')
            variations.update(parts)
            
        # Add mapped categories if available
        if base_name in self.category_mappings:
            variations.update(self.category_mappings[base_name])
            
        return variations

    def analyze_object_properties(self, mesh) -> Dict[str, str]:
        """Analyze mesh properties to determine object characteristics."""
        properties = {}
        
        try:
            # Analyze size based on bounding box
            bbox = mesh.bounding_box.extents
            volume = mesh.volume if hasattr(mesh, 'volume') else 0
            
            # Determine size category
            max_dim = max(bbox)
            if max_dim < 0.5:  # 50cm threshold
                properties['size'] = 'small'
            elif max_dim < 1.5:  # 150cm threshold
                properties['size'] = 'medium'
            else:
                properties['size'] = 'large'
                
            # Analyze shape
            aspect_ratios = bbox / max(bbox)
            if all(0.8 < r < 1.2 for r in aspect_ratios):
                properties['shape'] = 'cubic'
            elif any(r < 0.2 for r in aspect_ratios):
                properties['shape'] = 'flat'
            else:
                properties['shape'] = 'irregular'
                
            # Analyze complexity
            vertex_density = len(mesh.vertices) / volume if volume > 0 else 0
            if vertex_density < 1000:
                properties['complexity'] = 'simple'
            elif vertex_density < 5000:
                properties['complexity'] = 'moderate'
            else:
                properties['complexity'] = 'complex'
                
        except Exception as e:
            logger.warning(f"Error analyzing object properties: {e}")
            
        return properties

    def score_object_match(self, obj_data: Dict, target_properties: Dict) -> float:
        """Score how well an object matches desired properties."""
        score = 0.0
        max_score = len(target_properties)
        
        # Compare each target property
        for prop, target_value in target_properties.items():
            if prop in obj_data:
                if obj_data[prop] == target_value:
                    score += 1.0
                elif prop in self.object_properties:
                    # Partial match for categorical properties
                    prop_values = self.object_properties[prop]
                    try:
                        target_idx = prop_values.index(target_value)
                        actual_idx = prop_values.index(obj_data[prop])
                        score += 1.0 - (abs(target_idx - actual_idx) / len(prop_values))
                    except ValueError:
                        continue
                    
        return score / max_score if max_score > 0 else 0.0 