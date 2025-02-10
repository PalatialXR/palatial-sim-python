import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
from pathlib import Path

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def add_inertial_properties(link_element, link_name):
    """Add inertial properties to a link if they don't exist."""
    # Check if inertial properties already exist
    if link_element.find('inertial') is not None:
        return

    # Create inertial element
    inertial = ET.SubElement(link_element, 'inertial')
    
    # Set mass based on link type
    mass = ET.SubElement(inertial, 'mass')
    if link_name == "base":
        mass.set('value', '5.0')  # Base is usually heavier
    else:
        # For moving parts, use a reasonable mass
        mass.set('value', '1.0')
    
    # Add origin - try to use the first visual element's origin if available
    visual = link_element.find('visual')
    if visual is not None and visual.find('origin') is not None:
        visual_origin = visual.find('origin')
        origin = ET.SubElement(inertial, 'origin')
        if 'xyz' in visual_origin.attrib:
            origin.set('xyz', visual_origin.get('xyz'))
        else:
            origin.set('xyz', '0 0 0')
        if 'rpy' in visual_origin.attrib:
            origin.set('rpy', visual_origin.get('rpy'))
        else:
            origin.set('rpy', '0 0 0')
    else:
        origin = ET.SubElement(inertial, 'origin')
        origin.set('xyz', '0 0 0')
        origin.set('rpy', '0 0 0')
    
    # Add inertia tensor with more appropriate values
    inertia = ET.SubElement(inertial, 'inertia')
    if link_name == "base":
        # Larger inertia for base
        inertia.set('ixx', '0.5')
        inertia.set('ixy', '0')
        inertia.set('ixz', '0')
        inertia.set('iyy', '0.5')
        inertia.set('iyz', '0')
        inertia.set('izz', '0.5')
    else:
        # Smaller inertia for moving parts
        inertia.set('ixx', '0.1')
        inertia.set('ixy', '0')
        inertia.set('ixz', '0')
        inertia.set('iyy', '0.1')
        inertia.set('iyz', '0')
        inertia.set('izz', '0.1')

def update_urdf_file(urdf_path):
    """Update a single URDF file with inertial properties."""
    try:
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Find all links
        links = root.findall('link')
        
        # Add inertial properties to each link
        for link in links:
            link_name = link.get('name', '')
            add_inertial_properties(link, link_name)
        
        # Save the modified URDF with proper formatting
        with open(urdf_path, 'w', encoding='utf-8') as f:
            xml_str = prettify_xml(root)
            # Remove extra blank lines
            xml_str = '\n'.join(line for line in xml_str.split('\n') if line.strip())
            f.write(xml_str)
        return True
    except Exception as e:
        print(f"Error processing {urdf_path}: {str(e)}")
        return False

def update_all_urdfs(dataset_path):
    """Update all URDF files in the dataset."""
    dataset_path = Path(dataset_path)
    success_count = 0
    failure_count = 0
    
    # Walk through all subdirectories
    for obj_dir in dataset_path.iterdir():
        if not obj_dir.is_dir():
            continue
            
        urdf_path = obj_dir / 'mobility.urdf'
        if urdf_path.exists():
            if update_urdf_file(urdf_path):
                success_count += 1
                print(f"Successfully updated {urdf_path}")
            else:
                failure_count += 1
                print(f"Failed to update {urdf_path}")
    
    print(f"\nUpdate complete!")
    print(f"Successfully updated: {success_count} files")
    print(f"Failed to update: {failure_count} files")

if __name__ == "__main__":
    # Path to the dataset directory
    dataset_path = "src/datasets/partnet-mobility-v0/dataset"
    update_all_urdfs(dataset_path) 