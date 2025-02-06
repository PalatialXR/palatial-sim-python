import objaverse
import json
import os

def save_lvis_annotation():
    lvis_annotations = objaverse.load_lvis_annotations()
    with open("lvis_annotations.json", "w") as f:
        json.dump(lvis_annotations, f)

    print("LVIS annotations saved to lvis_annotations.json")

def lvis_annotation_keys():
    lvis_annotations = objaverse.load_lvis_annotations()
    with open("lvis_keys.txt", "w") as f:
        for key in lvis_annotations.keys():
            f.write(key + "\n")

    print("LVIS keys saved to lvis_keys.txt")

def download_lvis_annotations():
    try:
        lvis_annotations = objaverse.load_lvis_annotations()
        object_path = "downloaded_objects"  # Directory to save downloaded objects
        
        # Create directory if it doesn't exist
        if not os.path.exists(object_path):
            os.makedirs(object_path)
            
        total_downloads = 1
        start_file_count = 0
        
        # save the table object
        table_object = lvis_annotations["table"][0]
        
        # Use proper download method instead of internal _download_object
        objaverse._download_object([table_object], object_path=object_path, start_file_count=0, total_downloads=1)
        
    except Exception as e:
        print(f"Error downloading LVIS annotations: {str(e)}")

if __name__ == "__main__":
    save_lvis_annotation()
    lvis_annotation_keys()
    download_lvis_annotations()