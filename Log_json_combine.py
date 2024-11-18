import json
import os

def load_and_combine_json_files(directory, output_file):
    combined_data = []

    # Iterate through all files in the directory
    for dir in os.listdir(directory):
        print(dir)
        if(os.path.isdir(os.path.join(directory,dir))):
            for filename in os.listdir(os.path.join(directory, dir,"vis_data")):
                if filename.endswith(".json"):
                    file_path = os.path.join(directory,dir,"vis_data", filename)
                    print(file_path)
                    with open(file_path, 'r',encoding='utf-8') as file:
                        for line in file.readlines():
                            data = json.loads(line)
                            combined_data.append(data)

    # Write combined data to a new JSON file
    with open(output_file, 'w') as output:
        json.dump(combined_data, output, indent=4)

# Directory containing the JSON files
json_directory = 'FYP_mmdetection\work_dirs\conditional-detr_r50_8xb2-50e_coco'

# Output file for the combined JSON data
output_json_file = 'FYP_mmdetection\work_dirs\conditional-detr_r50_8xb2-50e_coco\combined_output.json'

# Call the function to combine JSON files
load_and_combine_json_files(json_directory, output_json_file)

print(f"Combined JSON data has been written to {output_json_file}")
