
import os

def get_dataset_length(dataset_dir):
    return len([name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))])

# Example usage:
root_hq_dir_test = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
length_test_dataset = get_dataset_length(root_hq_dir_test)
print(f"Length of test dataset: {length_test_dataset}")
