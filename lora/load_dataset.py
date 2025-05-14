from datasets import load_dataset
from torchvision.transforms import Compose, ColorJitter, ToTensor

DATASET_DIR = 'dataset'
SAMPLES_TO_SHOW = 5

# Load dataset
dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)

# Print the number of samples in the dataset.
print(f'\n› The dataset at {DATASET_DIR} contains {dataset["train"].num_rows} captioned images.\n')

# Print a few samples.
print(f'› Displaying {SAMPLES_TO_SHOW} samples.\n')
for sample in list(dataset['train'])[:SAMPLES_TO_SHOW]:
    print(sample["text"])

print()