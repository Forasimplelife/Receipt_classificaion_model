import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import time
from torchvision import transforms
from nets.resnet import resnet34  # Assuming resnet34 is defined in the nets module
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

# data load, transdorm and pre plot
from datamodule.dataloader import RotatedReceiptDataset, get_data_transforms
from plotmodule.plot_utils import plot_samples_from_each_class 



# Now you can call the function with any image size
data_transforms = get_data_transforms(image_size=224)

# Load the dataset for testing
prediction_dataset = RotatedReceiptDataset(root_dir='./data/Receipt_data/prediction', transform=data_transforms['pre'])

# Print the length (number of samples) in the dataset
print(f"Number of prediction samples: {len(prediction_dataset)}")

# Function to print the shape and label of randomly selected images
def print_random_samples(dataset, num_samples=5):
    # Randomly select indices from the dataset
    random_indices = random.sample(range(len(dataset)), num_samples)

    # Print the details of the randomly selected images
    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        print(f"Sample {i + 1}:")
        print(f" - Image shape: {image.shape}")  # Tensor shape: (Channels, Height, Width)
        print(f" - Label: {label}")

# Randomly print 5 images from the training dataset
print("\nRandom sample dataset:")
print_random_samples(prediction_dataset, num_samples=5)

# Call the function to plot 5 random images from each of the 4 classes in the training dataset
plot_samples_from_each_class(prediction_dataset, num_samples=5)

# Define the model path and batch size
MODEL_PATH = './logs/Epoch10-Total_Loss0.0067.pth'  # Update the model path to your trained model

# Prompt the user for the batch size, defaulting to 4 if no input is given
batch_size_input = input('Enter the number of images to predict at a time (default is 4): ')
Batch_Size = int(batch_size_input) if batch_size_input else 4

# DataLoader for the test dataset
gen_test = DataLoader(dataset=prediction_dataset, batch_size=Batch_Size, shuffle=True)

# Load the trained model
model = resnet34(num_classes=4)  # Adjust the number of classes if needed
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Prediction loop using a for loop to go through all batches
for batch_idx, (images, labels) in enumerate(gen_test):
    # Display the batch of images using torchvision.utils.make_grid
    img = torchvision.utils.make_grid(images, nrow=Batch_Size)
    img_array = img.numpy().transpose(1, 2, 0)

    # Move images to the appropriate device
    images = images.to(device)

    # Perform the prediction
    start_time = time.time()
    outputs = model(images)
    
    # If outputs is a tuple, get the first element
    if isinstance(outputs, tuple):
       outputs = outputs[0]

    # Use softmax to get prediction probabilities
    probabilities = F.softmax(outputs, dim=1)

    # Use torch.max to get the predicted class
    _, predicted = torch.max(outputs, 1)
    end_time = time.time()

    # Print the prediction results, probabilities, and time taken
    print(f'Batch {batch_idx + 1} - Prediction time:', end_time - start_time)
    print('Predicted labels:', predicted.cpu().numpy())
    print('Prediction probabilities:', probabilities.cpu().detach().numpy())

    # Display the images using matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(img_array)
    plt.title(f'Batch {batch_idx + 1} - Predicted Labels: {predicted.cpu().numpy()}')
    plt.axis('off')  # Hide the axis for better visualization
    plt.show()


