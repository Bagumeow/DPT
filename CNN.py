import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(self.vgg16.features.children()))

    def forward(self, x):
        x = self.features(x)
        return x


def extract_features_cnn(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model = VGG16FeatureExtractor()
    model.eval()

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()


def extract_and_save_features(data_folder, output_file):
    features_dict = {}
    for filename in os.listdir(data_folder):
        print(f"Extracting features from {filename}")
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(data_folder, filename)
            features = extract_features_cnn(image_path)
            features_dict[filename] = features

    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)


def compare_with_saved_features(input_image_path: str, features_file: str = './data/features.pkl'):
    input_features = extract_features_cnn(input_image_path)

    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)

    similarities = {}
    for filename, features in features_dict.items():
        similarity = cosine_similarity([input_features], [features])[0][0]
        similarities[filename] = similarity
    sorted_similarities = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True)

    return sorted_similarities


# Example usage
if __name__ == "__main__":
    data_folder = 'data'
    output_file = 'features.pkl'

    # Extract and save features
    extract_and_save_features(data_folder, output_file)
    print("Finish extracting and saving features")
    # Compare input image with saved features
    # input_image_path = 'path/to/your/input_image.jpg'  # Replace with your input image path
    # similarities = compare_with_saved_features(input_image_path, output_file)

    # # Print the sorted similarities
    # for filename, similarity in similarities:
    #     print(f"{filename}: {similarity}")
