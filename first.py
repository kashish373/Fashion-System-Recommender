# 📦 Step 1: Import libraries
from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import itertools

# 📘 Step 2: Helper function to convert image to feature vector
def image_to_vector(image_path):
    img = Image.open(image_path).resize((128, 128)).convert('RGB')
    return np.array(img).flatten() / 255.0  # normalize pixels

# 📗 Step 3: Upload images for each category
def upload_category(name):
    print(f"Upload {name} images (you can upload multiple):")
    uploaded = files.upload()
    return list(uploaded.keys())

tops = upload_category("TOPS")
bottoms = upload_category("BOTTOMS")
shoes = upload_category("SHOES")
accessories = upload_category("ACCESSORIES")

# 📙 Step 4: Convert all images to feature vectors
top_vecs = [image_to_vector(p) for p in tops]
bottom_vecs = [image_to_vector(p) for p in bottoms]
shoe_vecs = [image_to_vector(p) for p in shoes]
accessory_vecs = [image_to_vector(p) for p in accessories]

# 📒 Step 5: Compute similarity between outfit items
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Function to compute overall outfit compatibility
def outfit_score(top, bottom, shoe, accessory):
    sim1 = cosine_sim(top, bottom)
    sim2 = cosine_sim(bottom, shoe)
    sim3 = cosine_sim(top, accessory)
    return (sim1 + sim2 + sim3) / 3

# 📕 Step 6: Generate all possible outfit combinations
combinations = list(itertools.product(top_vecs, bottom_vecs, shoe_vecs, accessory_vecs))

scores = []
for i, (t, b, s, a) in enumerate(combinations):
    score = outfit_score(t, b, s, a)
    scores.append(score)

best_idx = np.argmax(scores)
best_combo = combinations[best_idx]
best_score = scores[best_idx]

print(f"\n✅ Best Outfit Score: {best_score:.4f}")

# 📊 Step 7: Show best outfit combination
def show_images(paths, titles):
    plt.figure(figsize=(10, 3))
    for i, (path, title) in enumerate(zip(paths, titles)):
        plt.subplot(1, 4, i+1)
        plt.imshow(Image.open(path))
        plt.title(title)
        plt.axis("off")
    plt.show()

# Calculate indices correctly (accessory changes fastest in product)
accessory_idx = best_idx % len(accessories)
shoe_idx = (best_idx // len(accessories)) % len(shoes)
bottom_idx = (best_idx // (len(accessories) * len(shoes))) % len(bottoms)
top_idx = (best_idx // (len(accessories) * len(shoes) * len(bottoms))) % len(tops)

best_paths = [
    tops[top_idx],
    bottoms[bottom_idx],
    shoes[shoe_idx],
    accessories[accessory_idx]
]

show_images(best_paths, ["Top", "Bottom", "Shoes", "Accessory"])
