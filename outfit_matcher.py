"""
Image-based outfit matching module.
Converts wardrobe items to image feature vectors and finds best matching outfits
based on cosine similarity between images.
"""
import os
import itertools
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from numpy.linalg import norm

from models import WardrobeItem


def image_to_vector(image_path: str) -> Optional[np.ndarray]:
    """
    Convert an image file to a normalized feature vector.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Flattened normalized pixel array or None if error
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        img = Image.open(image_path).resize((128, 128)).convert('RGB')
        return np.array(img).flatten() / 255.0  # normalize pixels
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: numpy arrays
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        a_norm = norm(a)
        b_norm = norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))
    except Exception:
        return 0.0


def outfit_score(top: np.ndarray, bottom: np.ndarray, 
                 shoe: np.ndarray, accessory: np.ndarray) -> float:
    """
    Compute overall outfit compatibility score based on pairwise similarities.
    
    Args:
        top, bottom, shoe, accessory: Feature vectors for each item
        
    Returns:
        Average similarity score
    """
    sim1 = cosine_sim(top, bottom)
    sim2 = cosine_sim(bottom, shoe)
    sim3 = cosine_sim(top, accessory)
    sim4 = cosine_sim(shoe, accessory)  # Additional similarity check
    return (sim1 + sim2 + sim3 + sim4) / 4.0


def categorize_items(items: List[WardrobeItem]) -> Dict[str, List[WardrobeItem]]:
    """
    Categorize wardrobe items into tops, bottoms, shoes, and accessories.
    
    Args:
        items: List of WardrobeItem objects
        
    Returns:
        Dictionary with keys 'tops', 'bottoms', 'shoes', 'accessories'
    """
    categories = {
        'tops': [],
        'bottoms': [],
        'shoes': [],
        'accessories': []
    }
    
    for item in items:
        if not item.image_url:
            continue  # Skip items without images
            
        cat_lower = (item.category or '').lower()
        
        # Categorize based on category name
        if cat_lower in {'shirt', 'tshirt', 'top', 'blouse', 'sweater', 'hoodie', 'blazer', 'cardigan'}:
            categories['tops'].append(item)
        elif cat_lower in {'pants', 'trousers', 'jeans', 'skirt', 'shorts'}:
            categories['bottoms'].append(item)
        elif cat_lower in {'shoes', 'sneakers', 'heels', 'boots', 'loafers'}:
            categories['shoes'].append(item)
        elif cat_lower in {'accessories', 'belt', 'watch', 'cap', 'hat', 'scarf', 'bag'}:
            categories['accessories'].append(item)
    
    return categories


def get_image_path(image_url: str, app_root: str = None) -> Optional[str]:
    """
    Convert image URL to filesystem path.
    
    Args:
        image_url: URL path (e.g., '/static/uploads/image.jpg' or 'static/uploads/image.jpg')
        app_root: Application root directory
        
    Returns:
        Full filesystem path or None if invalid
    """
    if not image_url:
        return None
    
    # Remove leading slash if present and handle Flask static URLs
    path = image_url.lstrip('/')
    
    # Extract filename from URL if it contains 'uploads/'
    if 'uploads/' in path:
        filename = path.split('uploads/')[-1].split('?')[0]  # Remove query params if any
    elif path.startswith('static/uploads/'):
        filename = path.replace('static/uploads/', '')
    else:
        filename = os.path.basename(path)
    
    # Try multiple path resolutions
    possible_paths = []
    
    # Path 1: Direct path from app_root
    if app_root:
        possible_paths.append(os.path.join(app_root, path))
        possible_paths.append(os.path.join(app_root, 'static', 'uploads', filename))
    
    # Path 2: Current working directory
    cwd = os.getcwd()
    possible_paths.append(os.path.join(cwd, path))
    possible_paths.append(os.path.join(cwd, 'static', 'uploads', filename))
    
    # Path 3: Relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths.append(os.path.join(script_dir, path))
    possible_paths.append(os.path.join(script_dir, 'static', 'uploads', filename))
    
    # Try each path
    for full_path in possible_paths:
        full_path = os.path.normpath(full_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return full_path
    
    return None


def find_best_outfits(items: List[WardrobeItem], app_root: str = None, 
                     max_results: int = 5) -> List[Tuple[List[WardrobeItem], float]]:
    """
    Find the best matching outfits from wardrobe items using image similarity.
    
    Args:
        items: List of WardrobeItem objects with images
        app_root: Application root directory for resolving image paths
        max_results: Maximum number of outfit recommendations to return
        
    Returns:
        List of tuples: (outfit_items, score) sorted by score descending
    """
    # Categorize items
    categories = categorize_items(items)
    
    tops = categories['tops']
    bottoms = categories['bottoms']
    shoes = categories['shoes']
    accessories = categories['accessories']
    
    # Validate we have items in each category
    if not tops or not bottoms:
        return []  # Need at least tops and bottoms
    
    # Convert items to feature vectors
    top_data = []
    for item in tops:
        path = get_image_path(item.image_url, app_root)
        if path:
            vec = image_to_vector(path)
            if vec is not None:
                top_data.append((item, vec))
    
    bottom_data = []
    for item in bottoms:
        path = get_image_path(item.image_url, app_root)
        if path:
            vec = image_to_vector(path)
            if vec is not None:
                bottom_data.append((item, vec))
    
    shoe_data = []
    for item in shoes:
        path = get_image_path(item.image_url, app_root)
        if path:
            vec = image_to_vector(path)
            if vec is not None:
                shoe_data.append((item, vec))
    
    accessory_data = []
    for item in accessories:
        path = get_image_path(item.image_url, app_root)
        if path:
            vec = image_to_vector(path)
            if vec is not None:
                accessory_data.append((item, vec))
    
    if not top_data or not bottom_data:
        return []
    
    # Generate all possible combinations
    outfits = []
    
    # Handle cases where shoes or accessories might be missing
    # If no shoes/accessories, we'll create minimal outfits with just top and bottom
    has_shoes = isinstance(shoe_data, list) and len(shoe_data) > 0
    has_accessories = isinstance(accessory_data, list) and len(accessory_data) > 0
    
    for top_item, top_vec in top_data:
        for bottom_item, bottom_vec in bottom_data:
            if has_shoes and has_accessories:
                # Full outfit with all items
                for shoe_item, shoe_vec in shoe_data:
                    for acc_item, acc_vec in accessory_data:
                        score = outfit_score(top_vec, bottom_vec, shoe_vec, acc_vec)
                        outfit_items = [top_item, bottom_item, shoe_item, acc_item]
                        outfits.append((outfit_items, score))
            elif has_shoes:
                # Outfit with top, bottom, and shoes
                for shoe_item, shoe_vec in shoe_data:
                    # Create a neutral accessory vector (gray)
                    dummy_acc_vec = np.full_like(top_vec, 0.5)
                    score = outfit_score(top_vec, bottom_vec, shoe_vec, dummy_acc_vec)
                    outfit_items = [top_item, bottom_item, shoe_item]
                    outfits.append((outfit_items, score))
            elif has_accessories:
                # Outfit with top, bottom, and accessories
                for acc_item, acc_vec in accessory_data:
                    # Create a neutral shoe vector (gray)
                    dummy_shoe_vec = np.full_like(top_vec, 0.5)
                    score = outfit_score(top_vec, bottom_vec, dummy_shoe_vec, acc_vec)
                    outfit_items = [top_item, bottom_item, acc_item]
                    outfits.append((outfit_items, score))
            else:
                # Minimal outfit with just top and bottom
                # Create neutral vectors for missing items
                dummy_shoe_vec = np.full_like(top_vec, 0.5)
                dummy_acc_vec = np.full_like(top_vec, 0.5)
                score = outfit_score(top_vec, bottom_vec, dummy_shoe_vec, dummy_acc_vec)
                outfit_items = [top_item, bottom_item]
                outfits.append((outfit_items, score))
    
    # Sort by score descending and return top results
    if not outfits:
        return []
    
    try:
        outfits.sort(key=lambda x: x[1], reverse=True)
        return outfits[:max_results]
    except Exception as e:
        print(f"Error sorting outfits: {e}")
        return []


def get_outfit_recommendations(user_items: List[WardrobeItem], 
                              app_root: str = None,
                              max_results: int = 5) -> List[Dict]:
    """
    Get outfit recommendations with metadata.
    
    Args:
        user_items: List of WardrobeItem objects for the user
        app_root: Application root directory
        max_results: Maximum number of recommendations
        
    Returns:
        List of dictionaries with outfit data and score
    """
    try:
        # Call the function, don't reference it
        outfits_result = find_best_outfits(user_items, app_root, max_results)
        
        # Ensure outfits_result is a list
        if not isinstance(outfits_result, list):
            print(f"Warning: find_best_outfits returned {type(outfits_result)}, expected list. Value: {outfits_result}")
            return []
        
        # Check if it's empty
        if len(outfits_result) == 0:
            print("No outfits found by find_best_outfits")
            return []
        
        results = []
        for outfit_tuple in outfits_result:
            # Handle tuple format: (outfit_items, score)
            if isinstance(outfit_tuple, tuple) and len(outfit_tuple) == 2:
                outfit_items, score = outfit_tuple
                # Ensure outfit_items is a list
                if isinstance(outfit_items, list):
                    results.append({
                        'items': outfit_items,
                        'score': float(score) if isinstance(score, (int, float)) else 0.0,
                        'score_percentage': float(score * 100) if isinstance(score, (int, float)) else 0.0
                    })
                else:
                    print(f"Warning: outfit_items is not a list: {type(outfit_items)}")
        
        return results
    except Exception as e:
        print(f"Error in get_outfit_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

