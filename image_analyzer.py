"""
AI Image Analyzer for Clothing Items
Analyzes uploaded clothing images to extract:
- Category (type of clothing)
- Color (dominant colors)
- Pattern (striped, plain, floral, etc.)
- Material (estimated fabric type)
- Fit (slim, regular, oversized)
- Seasonality (summer, winter, all-season)
- Occasion (where to wear it)
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import Counter
import cv2
from sklearn.cluster import KMeans


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB."""
    if not hex_color:
        return (128, 128, 128)  # Default gray
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (128, 128, 128)  # Default gray if invalid
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return (128, 128, 128)  # Default gray on error


def extract_dominant_colors(image_path: str, n_colors: int = 3) -> List[Tuple[str, float]]:
    """
    Extract dominant colors from image using K-means clustering.
    Returns list of (hex_color, percentage) tuples.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        # Remove very dark and very light pixels (background noise)
        mask = (img.mean(axis=1) > 30) & (img.mean(axis=1) < 240)
        img = img[mask]
        
        if len(img) < n_colors:
            n_colors = max(1, len(img))
        
        if n_colors == 0:
            return [("#000000", 1.0)]
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(img)
        
        labels = kmeans.labels_
        colors = kmeans.cluster_centers_.astype(int)
        counts = Counter(labels)
        total = sum(counts.values())
        
        color_data = []
        for i, color in enumerate(colors):
            r, g, b = color
            hex_color = rgb_to_hex(r, g, b)
            percentage = counts[i] / total
            color_data.append((hex_color, percentage))
        
        return sorted(color_data, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Error extracting colors: {e}")
        return [("#808080", 1.0)]


def detect_pattern(image_path: str) -> str:
    """
    Detect pattern type in clothing image.
    Returns: 'plain', 'striped', 'floral', 'graphic', 'checked', 'polka-dot', 'abstract'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "plain"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Horizontal lines (stripes)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        horizontal_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        vertical_density = np.sum(vertical_lines > 0) / vertical_lines.size
        
        # Checked pattern (both horizontal and vertical)
        if horizontal_density > 0.01 and vertical_density > 0.01:
            return "checked"
        
        # Striped pattern
        if horizontal_density > 0.02:
            return "striped"
        if vertical_density > 0.02:
            return "striped"
        
        # Texture analysis for floral/abstract patterns
        # Using Laplacian variance to detect texture
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var > 500:  # High texture
            # Check for circular patterns (polka dots)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=5, maxRadius=30)
            if circles is not None and len(circles[0]) > 5:
                return "polka-dot"
            
            # High texture could be floral or abstract
            if edge_density > 0.15:
                return "floral"
            else:
                return "abstract"
        
        # Graphic patterns (high contrast areas)
        if edge_density > 0.1:
            return "graphic"
        
        # Default to plain
        return "plain"
    
    except Exception as e:
        print(f"Error detecting pattern: {e}")
        return "plain"


def classify_clothing_category(image_path: str) -> str:
    """
    Classify clothing category from image using shape and visual features.
    Returns: category string like 'shirt', 'pants', 'shoes', etc.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "other"
        
        h, w = img.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect contours to understand shape
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
            contour_ratio = w_cont / h_cont if h_cont > 0 else 1.0
            
            # Shape-based classification
            if contour_ratio > 2.5:  # Wide (likely pants, skirts, or shoes)
                # Check for footwear features
                if aspect_ratio > 1.3 and aspect_ratio < 2.5:
                    # Look for shoe-like shapes
                    return "shoes"
                elif h_cont > w_cont * 1.5:
                    return "pants"
                else:
                    return "skirt"
            
            elif contour_ratio < 0.6:  # Tall and narrow
                # Could be dress or long top
                if h_cont > w_cont * 2:
                    return "shirt"
                return "dress"
            
            elif 0.8 < contour_ratio < 1.5:  # Square-ish (tops, jackets)
                # Check if it's outerwear (usually thicker/darker at edges)
                edges_dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
                edge_pixels = np.sum(edges_dark > 0) / edges_dark.size
                
                if edge_pixels > 0.3:
                    return "jacket"
                else:
                    return "shirt"
        
        # Fallback: aspect ratio based
        if aspect_ratio > 2.0:
            return "pants"
        elif aspect_ratio < 0.7:
            return "shirt"
        else:
            return "shirt"
    
    except Exception as e:
        print(f"Error classifying category: {e}")
        return "shirt"


def estimate_material(image_path: str) -> str:
    """
    Estimate material type from image texture and appearance.
    Returns: 'cotton', 'denim', 'leather', 'silk', 'wool', 'synthetic'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "cotton"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Texture analysis using Local Binary Patterns (simplified)
        # Calculate standard deviation of pixel intensities (texture roughness)
        texture_variance = np.std(gray)
        
        # Color brightness analysis
        brightness = np.mean(img)
        
        # Edge sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Material classification based on features
        if brightness < 80:  # Dark, likely denim or leather
            if texture_variance > 40:
                return "denim"
            else:
                return "leather"
        elif brightness > 200:  # Very bright, could be silk or synthetic
            if sharpness < 300:
                return "silk"
            else:
                return "synthetic"
        elif texture_variance > 50:  # Rough texture
            return "wool"
        else:  # Medium texture, likely cotton
            return "cotton"
    
    except Exception as e:
        print(f"Error estimating material: {e}")
        return "cotton"


def estimate_fit(image_path: str) -> str:
    """
    Estimate fit type from image.
    Returns: 'slim', 'regular', 'oversized'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "regular"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the main clothing item contour
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate how much of the image the item occupies
            img_area = gray.shape[0] * gray.shape[1]
            item_area = w * h
            coverage = item_area / img_area if img_area > 0 else 0
            
            # Fit estimation based on coverage
            if coverage > 0.7:  # Item takes up most of the image
                return "oversized"
            elif coverage < 0.4:  # Item is small relative to image
                return "slim"
            else:
                return "regular"
        
        return "regular"
    
    except Exception as e:
        print(f"Error estimating fit: {e}")
        return "regular"


def determine_seasonality(image_path: str, colors: List[Tuple[str, float]]) -> str:
    """
    Determine seasonality based on colors and image characteristics.
    Returns: 'summer', 'winter', 'all-season', 'spring', 'fall'
    """
    try:
        # Analyze dominant color brightness
        if not colors:
            return "all-season"
        
        main_color_hex = colors[0][0]
        r, g, b = hex_to_rgb(main_color_hex)
        brightness = (r + g + b) / 3
        
        # Summer: bright, light colors
        if brightness > 180:
            return "summer"
        
        # Winter: dark, muted colors
        if brightness < 100:
            return "winter"
        
        # Spring/Fall: medium brightness
        if 120 < brightness < 160:
            return "spring"
        
        return "all-season"
    
    except Exception as e:
        print(f"Error determining seasonality: {e}")
        return "all-season"


def suggest_occasion(category: str, pattern: str, material: str, colors: List[Tuple[str, float]]) -> str:
    """
    Suggest occasion tags based on clothing attributes.
    Returns: comma-separated occasion tags
    """
    occasions = []
    
    # Category-based occasions
    formal_categories = ['shirt', 'blazer', 'blouse', 'dress']
    casual_categories = ['tshirt', 'hoodie', 'sweater', 'jeans']
    sporty_categories = ['sneakers', 'shorts']
    
    if category.lower() in formal_categories:
        occasions.append("work")
        occasions.append("formal")
    elif category.lower() in casual_categories:
        occasions.append("casual")
        occasions.append("everyday")
    elif category.lower() in sporty_categories:
        occasions.append("gym")
        occasions.append("sport")
    
    # Pattern-based
    if pattern == "plain":
        occasions.append("office")
    if pattern in ["striped", "checked"]:
        occasions.append("smart-casual")
    if pattern == "floral":
        occasions.append("party")
        occasions.append("casual")
    
    # Material-based
    if material == "leather":
        occasions.append("party")
        occasions.append("formal")
    elif material == "denim":
        occasions.append("casual")
        occasions.append("weekend")
    elif material == "silk":
        occasions.append("formal")
        occasions.append("party")
    
    # Color-based (dark = formal, bright = casual)
    if colors:
        r, g, b = hex_to_rgb(colors[0][0])
        brightness = (r + g + b) / 3
        
        if brightness < 100:
            occasions.append("formal")
        elif brightness > 200:
            occasions.append("casual")
            occasions.append("party")
    
    # Remove duplicates and return
    unique_occasions = list(set(occasions))
    return ",".join(unique_occasions[:5])  # Limit to 5 occasions


def get_detailed_occasion_info(category: str, pattern: str, material: str, colors: List[Tuple[str, float]]) -> Dict[str, str]:
    """
    Get detailed information about where to wear the clothing and styling tips.
    Returns dictionary with occasion details and styling advice.
    """
    occasions_info = {}
    
    # Category-based styling
    category_styling = {
        'shirt': {
            'formal': 'Perfect for office, business meetings, and formal events. Pair with dress pants or a skirt.',
            'casual': 'Great for everyday wear. Looks good with jeans or casual pants.',
            'smart-casual': 'Versatile for office casual or weekend brunches. Style with chinos or dark jeans.'
        },
        'pants': {
            'formal': 'Ideal for professional settings. Pair with a blazer or formal shirt.',
            'casual': 'Perfect for relaxed outings. Great with t-shirts or casual tops.',
            'smart-casual': 'Works well for casual Fridays or weekend events.'
        },
        'jeans': {
            'casual': 'Classic casual wear. Perfect for weekends, shopping, or casual hangouts.',
            'everyday': 'Versatile everyday option. Style with any top from t-shirts to blouses.'
        },
        'shoes': {
            'formal': 'Appropriate for office, formal events, and business settings.',
            'casual': 'Perfect for everyday wear, walking, and casual outings.',
            'party': 'Great for social events, parties, and nightlife.'
        },
        'jacket': {
            'formal': 'Ideal for business meetings, formal dinners, and professional settings.',
            'casual': 'Perfect for layering in cool weather. Great for casual outings.',
            'party': 'Stylish outerwear for evening events and parties.'
        }
    }
    
    # Material-based styling tips
    material_tips = {
        'cotton': 'Breathable and comfortable. Great for warm weather and everyday wear.',
        'denim': 'Durable and versatile. Perfect for casual and smart-casual occasions.',
        'leather': 'Stylish and edgy. Works well for parties, casual events, and cooler weather.',
        'silk': 'Elegant and luxurious. Perfect for formal events, parties, and special occasions.',
        'wool': 'Warm and cozy. Ideal for winter and formal settings.',
        'synthetic': 'Easy to care for. Suitable for active wear and casual occasions.'
    }
    
    # Pattern-based styling
    pattern_tips = {
        'plain': 'Classic and versatile. Easy to pair with patterned or colorful items.',
        'striped': 'Timeless pattern. Pairs well with solid colors.',
        'floral': 'Feminine and playful. Great for casual outings, parties, and spring/summer events.',
        'checked': 'Traditional pattern. Works well for smart-casual and office settings.',
        'polka-dot': 'Playful and retro. Perfect for casual and party occasions.',
        'graphic': 'Modern and bold. Great for casual and streetwear styles.'
    }
    
    return {
        'category_tips': category_styling.get(category.lower(), {}),
        'material_tips': material_tips.get(material.lower(), 'Comfortable and versatile for various occasions.'),
        'pattern_tips': pattern_tips.get(pattern.lower(), 'Versatile pattern suitable for many occasions.')
    }


def get_style_description(category: str, pattern: str, material: str, fit: str, colors: List[Tuple[str, float]]) -> str:
    """
    Generate a detailed style description of the clothing item.
    """
    color_desc = colors[0][0] if colors else "neutral"
    color_name = get_color_name(colors[0][0]) if colors else "neutral"
    
    description_parts = []
    
    # Color description
    if colors:
        description_parts.append(f"This {color_name.lower()} {category}")
    else:
        description_parts.append(f"This {category}")
    
    # Pattern
    if pattern and pattern != "plain":
        description_parts.append(f"features a {pattern} pattern")
    
    # Material
    if material:
        description_parts.append(f"made from {material}")
    
    # Fit
    if fit:
        description_parts.append(f"with a {fit} fit")
    
    return " ".join(description_parts) + "."


def get_color_name(hex_color: str) -> str:
    """Convert hex color to approximate color name."""
    if not hex_color:
        return "neutral"
    
    try:
        r, g, b = hex_to_rgb(hex_color)
        
        # Simple color name mapping
        if r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        elif r > g and r > b and r > 150:
            return "Red"
        elif g > r and g > b and g > 150:
            return "Green"
        elif b > r and b > g and b > 150:
            return "Blue"
        elif r > 150 and g > 150 and b < 100:
            return "Yellow"
        elif r > 150 and b > 150 and g < 100:
            return "Magenta"
        elif r > 150 and g > 100 and b < 100:
            return "Orange"
        elif 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
            return "Gray"
        elif r > 150 and g > 100 and b > 100:
            return "Beige"
        else:
            return "Colored"
    except:
        return "Colored"


def analyze_clothing_image(image_path: str) -> Dict[str, any]:
    """
    Main function to analyze a clothing image and extract all attributes with detailed information.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with extracted attributes:
        - category: str
        - color_hex: str (dominant color)
        - pattern: str
        - material: str
        - fit: str
        - seasonality: str
        - occasion_tags: str
        - confidence: float (0-1, overall confidence score)
        - colors: List[Tuple[str, float]] (top colors with percentages)
        - detailed_info: Dict with styling tips and descriptions
        - style_description: str (human-readable description)
        - where_to_wear: List[str] (detailed occasion descriptions)
        - styling_tips: List[str] (styling advice)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"[IMAGE_ANALYZER] Analyzing image: {image_path}")
    
    # Extract all attributes
    try:
        colors = extract_dominant_colors(image_path, n_colors=3)
        print(f"[IMAGE_ANALYZER] Colors detected: {colors}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Color extraction error: {e}")
        colors = [("#808080", 1.0)]
    
    try:
        pattern = detect_pattern(image_path)
        print(f"[IMAGE_ANALYZER] Pattern detected: {pattern}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Pattern detection error: {e}")
        pattern = "plain"
    
    try:
        category = classify_clothing_category(image_path)
        print(f"[IMAGE_ANALYZER] Category detected: {category}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Category classification error: {e}")
        category = "shirt"
    
    try:
        material = estimate_material(image_path)
        print(f"[IMAGE_ANALYZER] Material detected: {material}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Material estimation error: {e}")
        material = "cotton"
    
    try:
        fit = estimate_fit(image_path)
        print(f"[IMAGE_ANALYZER] Fit detected: {fit}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Fit estimation error: {e}")
        fit = "regular"
    
    try:
        seasonality = determine_seasonality(image_path, colors)
        print(f"[IMAGE_ANALYZER] Seasonality detected: {seasonality}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Seasonality determination error: {e}")
        seasonality = "all-season"
    
    try:
        occasion_tags = suggest_occasion(category, pattern, material, colors)
        print(f"[IMAGE_ANALYZER] Occasions detected: {occasion_tags}")
    except Exception as e:
        print(f"[IMAGE_ANALYZER] Occasion suggestion error: {e}")
        occasion_tags = "casual,everyday"
    
    # Get dominant color
    color_hex = colors[0][0] if colors else "#808080"
    
    # Generate detailed information
    detailed_info = get_detailed_occasion_info(category, pattern, material, colors)
    style_description = get_style_description(category, pattern, material, fit, colors)
    
    # Where to wear - detailed descriptions
    where_to_wear = []
    occasion_list = [o.strip() for o in occasion_tags.split(',') if o.strip()]
    
    for occ in occasion_list:
        if occ == "work" or occ == "office" or occ == "formal":
            where_to_wear.append("🏢 Work & Office: Perfect for professional settings, business meetings, and office environments. Pair with dress pants or a skirt for a polished look.")
        elif occ == "party":
            where_to_wear.append("🎉 Parties & Events: Great for social gatherings, parties, and evening events. Style with bold accessories for extra flair.")
        elif occ == "casual" or occ == "everyday":
            where_to_wear.append("👕 Casual & Everyday: Ideal for daily wear, shopping, coffee dates, and relaxed outings. Versatile and comfortable.")
        elif occ == "gym" or occ == "sport":
            where_to_wear.append("🏃 Gym & Sports: Perfect for workouts, running, and active sports activities. Designed for movement and comfort.")
        elif occ == "smart-casual":
            where_to_wear.append("🎯 Smart Casual: Works well for casual Fridays, brunches, and weekend events. Strikes the perfect balance between formal and casual.")
        elif occ == "weekend":
            where_to_wear.append("🌅 Weekend: Great for relaxed weekend activities, outings with friends, and leisure time.")
    
    # Styling tips
    styling_tips = []
    styling_tips.append(f"Material: {detailed_info.get('material_tips', '')}")
    styling_tips.append(f"Pattern: {detailed_info.get('pattern_tips', '')}")
    
    if fit:
        if fit == "slim":
            styling_tips.append("Fit: Slim fit creates a modern, tailored look. Great for layering and adding structure to your outfit.")
        elif fit == "oversized":
            styling_tips.append("Fit: Oversized fit offers comfort and a relaxed, trendy aesthetic. Perfect for casual and streetwear styles.")
        else:
            styling_tips.append("Fit: Regular fit provides comfort and versatility. Easy to style for various occasions.")
    
    if seasonality:
        if seasonality == "summer":
            styling_tips.append("Season: Perfect for warm weather. Light and breathable, ideal for summer activities.")
        elif seasonality == "winter":
            styling_tips.append("Season: Great for cold weather. Provides warmth and protection from winter elements.")
        elif seasonality == "spring" or seasonality == "fall":
            styling_tips.append(f"Season: Ideal for {seasonality} weather. Transitional piece perfect for moderate temperatures.")
        else:
            styling_tips.append("Season: Versatile all-season piece. Can be worn year-round with appropriate layering.")
    
    # Calculate confidence
    confidence = 0.7  # Base confidence
    if colors:
        confidence += 0.1
    if pattern != "plain":
        confidence += 0.05
    if category != "other":
        confidence += 0.05
    if occasion_tags:
        confidence += 0.1
    confidence = min(1.0, confidence)
    
    result = {
        "category": category,
        "color_hex": color_hex,
        "pattern": pattern,
        "material": material,
        "fit": fit,
        "seasonality": seasonality,
        "occasion_tags": occasion_tags,
        "confidence": confidence,
        "colors": colors,
        "detailed_info": detailed_info,
        "style_description": style_description,
        "where_to_wear": where_to_wear if where_to_wear else ["Perfect for various casual and everyday occasions."],
        "styling_tips": styling_tips
    }
    
    print(f"[IMAGE_ANALYZER] Analysis complete. Confidence: {confidence:.0%}")
    return result

