from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple color utilities and rule-based outfit generator. Replace/extend with ML later.

@dataclass
class Item:
    id: int
    category: str
    color_hex: Optional[str] = None
    pattern: Optional[str] = None
    material: Optional[str] = None
    fit: Optional[str] = None
    seasonality: Optional[str] = None
    occasions: Optional[List[str]] = None


def hex_to_rgb(hex_color: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not hex_color:
        return None
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # type: ignore
    except Exception:
        return None


def color_distance(c1: Optional[Tuple[int, int, int]], c2: Optional[Tuple[int, int, int]]) -> float:
    if not c1 or not c2:
        return 128.0
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def is_color_compatible(top: Item, bottom: Item) -> bool:
    # Very light rule: avoid very close or very clashing combinations
    r1, r2 = hex_to_rgb(top.color_hex), hex_to_rgb(bottom.color_hex)
    dist = color_distance(r1, r2)
    return 40 <= dist <= 360


def filter_items_by_context(items: List[Item], context: Dict, user_prefs: Dict) -> List[Item]:
    """Filter items based on weather, activity, occasion, and other context.
    Properly filters out inappropriate items while keeping good matches."""
    filtered = []
    
    weather = context.get('weather', '').lower()
    activity = context.get('activity', '').lower()
    occasion = context.get('occasion', '').lower()
    color_theme = context.get('color_theme') or user_prefs.get('color_theme', '').lower()
    
    for item in items:
        score = 1.0  # Start with full score
        exclude = False  # Flag to exclude completely inappropriate items
        
        # STRICT Weather filtering - exclude inappropriate items
        if weather:
            item_season = (item.seasonality or '').lower()
            item_material = (item.material or '').lower()
            
            if weather in ['hot', 'warm']:
                # Exclude heavy winter items in hot weather
                if item_season == 'winter':
                    # Heavy materials in hot weather are inappropriate
                    if item_material in ['wool', 'fleece', 'thick', 'heavy']:
                        exclude = True
                    else:
                        score *= 0.3  # Heavily penalize but allow light winter items
                elif item_season in ['summer', 'all-season']:
                    score *= 1.5  # Strong boost for summer items
                # Prefer light materials
                if item_material in ['cotton', 'linen', 'synthetic', 'light']:
                    score *= 1.3
                elif item_material in ['wool', 'fleece', 'thick', 'heavy']:
                    score *= 0.4  # Penalize heavy materials
                    
            elif weather in ['cold', 'cool']:
                # Exclude summer-only items in cold weather
                if item_season == 'summer':
                    # Light summer materials in cold weather are inappropriate
                    if item_material in ['linen', 'thin', 'light'] and 'jacket' not in item.category.lower() and 'coat' not in item.category.lower():
                        exclude = True
                    else:
                        score *= 0.3  # Heavily penalize
                elif item_season in ['winter', 'all-season']:
                    score *= 1.5  # Strong boost for winter items
                # Prefer warmer materials
                if item_material in ['wool', 'fleece', 'thick', 'cotton']:
                    score *= 1.3
                elif item_material in ['linen', 'thin', 'light']:
                    score *= 0.5  # Penalize light materials
                    
            elif weather == 'rainy':
                # Prefer water-resistant materials
                if item_material in ['synthetic', 'nylon', 'polyester']:
                    score *= 1.4
                elif item_material in ['silk', 'wool']:
                    score *= 0.6  # Penalize delicate materials
        
        # Activity level filtering
        if activity and not exclude:
            if activity in ['high', 'moderate']:
                # Prefer comfortable, flexible materials
                if item_material in ['cotton', 'synthetic', 'stretch']:
                    score *= 1.3
                if item.fit and item.fit.lower() in ['regular', 'slim', 'athletic']:
                    score *= 1.2
                elif item.fit and item.fit.lower() == 'oversized':
                    score *= 0.7  # Less ideal for active movement
            elif activity == 'sedentary':
                # More flexible, any material is fine
                score *= 1.0
        
        # Occasion filtering - boost matching items
        if occasion and not exclude:
            item_occasions = [o.lower() for o in (item.occasions or [])]
            if occasion in item_occasions:
                score *= 1.5  # Strong boost if occasion matches
            elif occasion in ['formal', 'work', 'office']:
                # For formal, exclude casual items
                if item.category.lower() in ['tshirt', 'hoodie', 'sweatpants']:
                    score *= 0.4  # Heavily penalize casual items
                if item.pattern and item.pattern.lower() not in ['plain', 'striped', 'checked']:
                    score *= 0.6  # Reduce bold patterns
            elif occasion in ['party', 'casual', 'date']:
                # Boost patterns and casual items
                if item.pattern and item.pattern.lower() in ['floral', 'graphic', 'printed']:
                    score *= 1.3
        
        # Color theme filtering
        if color_theme and not exclude:
            item_color = hex_to_rgb(item.color_hex)
            if item_color:
                r, g, b = item_color
                brightness = (r + g + b) / 3
                
                if color_theme == 'neutral':
                    if 100 < brightness < 200 and abs(r-g) < 30 and abs(g-b) < 30:
                        score *= 1.3
                    elif brightness > 230 or brightness < 50:
                        score *= 1.2
                    else:
                        score *= 0.7  # Reduce colorful items
                elif color_theme == 'bold':
                    if brightness > 150 and (r > 180 or g > 180 or b > 180):
                        score *= 1.3
                    else:
                        score *= 0.8
                elif color_theme == 'pastel':
                    if 150 < brightness < 220:
                        score *= 1.2
                elif color_theme == 'dark':
                    if brightness < 120:
                        score *= 1.3
                    else:
                        score *= 0.7
        
        # Only include items that are appropriate (not excluded and have reasonable score)
        if not exclude and score >= 0.5:  # Stricter threshold
            filtered.append(item)
    
    # If filtering removed everything, return original list (fallback)
    return filtered if filtered else items


def category_priority(cat: str) -> int:
    order = {
        'shirt': 1, 'tshirt': 1, 'top': 1,
        'pants': 2, 'trousers': 2, 'jeans': 2, 'skirt': 2,
        'jacket': 3, 'coat': 3, 'blazer': 3,
        'shoes': 4, 'sneakers': 4, 'heels': 4,
        'accessories': 5, 'belt': 5, 'watch': 5
    }
    return order.get(cat.lower(), 99)


def compose_outfits(items: List[Item], context: Dict) -> List[List[Item]]:
    """Enhanced outfit composition with better context awareness."""
    occasion = (context.get('occasion') or '').lower().strip()
    weather = (context.get('weather') or '').lower().strip()
    activity = (context.get('activity') or '').lower().strip()
    time_of_day = (context.get('time') or '').lower().strip()

    tops = [i for i in items if i.category.lower() in {'shirt', 'tshirt', 'top', 'blouse', 'sweater', 'hoodie'}]
    bottoms = [i for i in items if i.category.lower() in {'pants', 'trousers', 'jeans', 'skirt', 'shorts'}]
    outers = [i for i in items if i.category.lower() in {'jacket', 'coat', 'blazer', 'cardigan'}]
    shoes = [i for i in items if i.category.lower() in {'shoes', 'sneakers', 'heels', 'boots', 'loafers'}]
    accs = [i for i in items if i.category.lower() in {'accessories', 'belt', 'watch', 'cap', 'hat'}]

    results: List[List[Item]] = []
    
    # Enhanced logic for outerwear based on weather and activity
    needs_outerwear = (
        weather in {'cold', 'cool', 'rainy'} or 
        occasion in {'office', 'wedding', 'formal', 'date'} or
        time_of_day in {'evening', 'night'}
    )
    
    # Enhanced logic for shoes based on occasion and activity
    preferred_shoe_types = []
    if occasion in {'formal', 'wedding', 'office'}:
        preferred_shoe_types = ['heels', 'loafers', 'shoes']
    elif occasion in {'sports', 'gym'} or activity in {'high', 'moderate'}:
        preferred_shoe_types = ['sneakers', 'shoes']
    elif occasion in {'party', 'date', 'casual'}:
        preferred_shoe_types = ['heels', 'sneakers', 'shoes']
    else:
        preferred_shoe_types = ['shoes', 'sneakers']
    
    # Filter shoes by preference
    preferred_shoes = [s for s in shoes if any(pref in s.category.lower() for pref in preferred_shoe_types)]
    if not preferred_shoes:
        preferred_shoes = shoes  # Fallback to all shoes
    
    # Ensure we use ALL tops and ALL bottoms for maximum variety
    if not tops or not bottoms:
        return []  # Need at least one top and one bottom
    
    # Limit combinations to prevent too many results, but ensure variety
    # Use all items but limit per top-bottom pair to avoid explosion
    max_shoes_per_combo = min(5, len(shoes)) if shoes else 0
    max_outers_per_combo = min(3, len(outers)) if outers else 0
    max_accs_per_combo = min(3, len(accs)) if accs else 0
    
    for t in tops:  # Use ALL tops
        for b in bottoms:  # Use ALL bottoms - this ensures bottom wear changes
            if not t or not b:  # Need at least top and bottom
                continue
                
            if not is_color_compatible(t, b):
                continue
                
            # Try ALL shoe combinations but limit per combo to ensure variety across different bottoms
            shoes_to_try = preferred_shoes if preferred_shoes else ([None] if not shoes else shoes)
            # Use all shoes but rotate through them to ensure different bottoms get different shoes
            for idx, s in enumerate(shoes_to_try[:max_shoes_per_combo] if max_shoes_per_combo > 0 else shoes_to_try):
                combo = [t, b]
                if s:
                    combo.append(s)
                
                # Add outerwear - try multiple outerwear options for variety
                if outers:
                    # Try each outerwear option to create more combinations
                    # Rotate through outerwear to ensure different bottoms get different outerwear
                    for outer in outers[:max_outers_per_combo] if max_outers_per_combo > 0 else outers:
                        combo_with_outer = combo + [outer]
                        
                        # Add accessories with this outerwear combination - limit to ensure variety
                        if accs:
                            accs_to_use = accs[:max_accs_per_combo] if max_accs_per_combo > 0 else accs
                            if occasion in {'formal', 'office'}:
                                formal_accs = [a for a in accs_to_use if a.category.lower() in {'belt', 'watch'}]
                                if formal_accs:
                                    for acc in formal_accs:
                                        results.append(combo_with_outer + [acc])
                                else:
                                    for acc in accs_to_use:
                                        results.append(combo_with_outer + [acc])
                            else:
                                for acc in accs_to_use:
                                    results.append(combo_with_outer + [acc])
                        else:
                            results.append(combo_with_outer)
                    
                    # Also add combo without outerwear for variety
                    if accs:
                        accs_to_use = accs[:max_accs_per_combo] if max_accs_per_combo > 0 else accs
                        if occasion in {'formal', 'office'}:
                            formal_accs = [a for a in accs_to_use if a.category.lower() in {'belt', 'watch'}]
                            if formal_accs:
                                for acc in formal_accs:
                                    results.append(combo + [acc])
                            else:
                                for acc in accs_to_use:
                                    results.append(combo + [acc])
                        else:
                            for acc in accs_to_use:
                                results.append(combo + [acc])
                    else:
                        results.append(combo)
                else:
                    # No outerwear available, proceed with accessories
                    if accs:
                        accs_to_use = accs[:max_accs_per_combo] if max_accs_per_combo > 0 else accs
                        if occasion in {'formal', 'office'}:
                            formal_accs = [a for a in accs_to_use if a.category.lower() in {'belt', 'watch'}]
                            if formal_accs:
                                for acc in formal_accs:
                                    results.append(combo + [acc])
                            else:
                                for acc in accs_to_use:
                                    results.append(combo + [acc])
                        else:
                            for acc in accs_to_use:
                                results.append(combo + [acc])
                    else:
                        results.append(combo)
                
    
    # Enhanced sorting: consider multiple factors with STRICT weather matching
    def outfit_quality_score(outfit: List[Item]) -> float:
        score = 0.0
        
        # Category completeness (top, bottom, shoes) - REQUIRED
        has_top = any(i.category.lower() in {'shirt', 'tshirt', 'top', 'blouse', 'sweater', 'hoodie'} for i in outfit)
        has_bottom = any(i.category.lower() in {'pants', 'trousers', 'jeans', 'skirt', 'shorts'} for i in outfit)
        has_shoes = any(i.category.lower() in {'shoes', 'sneakers', 'heels', 'boots', 'loafers'} for i in outfit)
        
        if has_top and has_bottom:
            score += 15.0  # Base requirement
        else:
            return 0.0  # Must have top and bottom
        
        if has_shoes:
            score += 8.0  # Strongly prefer shoes
        
        if len(outfit) >= 3:
            score += 3.0  # Prefer complete outfits
        
        # STRICT Weather matching - heavily penalize mismatches
        weather_match_score = 0.0
        weather_penalty = 0.0
        for item in outfit:
            item_season = (item.seasonality or '').lower()
            item_material = (item.material or '').lower()
            
            if weather:
                if weather in {'hot', 'warm'}:
                    if item_season in {'summer', 'all-season'}:
                        weather_match_score += 3.0  # Strong match
                    elif item_season == 'winter':
                        if item_material in ['wool', 'fleece', 'thick', 'heavy']:
                            weather_penalty += 5.0  # Heavy penalty for heavy winter items
                        else:
                            weather_penalty += 2.0  # Penalty for winter items
                    # Material check
                    if item_material in ['cotton', 'linen', 'synthetic', 'light']:
                        weather_match_score += 1.5
                    elif item_material in ['wool', 'fleece', 'thick', 'heavy']:
                        weather_penalty += 3.0
                        
                elif weather in {'cold', 'cool'}:
                    if item_season in {'winter', 'all-season'}:
                        weather_match_score += 3.0  # Strong match
                    elif item_season == 'summer':
                        if item_material in ['linen', 'thin', 'light']:
                            weather_penalty += 4.0  # Heavy penalty for light summer items
                        else:
                            weather_penalty += 2.0
                    # Material check
                    if item_material in ['wool', 'fleece', 'thick', 'cotton']:
                        weather_match_score += 1.5
                    elif item_material in ['linen', 'thin', 'light']:
                        weather_penalty += 2.5
                        
                elif weather == 'rainy':
                    if item_material in ['synthetic', 'nylon', 'polyester']:
                        weather_match_score += 2.0
                    elif item_material in ['silk', 'wool']:
                        weather_penalty += 2.0
        
        score += weather_match_score
        score -= weather_penalty  # Subtract penalties
        
        # Color harmony - improved check
        colors = [hex_to_rgb(i.color_hex) for i in outfit if i.color_hex]
        if len(colors) >= 2:
            # Check color compatibility
            color_harmony = 0.0
            for i in range(len(colors)):
                for j in range(i+1, len(colors)):
                    if colors[i] and colors[j]:
                        dist = color_distance(colors[i], colors[j])
                        if 40 <= dist <= 200:  # Good color distance
                            color_harmony += 2.0
                        elif dist < 40:  # Too similar
                            color_harmony -= 1.0
                        elif dist > 300:  # Too clashing
                            color_harmony -= 0.5
            score += color_harmony / max(1, len(colors) - 1)
        
        # Material consistency (prefer similar materials for formality)
        materials = [i.material for i in outfit if i.material]
        if materials:
            unique_materials = len(set(m.lower() for m in materials))
            if unique_materials <= 2:
                score += 2.0  # Bonus for material consistency
            elif unique_materials == 3:
                score += 1.0
            # No penalty for more variety
        
        # Occasion matching - strong boost
        occasion_match_count = 0
        for item in outfit:
            if item.occasions and occasion:
                if any(occ in (o.lower() for o in item.occasions) for occ in [occasion]):
                    occasion_match_count += 1
                    score += 3.0  # Strong boost per matching item
        
        # Style consistency
        patterns = [i.pattern for i in outfit if i.pattern]
        if patterns:
            # Prefer consistent pattern style
            if len(set(p.lower() for p in patterns)) <= 2:
                score += 1.5
        
        return max(0.0, score)  # Ensure non-negative
    
    # Remove duplicates (same items in same order)
    seen = set()
    unique_results = []
    for outfit in results:
        outfit_ids = tuple(sorted([item.id for item in outfit]))
        if outfit_ids not in seen:
            seen.add(outfit_ids)
            unique_results.append(outfit)
    
    # Sort by quality score
    unique_results.sort(key=outfit_quality_score, reverse=True)
    
    # Filter by quality - STRICTER threshold for better recommendations
    # Only return outfits with good quality scores
    filtered = [o for o in unique_results if outfit_quality_score(o) >= 15.0]
    
    # If we have good quality outfits, return top 15
    # If not enough good ones, lower threshold slightly but still maintain quality
    if len(filtered) < 5:
        filtered = [o for o in unique_results if outfit_quality_score(o) >= 12.0]
    
    # Return top quality outfits (up to 15 for variety but maintain quality)
    return filtered[:15] if len(filtered) > 15 else filtered


def recommend(items: List[Item], user_prefs: Dict, context: Dict) -> List[List[Item]]:
    """Entry point for recommendations. Enhanced with image analysis data and more context.
    user_prefs: {"style": str, "favorite_colors": ["#aabbcc"], "color_theme": str, ...}
    context: {"occasion": str, "weather": str, "time": str, "activity": str, "style": str, "color_theme": str}
    """
    # Step 1: Filter items based on context before composing
    filtered_items = filter_items_by_context(items, context, user_prefs)
    
    # Step 2: Compose outfits using enhanced rules
    outfits = compose_outfits(filtered_items, context)

    if not outfits:
        return outfits

    # Step 2: Build text features for items (category, pattern, material, fit, seasonality, occasions)
    def item_text(i: Item) -> str:
        parts = [
            (i.category or ''),
            (i.pattern or ''),
            (i.material or ''),
            (i.fit or ''),
            (i.seasonality or ''),
            ' '.join(i.occasions or []),
        ]
        return ' '.join([p.lower() for p in parts if p])

    # Map items to consistent index
    all_items = list({it.id: it for outfit in outfits for it in outfit}.values())
    texts = [item_text(i) for i in all_items]
    if not any(texts):
        # If no textual features, just return rule-based
        return outfits

    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(X)

    # Color utility for similarity against favorite colors
    fav_colors = [c.strip() for c in (user_prefs.get('favorite_colors') or []) if c.strip()]
    fav_rgbs = [hex_to_rgb(c) for c in fav_colors]
    fav_rgbs = [c for c in fav_rgbs if c]

    def color_similarity(i: Item, j: Item) -> float:
        c1, c2 = hex_to_rgb(i.color_hex), hex_to_rgb(j.color_hex)
        if not c1 or not c2:
            return 0.5
        dist = color_distance(c1, c2)
        # Normalize distance to similarity in [0,1]
        return max(0.0, 1.0 - min(dist, 360.0) / 360.0)

    def favorite_color_affinity(i: Item) -> float:
        if not fav_rgbs:
            return 0.5
        c = hex_to_rgb(i.color_hex)
        if not c:
            return 0.5
        dists = [color_distance(c, f) for f in fav_rgbs]
        best = min(dists) if dists else 180.0
        return max(0.0, 1.0 - min(best, 360.0) / 360.0)

    # Index map for sim lookup
    idx = {it.id: k for k, it in enumerate(all_items)}

    def outfit_score(outfit: List[Item]) -> float:
        # Average pairwise content similarity + color similarity + affinity to favorite colors
        if not outfit:
            return 0.0
        pair_scores = []
        color_scores = []
        for a_idx in range(len(outfit)):
            for b_idx in range(a_idx + 1, len(outfit)):
                ia, ib = outfit[a_idx], outfit[b_idx]
                ia_i, ib_i = idx.get(ia.id), idx.get(ib.id)
                if ia_i is not None and ib_i is not None:
                    pair_scores.append(sim_matrix[ia_i, ib_i])
                color_scores.append(color_similarity(ia, ib))
        pair_sim = float(np.mean(pair_scores)) if pair_scores else 0.0
        color_sim = float(np.mean(color_scores)) if color_scores else 0.5
        fav_aff = float(np.mean([favorite_color_affinity(it) for it in outfit]))
        # Weighted sum
        return 0.55 * pair_sim + 0.30 * color_sim + 0.15 * fav_aff

    outfits_sorted = sorted(outfits, key=outfit_score, reverse=True)
    return outfits_sorted[:10]
