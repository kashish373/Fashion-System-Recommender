import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.utils import secure_filename

from extensions import db, login_manager
from models import User, WardrobeItem
from recommender import Item, recommend
from ml_model import load_model, score_outfit, SimpleItem
from image_analyzer import analyze_clothing_image
from outfit_matcher import get_outfit_recommendations


def create_app():
    app = Flask(__name__)
    # Use a stable dev key unless SECRET_KEY is defined in env to avoid session resets on reload
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fashion.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    db.init_app(app)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    with app.app_context():
        db.create_all()
        # ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dev/seed')
    @login_required
    def dev_seed():
        """Seed some sample wardrobe items for the current user (dev helper)."""
        samples = [
            {"category": "shirt", "color_hex": "#1E90FF", "pattern": "striped", "material": "cotton", "fit": "slim", "seasonality": "summer", "occasion_tags": "office,meeting", "brand": "Uniqlo", "price": 29.99, "size": "M"},
            {"category": "jeans", "color_hex": "#2B2B2B", "pattern": "plain", "material": "denim", "fit": "regular", "seasonality": "all-season", "occasion_tags": "casual", "brand": "Levis", "price": 49.99, "size": "32"},
            {"category": "shoes", "color_hex": "#000000", "pattern": "plain", "material": "leather", "fit": "regular", "seasonality": "all-season", "occasion_tags": "formal", "brand": "Clarks", "price": 89.99, "size": "9"},
            {"category": "jacket", "color_hex": "#000000", "pattern": "plain", "material": "leather", "fit": "regular", "seasonality": "winter", "occasion_tags": "formal", "brand": "Zara", "price": 119.0, "size": "M"},
            {"category": "belt", "color_hex": "#8B4513", "pattern": "plain", "material": "leather", "fit": "regular", "seasonality": "all-season", "occasion_tags": "office", "brand": "H&M", "price": 15.5, "size": "M"},
        ]
        for data in samples:
            item = WardrobeItem(user_id=current_user.id, **data)
            db.session.add(item)
        db.session.commit()
        flash('Seeded sample wardrobe items.', 'success')
        return redirect(url_for('wardrobe'))

    @app.route('/dev/import')
    @login_required
    def dev_import_from_csv():
        """Import wardrobe items for the logged-in user from a CSV located in the project folder.
        Usage: /dev/import?file=filtered_data_updated.csv&limit=100
        Recognized columns: articleType, subCategory, baseColour, season, usage, productDisplayName, gender.
        """
        import pandas as pd

        file_name = request.args.get('file', 'filtered_data_updated.csv')
        try:
            limit = int(request.args.get('limit', '50'))
        except ValueError:
            limit = 50

        csv_path = os.path.join(os.getcwd(), file_name)
        if not os.path.exists(csv_path):
            flash(f'CSV file not found: {file_name}', 'danger')
            return redirect(url_for('wardrobe'))

        # Minimal color name to hex mapping
        color_map = {
            'black': '#000000', 'white': '#FFFFFF', 'blue': '#0000FF', 'navy blue': '#000080', 'red': '#FF0000',
            'green': '#008000', 'yellow': '#FFFF00', 'grey': '#808080', 'gray': '#808080', 'purple': '#800080',
            'pink': '#FFC0CB', 'brown': '#8B4513', 'khaki': '#F0E68C', 'beige': '#F5F5DC', 'silver': '#C0C0C0',
            'gold': '#FFD700', 'white & navy blue': '#FFFFFF', 'off white': '#F8F8F8', 'coffee brown': '#4B3621',
            'charcoal': '#36454F', 'copper': '#B87333'
        }

        def pick_category(row):
            # Prefer more specific articleType, else subCategory
            for col in ['articleType', 'subCategory']:
                val = str(row.get(col, '')).strip()
                if val:
                    return val
            return 'Other'

        def color_to_hex(name: str):
            n = (name or '').strip().lower()
            return color_map.get(n)

        df = pd.read_csv(csv_path)
        take = df.head(max(1, min(limit, len(df))))

        added = 0
        for _, r in take.iterrows():
            try:
                data = {
                    'category': pick_category(r),
                    'color_hex': color_to_hex(str(r.get('baseColour', ''))),
                    'pattern': None,
                    'material': None,
                    'fit': None,
                    'seasonality': str(r.get('season', '') or '').lower() or None,
                    'occasion_tags': str(r.get('usage', '') or '').lower() or None,
                    'brand': None,
                    'price': None,
                    'size': None,
                    'image_url': None,
                }
                item = WardrobeItem(user_id=current_user.id, **data)
                db.session.add(item)
                added += 1
            except Exception:
                continue
        db.session.commit()
        flash(f'Imported {added} items from {file_name}.', 'success')
        return redirect(url_for('wardrobe'))

    @app.route('/ml/train', methods=['GET', 'POST'])
    @login_required
    def ml_train():
        """Upload a CSV dataset to train an ML model stored under instance/model.joblib."""
        metrics = None
        info = None
        if request.method == 'POST':
            file = request.files.get('dataset')
            if not file or not file.filename:
                flash('Please select a CSV file to upload.', 'warning')
                return redirect(url_for('ml_train'))
            if not file.filename.lower().endswith('.csv'):
                flash('Only .csv files are supported.', 'warning')
                return redirect(url_for('ml_train'))
            # Save to instance folder
            os.makedirs('instance', exist_ok=True)
            save_path = os.path.join('instance', secure_filename(file.filename))
            file.save(save_path)
            # Train
            try:
                from ml_model import train_from_csv
                result = train_from_csv(save_path)
                metrics = result.get('metrics')
                info = {k: v for k, v in result.items() if k != 'metrics'}
                flash('Model trained successfully.', 'success')
            except Exception as e:
                flash(f'Model training failed: {e}', 'danger')
        # Whether a model is currently available
        model_available = load_model() is not None
        return render_template('ml_train.html', metrics=metrics, info=info, model_available=model_available)

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            name = request.form.get('name', '').strip()
            if not email or not password:
                flash('Email and password are required.', 'danger')
                return redirect(url_for('signup'))
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'warning')
                return redirect(url_for('signup'))
            user = User(email=email, name=name)
            user.set_password(password)
            # Optional preferences on signup
            user.gender = request.form.get('gender')
            user.style_preference = request.form.get('style_preference')
            user.favorite_colors = request.form.get('favorite_colors')
            # Safe numeric parsing
            def _to_float(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
            user.height_cm = _to_float(request.form.get('height_cm'))
            user.weight_kg = _to_float(request.form.get('weight_kg'))
            user.size = request.form.get('size')
            user.fashion_goals = request.form.get('fashion_goals')
            db.session.add(user)
            db.session.commit()
            flash('Account created. Please log in.', 'success')
            return redirect(url_for('login'))
        return render_template('signup.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                remember = bool(request.form.get('remember'))
                login_user(user, remember=remember)
                flash('Welcome back!', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))
        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Logged out successfully.', 'info')
        return redirect(url_for('index'))

    def _compute_body_analysis(user: User):
        """Compute simple BMI-based analysis and clothing suggestions."""
        height_cm = user.height_cm or 0.0
        weight_kg = user.weight_kg or 0.0
        height_m = (height_cm / 100.0) if height_cm else 0.0
        bmi = round(weight_kg / (height_m ** 2), 1) if height_m and weight_kg else None

        if bmi is None:
            category = 'Unknown'
        elif bmi < 18.5:
            category = 'Underweight'
        elif bmi < 25:
            category = 'Normal'
        elif bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obese'

        # Basic fit and style guidance
        fit_reco = []
        if category == 'Underweight':
            fit_reco = ['regular-fit tops', 'layered looks', 'straight-fit bottoms']
        elif category == 'Normal':
            fit_reco = ['regular/slim-fit tops', 'straight or tapered bottoms']
        elif category == 'Overweight':
            fit_reco = ['relaxed-fit tops', 'straight/comfort-fit bottoms', 'structured outerwear']
        elif category == 'Obese':
            fit_reco = ['loose/relaxed tops', 'comfort-fit bottoms', 'longline outer layers']

        # Color and pattern tips
        color_tips = []
        if category in {'Overweight', 'Obese'}:
            color_tips = ['darker solid colors', 'vertical patterns', 'avoid overly shiny fabrics']
        elif category == 'Underweight':
            color_tips = ['lighter tones', 'horizontal stripes', 'textured knits']
        else:
            color_tips = ['neutrals with accent colors', 'subtle patterns']

        favorites = [c.strip() for c in (user.favorite_colors or '').split(',') if c.strip()]

        suggestions = [
            {
                'title': 'Fits to Try',
                'items': fit_reco or ['regular-fit staples']
            },
            {
                'title': 'Colors & Patterns',
                'items': (favorites[:3] + color_tips) if favorites else color_tips
            },
            {
                'title': 'Wardrobe Staples',
                'items': [
                    'well-fitted basic tee/shirt',
                    'versatile jeans/trousers',
                    'neutral sneakers/shoes',
                    'seasonal outer layer'
                ]
            }
        ]

        return {
            'bmi': bmi,
            'bmi_category': category,
            'height_cm': height_cm or None,
            'weight_kg': weight_kg or None,
            'size': user.size,
            'style_preference': user.style_preference,
            'favorite_colors': favorites,
            'suggestions': suggestions,
        }

    @app.route('/dashboard', methods=['GET', 'POST'])
    @login_required
    def dashboard():
        if request.method == 'POST':
            # Allow quick updates to measurements and preferences from the dashboard
            def _to_float(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
            current_user.height_cm = _to_float(request.form.get('height_cm'))
            current_user.weight_kg = _to_float(request.form.get('weight_kg'))
            current_user.size = request.form.get('size') or current_user.size
            current_user.style_preference = request.form.get('style_preference') or current_user.style_preference
            fav = request.form.get('favorite_colors')
            if fav is not None:
                current_user.favorite_colors = fav
            current_user.fashion_goals = request.form.get('fashion_goals') or current_user.fashion_goals
            db.session.commit()
            flash('Profile updated. Analysis refreshed.', 'success')

        items_count = WardrobeItem.query.filter_by(user_id=current_user.id).count()
        analysis = _compute_body_analysis(current_user)
        return render_template('dashboard.html', items_count=items_count, analysis=analysis)

    def _allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @app.route('/wardrobe/analysis')
    @login_required
    def wardrobe_analysis():
        """Display detailed analysis results from last image upload."""
        last_analysis = session.get('last_analysis', None)
        if not last_analysis:
            flash('No analysis results found. Please upload an image first.', 'warning')
            return redirect(url_for('wardrobe'))
        
        analysis = last_analysis.get('analysis', {})
        image_url = last_analysis.get('image_url', '')
        return render_template('image_analysis.html', analysis=analysis, image_url=image_url)

    @app.route('/wardrobe', methods=['GET', 'POST'])
    @login_required
    def wardrobe():
        if request.method == 'POST':
            data = {
                'category': request.form.get('category'),
                'color_hex': request.form.get('color_hex') or None,
                'pattern': request.form.get('pattern') or None,
                'material': request.form.get('material') or None,
                'fit': request.form.get('fit') or None,
                'seasonality': request.form.get('seasonality') or None,
                'occasion_tags': request.form.get('occasion_tags') or None,
                'brand': request.form.get('brand') or None,
                # parse price; keep 0.0 if provided, None if empty
                'price': None,
                'size': request.form.get('size') or None,
                'image_url': request.form.get('image_url') or None,
            }
            price_raw = request.form.get('price')
            if price_raw not in (None, ''):
                try:
                    data['price'] = float(price_raw)
                except ValueError:
                    flash('Price must be a number.', 'warning')
            # Handle file upload
            file = request.files.get('image_file')
            analysis_results = None
            analyzed_image_path = None
            
            if file and file.filename:
                if _allowed_file(file.filename):
                    fname = secure_filename(file.filename)
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    # avoid overwrite by adding suffix if needed
                    base, ext = os.path.splitext(fname)
                    counter = 1
                    while os.path.exists(save_path):
                        fname = f"{base}_{counter}{ext}"
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                        counter += 1
                    file.save(save_path)
                    data['image_url'] = url_for('static', filename=f'uploads/{fname}')
                    analyzed_image_path = save_path
                    
                    # AI Image Analysis - analyze the uploaded image
                    try:
                        analysis_results = analyze_clothing_image(save_path)
                        
                        # Store detailed analysis in session for display
                        session['last_analysis'] = {
                            'analysis': analysis_results,
                            'image_url': data['image_url']
                        }
                        
                        # Use AI-detected values to OVERWRITE form data (AI analysis takes priority)
                        # Overwrite all fields with AI analysis results
                        if analysis_results.get('category'):
                            data['category'] = analysis_results.get('category')
                        
                        if analysis_results.get('color_hex'):
                            data['color_hex'] = analysis_results.get('color_hex')
                        
                        if analysis_results.get('pattern'):
                            data['pattern'] = analysis_results.get('pattern')
                        
                        if analysis_results.get('material'):
                            data['material'] = analysis_results.get('material')
                        
                        if analysis_results.get('fit'):
                            data['fit'] = analysis_results.get('fit')
                        
                        if analysis_results.get('seasonality'):
                            data['seasonality'] = analysis_results.get('seasonality')
                        
                        if analysis_results.get('occasion_tags'):
                            data['occasion_tags'] = analysis_results.get('occasion_tags')
                        
                        # Show detailed flash messages
                        flash('✨ AI Analysis Complete! See details below:', 'success')
                        flash(f'📋 Category: {analysis_results.get("category", "Not detected").title()}', 'info')
                        
                        if analysis_results.get('color_hex'):
                            flash(f'🎨 Dominant Color: {analysis_results.get("color_hex")}', 'info')
                        
                        if analysis_results.get('pattern'):
                            flash(f'🔍 Pattern: {analysis_results.get("pattern").title()}', 'info')
                        
                        if analysis_results.get('material'):
                            flash(f'🧵 Material: {analysis_results.get("material").title()}', 'info')
                        
                        if analysis_results.get('fit'):
                            flash(f'📏 Fit: {analysis_results.get("fit").title()}', 'info')
                        
                        if analysis_results.get('seasonality'):
                            flash(f'🌤️ Season: {analysis_results.get("seasonality").title()}', 'info')
                        
                        if analysis_results.get('occasion_tags'):
                            flash(f'📍 Occasions: {analysis_results.get("occasion_tags")}', 'info')
                        
                        if analysis_results.get('where_to_wear'):
                            flash(f'💡 See full analysis details for styling tips and where to wear suggestions!', 'info')
                        
                        # Show overall confidence
                        confidence = analysis_results.get('confidence', 0.0)
                        if confidence > 0:
                            flash(f'🎯 Analysis Confidence: {confidence:.0%}', 'success')
                    
                    except Exception as e:
                        # If analysis fails, continue with manual input
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"Image analysis error: {e}")
                        print(f"Traceback: {error_details}")
                        flash(f'⚠️ AI analysis encountered an error: {str(e)}. You can still add the item manually.', 'warning')
                else:
                    flash('Unsupported image format. Allowed: png, jpg, jpeg, gif, webp', 'warning')
            
            if not data['category']:
                flash('Category is required. Please select one or upload an image for AI detection.', 'danger')
            else:
                item = WardrobeItem(user_id=current_user.id, **data)
                db.session.add(item)
                db.session.commit()
                flash('Item added to wardrobe.', 'success')
            return redirect(url_for('wardrobe'))
        items = WardrobeItem.query.filter_by(user_id=current_user.id).order_by(WardrobeItem.created_at.desc()).all()

        def _group_label(cat: str) -> str:
            c = (cat or '').lower()
            if c in {'shirt', 'tshirt', 'top', 'blouse', 'sweater', 'hoodie'}:
                return 'Top Wear'
            if c in {'pants', 'trousers', 'jeans', 'skirt', 'shorts'}:
                return 'Bottom Wear'
            if c in {'shoes', 'sneakers', 'heels', 'boots', 'loafers'}:
                return 'Shoes'
            if c in {'accessories', 'belt', 'watch', 'cap', 'hat', 'scarf'}:
                return 'Accessories'
            if c in {'jacket', 'coat', 'blazer', 'cardigan'}:
                return 'Outerwear'
            return 'Other'

        groups = {
            'Top Wear': [],
            'Bottom Wear': [],
            'Shoes': [],
            'Accessories': [],
            'Outerwear': [],
            'Other': [],
        }
        for it in items:
            groups[_group_label(it.category)].append(it)

        return render_template('wardrobe.html', items=items, groups=groups)

    @app.route('/wardrobe/delete/<int:item_id>', methods=['POST'])
    @login_required
    def wardrobe_delete(item_id):
        """Delete a wardrobe item."""
        item = WardrobeItem.query.get_or_404(item_id)
        # Ensure user owns this item
        if item.user_id != current_user.id:
            flash('You do not have permission to delete this item.', 'danger')
            return redirect(url_for('wardrobe'))
        
        # Delete associated image file if exists
        if item.image_url:
            try:
                # Extract filename from URL
                if 'uploads/' in item.image_url:
                    filename = item.image_url.split('uploads/')[-1]
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.exists(image_path):
                        os.remove(image_path)
            except Exception as e:
                print(f"Error deleting image file: {e}")
        
        db.session.delete(item)
        db.session.commit()
        flash('Item deleted from wardrobe.', 'success')
        return redirect(url_for('wardrobe'))

    @app.route('/wardrobe/delete-multiple', methods=['POST'])
    @login_required
    def wardrobe_delete_multiple():
        """Delete multiple wardrobe items at once."""
        item_ids = request.form.getlist('item_ids')
        
        if not item_ids:
            flash('No items selected for deletion.', 'warning')
            return redirect(url_for('wardrobe'))
        
        deleted_count = 0
        for item_id_str in item_ids:
            try:
                item_id = int(item_id_str)
                item = WardrobeItem.query.get(item_id)
                
                if item and item.user_id == current_user.id:
                    # Delete associated image file if exists
                    if item.image_url:
                        try:
                            if 'uploads/' in item.image_url:
                                filename = item.image_url.split('uploads/')[-1]
                                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                if os.path.exists(image_path):
                                    os.remove(image_path)
                        except Exception as e:
                            print(f"Error deleting image file: {e}")
                    
                    db.session.delete(item)
                    deleted_count += 1
            except (ValueError, TypeError):
                continue
        
        db.session.commit()
        flash(f'Successfully deleted {deleted_count} item(s) from wardrobe.', 'success')
        return redirect(url_for('wardrobe'))

    @app.route('/wardrobe/import', methods=['POST'])
    @login_required
    def wardrobe_import_csv():
        """Bulk import wardrobe items from an uploaded CSV file.
        Expected columns (any subset): category, color_hex, pattern, material, fit, seasonality, occasions/occasion_tags,
        brand, price, size, image_url
        """
        file = request.files.get('csv_file')
        if not file or not file.filename:
            flash('Please choose a CSV file.', 'warning')
            return redirect(url_for('wardrobe'))
        if not file.filename.lower().endswith('.csv'):
            flash('Only .csv files are supported for bulk import.', 'warning')
            return redirect(url_for('wardrobe'))
        # Save temp to instance
        os.makedirs('instance', exist_ok=True)
        tmp_path = os.path.join('instance', secure_filename(file.filename))
        file.save(tmp_path)
        # Parse
        try:
            import pandas as pd
            df = pd.read_csv(tmp_path)
        except Exception as e:
            flash(f'Failed to read CSV: {e}', 'danger')
            return redirect(url_for('wardrobe'))

        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        def col(name_list):
            for n in name_list:
                if n in cols:
                    return cols[n]
            return None

        cat_col = col(['category', 'articletype', 'subcategory'])
        color_col = col(['color_hex', 'colorhex', 'basecolour', 'basecolor', 'colour', 'color'])
        occ_col = col(['occasion_tags', 'occasions', 'usage'])
        price_col = col(['price'])
        brand_col = col(['brand'])
        size_col = col(['size'])
        pattern_col = col(['pattern'])
        material_col = col(['material'])
        fit_col = col(['fit'])
        season_col = col(['seasonality', 'season'])
        image_col = col(['image_url', 'imageurl'])

        added = 0
        max_rows = 500  # safety guard
        for _, r in df.head(max_rows).iterrows():
            category = (str(r.get(cat_col)) if cat_col else '').strip() or None
            if not category:
                continue
            def getv(c):
                return (str(r.get(c)).strip() if c and pd.notna(r.get(c)) else None)
            data = {
                'category': category,
                'color_hex': getv(color_col),
                'pattern': getv(pattern_col),
                'material': getv(material_col),
                'fit': getv(fit_col),
                'seasonality': getv(season_col),
                'occasion_tags': getv(occ_col),
                'brand': getv(brand_col),
                'size': getv(size_col),
                'image_url': getv(image_col),
                'price': None,
            }
            if price_col and pd.notna(r.get(price_col)):
                try:
                    data['price'] = float(r.get(price_col))
                except Exception:
                    data['price'] = None
            try:
                it = WardrobeItem(user_id=current_user.id, **data)
                db.session.add(it)
                added += 1
            except Exception:
                continue
        db.session.commit()
        flash(f'Imported {added} items from CSV.', 'success')
        return redirect(url_for('wardrobe'))

    @app.route('/recommendations', methods=['GET', 'POST'])
    @login_required
    def recommendations():
        # Build comprehensive context from form inputs (or empty for GET requests)
        use_image_matching = request.form.get('use_image_matching') == 'on' if request.method == 'POST' else False
        context = {
            "occasion": (request.form.get('occasion', '') or request.args.get('occasion', '')).lower().strip(),
            "weather": (request.form.get('weather', '') or request.args.get('weather', '')).lower().strip(),
            "time": (request.form.get('time_of_day', '') or request.args.get('time_of_day', '')).lower().strip(),
            "activity": (request.form.get('activity', '') or request.args.get('activity', '')).lower().strip(),
            "style": (request.form.get('style_preference', '') or request.args.get('style_preference', '') or current_user.style_preference).lower().strip(),
            "color_theme": (request.form.get('color_theme', '') or request.args.get('color_theme', '')).lower().strip()
        }
        
        # Map DB items to recommender Items (now includes all image analysis data)
        # Get ALL wardrobe items for the user - no filtering here
        db_items = WardrobeItem.query.filter_by(user_id=current_user.id).all()
        
        # Convert ALL items to Item objects - ensure we use everything from wardrobe
        items = [Item(id=i.id, category=i.category, color_hex=i.color_hex, pattern=i.pattern,
                      material=i.material, fit=i.fit, seasonality=i.seasonality,
                      occasions=i.occasion_list()) for i in db_items]
        
        # Debug: Log how many items we're working with
        print(f"Total wardrobe items: {len(db_items)}, Converted to Item objects: {len(items)}")
        
        # Build user preferences with more context
        user_prefs = {
            "style": context.get('style') or current_user.style_preference,
            "favorite_colors": [c.strip() for c in (current_user.favorite_colors or '').split(',') if c.strip()],
            "color_theme": context.get('color_theme')
        }
        
        outfits_db = []
        recommendation_type = "rule-based"
        
        # Helper function to get rule-based recommendations
        def get_rule_based_outfits():
            outfits = recommend(items, user_prefs, context)
            
            # If a trained model exists, re-rank outfits using model score
            model_obj = load_model()
            if model_obj and outfits:
                def to_simple(i: Item) -> SimpleItem:
                    return SimpleItem(
                        category=i.category,
                        pattern=i.pattern,
                        material=i.material,
                        fit=i.fit,
                        seasonality=i.seasonality,
                        color_hex=i.color_hex,
                        occasions=i.occasions or []
                    )
                def model_score(outfit: list[Item]) -> float:
                    simple_items = [to_simple(x) for x in outfit]
                    # Enhanced context for model
                    model_context = {
                        "occasion": context.get("occasion", ""),
                        "weather": context.get("weather", ""),
                        "time": context.get("time", ""),
                        "activity": context.get("activity", "")
                    }
                    return score_outfit(simple_items, model_context, model_obj)
                outfits = sorted(outfits, key=model_score, reverse=True)
            
            # Convert back to DB items for rendering
            id_to_item = {i.id: i for i in db_items}
            return [[id_to_item[x.id] for x in outfit if x.id in id_to_item] for outfit in outfits]
        
        # For GET requests, always show rule-based recommendations
        if request.method == 'GET':
            outfits_db = get_rule_based_outfits()
        # For POST requests, check if image matching is requested
        elif use_image_matching:
            # Check if user has items with images
            items_with_images = [item for item in db_items if item.image_url]
            
            if not items_with_images:
                flash('Image-based matching requires items with uploaded images. Please add images to your wardrobe items first.', 'info')
                outfits_db = get_rule_based_outfits()
            else:
                try:
                    # Get image-based recommendations
                    app_root = app.root_path
                    max_results = 5
                    image_recommendations = get_outfit_recommendations(items_with_images, app_root, max_results)
                    
                    if image_recommendations and isinstance(image_recommendations, list) and len(image_recommendations) > 0:
                        # Convert image-based recommendations to DB items format
                        try:
                            outfits_db = []
                            for rec in image_recommendations:
                                if isinstance(rec, dict) and 'items' in rec:
                                    outfits_db.append(rec['items'])
                            
                            if outfits_db:
                                recommendation_type = "image-based"
                                flash(f'Found {len(outfits_db)} image-based outfit recommendations!', 'success')
                            else:
                                flash('No valid image-based outfits found. Make sure you have items in multiple categories (tops, bottoms, shoes, accessories) with images.', 'info')
                                outfits_db = get_rule_based_outfits()
                        except Exception as e:
                            print(f"Error processing image recommendations: {e}")
                            flash('Error processing image-based recommendations. Falling back to rule-based.', 'warning')
                            outfits_db = get_rule_based_outfits()
                    else:
                        flash('No image-based outfits found. Make sure you have items in multiple categories (tops, bottoms, shoes, accessories) with images.', 'info')
                        outfits_db = get_rule_based_outfits()
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Image-based matching error: {error_details}")
                    flash(f'Image-based matching encountered an error: {str(e)}. Falling back to rule-based recommendations.', 'warning')
                    outfits_db = get_rule_based_outfits()
        else:
            # POST request without image matching - use rule-based
            outfits_db = get_rule_based_outfits()
        
        return render_template('recommendations.html', 
                             outfits=outfits_db, 
                             context=context,
                             recommendation_type=recommendation_type,
                             use_image_matching=use_image_matching)

    return app


if __name__ == '__main__':
    app = create_app()
    host = os.environ.get('HOST', '127.0.0.1')
    try:
        port = int(os.environ.get('PORT', '5000'))
    except ValueError:
        port = 5000
    debug = os.environ.get('FLASK_ENV', '').lower() == 'development'
    app.run(host=host, port=port, debug=debug)
