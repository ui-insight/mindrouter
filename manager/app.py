# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import secrets
from functools import wraps
import sqlite3

# App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mindrouter.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/profile_pics'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['RECAPTCHA_SITE_KEY'] = '6Lf9g5AqAAAAAJKSMBpdd9VydS3nEQ312U6KCFB6'  
app.config['RECAPTCHA_SECRET_KEY'] = '6Lf9g5AqAAAAAODk_A7vy_ups9RdsDHxur9w0p0l'  

# SHENEMAN
app.config['SESSION_COOKIE_SECURE'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    full_name = db.Column(db.String(100))
    department = db.Column(db.String(100))
    college = db.Column(db.String(100))
    title = db.Column(db.String(100))
    profile_pic = db.Column(db.String(200))
    description = db.Column(db.Text)
    budget_index = db.Column(db.String(50))
    monthly_token_limit = db.Column(db.Integer, default=100000)
    is_admin = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=False)
    registration_token = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    api_keys = db.relationship('APIKey', backref='user', lazy=True)
    usage_records = db.relationship('UsageRecord', backref='user', lazy=True)

class APIKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    permissions = db.Column(db.String(200))
    last_used = db.Column(db.DateTime)

class UsageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_key.id'), nullable=False)
    tokens_used = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    request_type = db.Column(db.String(50))
    model_used = db.Column(db.String(50))

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need administrator privileges to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        # Verify reCAPTCHA here
        # Add your reCAPTCHA verification logic
        
        registration_token = secrets.token_urlsafe(32)
        user = User(
            email=email,
            registration_token=registration_token,
            is_approved=False
        )
        
        db.session.add(user)
        db.session.commit()
        
        # TODO: Send registration email with token
        # Add your email sending logic here
        
        flash('Registration link has been sent to your email', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/complete-registration/<token>', methods=['GET', 'POST'])
def complete_registration(token):
    user = User.query.filter_by(registration_token=token).first()
    
    if not user:
        flash('Invalid or expired registration link', 'error')
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        user.password_hash = generate_password_hash(password)
        user.full_name = request.form.get('full_name')
        user.department = request.form.get('department')
        user.college = request.form.get('college')
        user.title = request.form.get('title')
        user.description = request.form.get('description')
        user.budget_index = request.form.get('budget_index')
        
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename:
                filename = secure_filename(f"{user.id}_{file.filename}")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                user.profile_pic = filename
        
        user.registration_token = None
        db.session.commit()
        
        flash('Registration completed successfully', 'success')
        return redirect(url_for('login'))
    
    return render_template('complete_registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            if not user.is_approved:
                flash('Your account is pending approval', 'warning')
                return redirect(url_for('login'))
            
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
            
        flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get last 30 days of usage data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    # Query usage records
    usage_records = UsageRecord.query.filter(
        UsageRecord.user_id == current_user.id,
        UsageRecord.timestamp >= start_date,
        UsageRecord.timestamp <= end_date
    ).order_by(UsageRecord.timestamp).all()
    
    # Create daily usage dictionary
    daily_usage = {}
    dates = []
    current_date = start_date
    while current_date <= end_date:
        daily_usage[current_date.strftime('%Y-%m-%d')] = 0
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # Fill in actual usage data
    for record in usage_records:
        date_key = record.timestamp.strftime('%Y-%m-%d')
        daily_usage[date_key] = daily_usage.get(date_key, 0) + record.tokens_used
    
    # Convert to list for Chart.js
    usage_data = [daily_usage[date] for date in dates]
    
    return render_template('dashboard.html', 
                         dates=dates,
                         usage_data=usage_data)

@app.route('/api/usage-data')
@login_required
def get_usage_data():
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    usage_records = UsageRecord.query.filter(
        UsageRecord.user_id == current_user.id,
        UsageRecord.timestamp >= start_date,
        UsageRecord.timestamp <= end_date
    ).order_by(UsageRecord.timestamp).all()
    
    daily_usage = {}
    dates = []
    current_date = start_date
    while current_date <= end_date:
        daily_usage[current_date.strftime('%Y-%m-%d')] = 0
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    for record in usage_records:
        date_key = record.timestamp.strftime('%Y-%m-%d')
        daily_usage[date_key] = daily_usage.get(date_key, 0) + record.tokens_used
    
    usage_data = [daily_usage[date] for date in dates]
    
    return jsonify({
        'dates': dates,
        'usage': usage_data
    })

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    pending_users = User.query.filter_by(is_approved=False).all()
    all_users = User.query.all()
    return render_template('admin_dashboard.html', 
                         pending_users=pending_users,
                         all_users=all_users)

@app.route('/admin/approve/<int:user_id>')
@login_required
@admin_required
def approve_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_approved = True
    db.session.commit()
    
    # TODO: Send approval notification email
    
    flash(f'User {user.email} has been approved', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.full_name = request.form.get('full_name')
        current_user.department = request.form.get('department')
        current_user.college = request.form.get('college')
        current_user.title = request.form.get('title')
        current_user.description = request.form.get('description')
        current_user.budget_index = request.form.get('budget_index')
        
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename:
                filename = secure_filename(f"{current_user.id}_{file.filename}")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                current_user.profile_pic = filename
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

@app.route('/api-keys')
@login_required
def api_keys():
    return render_template('api_keys.html', keys=current_user.api_keys)

@app.route('/api-keys/generate', methods=['POST'])
@login_required
def generate_api_key():
    api_key = secrets.token_urlsafe(32)
    new_key = APIKey(
        key=api_key,
        user_id=current_user.id,
        permissions=request.form.get('permissions', 'basic')
    )
    db.session.add(new_key)
    db.session.commit()
    
    flash('New API key generated successfully', 'success')
    return redirect(url_for('api_keys'))

@app.route('/api-keys/revoke/<int:key_id>', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    api_key = APIKey.query.get_or_404(key_id)
    if api_key.user_id != current_user.id and not current_user.is_admin:
        flash('Unauthorized action', 'error')
        return redirect(url_for('api_keys'))
    
    api_key.is_active = False
    db.session.commit()
    
    flash('API key revoked successfully', 'success')
    return redirect(url_for('api_keys'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Initialize the database
def init_db():
    with app.app_context():
        db.create_all()
        # Create admin user if none exists
        admin = User.query.filter_by(is_admin=True).first()
        if not admin:
            admin = User(
                email='admin@example.com',
                password_hash=generate_password_hash('admin'),
                full_name='Admin User',
                is_admin=True,
                is_approved=True
            )
            db.session.add(admin)
            db.session.commit()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8009)

