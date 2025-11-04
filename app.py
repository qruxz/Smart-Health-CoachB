# app.py - CORRECTED VERSION
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import uuid
from typing import Dict, Optional
from flask_jwt_extended.exceptions import NoAuthorizationError
# Authentication
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, 
    unset_jwt_cookies, verify_jwt_in_request
)


# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app, supports_credentials=True, origins=['http://localhost:3000', 'http://localhost:5173'])

# Setup JWT
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "supersecretkey123changethis")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=12)
jwt = JWTManager(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('health_coach.log'),
        logging.StreamHandler()
    ]
)

# Create necessary directories
logs_dir = Path('logs')
user_data_dir = Path('user_data')
logs_dir.mkdir(exist_ok=True)
user_data_dir.mkdir(exist_ok=True)

# Users file
users_file = Path('users.json')
if not users_file.exists():
    users_file.write_text('[]')

# Get Groq API key
api_key = os.getenv('GROQ_API_KEY')
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None
    logging.warning("⚠️  GROQ_API_KEY not found in environment variables")

# ---------------------------
# Utility: Users store
# ---------------------------
def load_users():
    try:
        with open(users_file, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_users(users):
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=2)

def find_user_by_email(email: str) -> Optional[Dict]:
    users = load_users()
    return next((u for u in users if u.get('email') == email), None)

def find_user_by_id(user_id: str) -> Optional[Dict]:
    users = load_users()
    return next((u for u in users if u.get('id') == user_id), None)

# ---------------------------
# Profile helpers
# ---------------------------
def default_profile_for_user(user_id: str, email: str, username: str) -> Dict:
    return {
        'user_id': user_id,
        'email': email,
        'username': username,
        'created_at': datetime.now().isoformat(),
        'profile': {
            'name': None,
            'age': None,
            'weight': None,
            'height': None,
            'fitness_level': 'beginner',
            'health_goals': [],
            'dietary_preferences': [],
            'medical_conditions': []
        },
        'history': {
            'workouts': [],
            'meals': [],
            'health_metrics': [],
            'goals': [],
            'achievements': []
        },
        'preferences': {
            'workout_time': 'morning',
            'meal_preferences': [],
            'notification_enabled': True
        }
    }

def profile_path_for_user_id(user_id: str) -> Path:
    return user_data_dir / f'{user_id}.json'

def load_profile_by_user_id(user_id: str) -> Dict:
    p = profile_path_for_user_id(user_id)
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return default_profile_for_user(user_id, "", "")

def save_profile_by_user_id(user_id: str, profile: Dict):
    p = profile_path_for_user_id(user_id)
    with open(p, 'w') as f:
        json.dump(profile, f, indent=2)

# ---------------------------
# Logging interactions
# ---------------------------
def log_interaction(user_id: str, message: str, response: str, category: str = None):
    log_file = logs_dir / f'interactions_{datetime.now().strftime("%Y-%m-%d")}.json'
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'category': category,
        'message': message,
        'response': response[:200] + '...' if len(response) > 200 else response,
        'ip_address': request.remote_addr
    }
    try:
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write log: {e}")

# ---------------------------
# System prompts & AI helper
# ---------------------------
SYSTEM_PROMPTS = {
    'nutrition': """You are an expert nutritionist and meal planning specialist. 
    Help users create balanced, nutritious meal plans based on their goals, dietary preferences, and restrictions.
    Provide specific meal suggestions with approximate calories and macronutrients.
    Always consider allergies, dietary restrictions, and cultural preferences.""",
    
    'fitness': """You are a certified fitness trainer and exercise physiologist.
    Create personalized workout plans based on user's fitness level, goals, and available equipment.
    Provide clear exercise instructions, sets, reps, and rest periods.
    Always prioritize safety and proper form. Adapt workouts for injuries or limitations.""",
    
    'schedule': """You are a productivity and wellness scheduling expert.
    Help users create balanced daily/weekly schedules that incorporate work, exercise, meals, sleep, and self-care.
    Use time-blocking techniques and ensure realistic, sustainable routines.""",
    
    'wellness': """You are a holistic wellness coach specializing in mental health, stress management, and lifestyle optimization.
    Provide evidence-based advice on sleep hygiene, stress reduction, mindfulness, and work-life balance.
    Be empathetic and supportive while maintaining professional boundaries.""",
    
    'goals': """You are a goal-setting and achievement coach.
    Help users set SMART (Specific, Measurable, Achievable, Relevant, Time-bound) health goals.
    Break down large goals into manageable milestones and provide motivation strategies.""",
    
    'analytics': """You are a data analyst specializing in health metrics and progress tracking.
    Analyze user's health data, identify patterns, and provide actionable insights.
    Use clear visualizations concepts and celebrate wins while addressing areas for improvement.""",
    
    'general': """You are an AI-powered Smart Health Coach providing comprehensive wellness support.
    You can help with nutrition, fitness, scheduling, mental wellness, goal setting, and health analytics.
    Always be encouraging, supportive, and provide evidence-based advice.
    If asked about medical conditions, remind users to consult healthcare professionals."""
}

def detect_intent(message: str) -> str:
    message_lower = message.lower()
    keywords = {
        'nutrition': ['meal', 'diet', 'food', 'eat', 'nutrition', 'calorie', 'recipe', 'hungry'],
        'fitness': ['workout', 'exercise', 'train', 'gym', 'fitness', 'muscle', 'cardio', 'strength'],
        'schedule': ['schedule', 'plan', 'routine', 'time', 'organize', 'calendar', 'daily'],
        'wellness': ['sleep', 'stress', 'mental', 'relax', 'meditation', 'wellness', 'tired', 'anxiety'],
        'goals': ['goal', 'target', 'achieve', 'motivation', 'progress', 'milestone'],
        'analytics': ['track', 'progress', 'analyze', 'data', 'metrics', 'statistics', 'report']
    }
    for category, words in keywords.items():
        if any(word in message_lower for word in words):
            return category
    return 'general'

def generate_health_response(message: str, category: Optional[str], user_profile: Dict) -> str:
    if not client:
        return "🔧 I'm currently unavailable. Please ensure the GROQ_API_KEY is configured in your .env file."
    
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS['general'])
    
    # Build user context
    fitness_level = user_profile['profile'].get('fitness_level', 'Not specified')
    health_goals = ', '.join(user_profile['profile'].get('health_goals', [])) or 'Not specified'
    dietary_prefs = ', '.join(user_profile['profile'].get('dietary_preferences', [])) or 'Not specified'
    medical_conditions = ', '.join(user_profile['profile'].get('medical_conditions', [])) or 'None reported'
    
    user_context = f"""
USER PROFILE:
- Fitness Level: {fitness_level}
- Health Goals: {health_goals}
- Dietary Preferences: {dietary_prefs}
- Medical Conditions: {medical_conditions}
"""
    
    full_prompt = f"""{user_context}

USER MESSAGE: {message}

INSTRUCTIONS:
- Provide a helpful, personalized response based on the user's profile and message
- Keep responses concise but informative (150-300 words)
- Include actionable advice and specific recommendations
- Use formatting for better readability
- If creating plans, use structured formats with clear sections
- Always encourage and motivate the user
- For medical questions, remind them to consult healthcare professionals

Response:"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return "💪 I encountered an issue processing your request. Please try again!"

# ---------------------------
# AUTH ROUTES
# ---------------------------
@app.route("/api/auth/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json(force=True)
        email = (data.get("email") or "").strip().lower()
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not email or not username or not password:
            return jsonify({"detail": "All fields required"}), 400

        if find_user_by_email(email):
            return jsonify({"detail": "Email already registered"}), 400

        user_id = str(uuid.uuid4())
        hashed_pw = generate_password_hash(password)

        new_user = {
            "id": user_id,
            "email": email,
            "username": username,
            "password": hashed_pw,
            "created_at": datetime.now().isoformat()
        }
        users = load_users()
        users.append(new_user)
        save_users(users)

        # Create default profile
        profile = default_profile_for_user(user_id, email, username)
        save_profile_by_user_id(user_id, profile)

        logging.info(f"✅ New user registered: {username} ({email})")
        return jsonify({
            "message": "User registered successfully",
            "email": email,
            "username": username
        }), 201
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({"detail": "Failed to register user"}), 500

@app.route("/api/auth/login", methods=["POST"])
def login_user():
    try:
        data = request.get_json(force=True)
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        user = find_user_by_email(email)
        if not user or not check_password_hash(user.get("password", ""), password):
            return jsonify({"detail": "Invalid email or password"}), 401

        access_token = create_access_token(identity=user["id"])
        
        logging.info(f"✅ User logged in: {user['username']}")
        return jsonify({
            "access_token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "username": user["username"]
            }
        })
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({"detail": "Failed to login"}), 500

@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def get_current_user():
    try:
        user_id = get_jwt_identity()
        user = find_user_by_id(user_id)
        if not user:
            return jsonify({"detail": "User not found"}), 404
        
        return jsonify({
            "id": user["id"],
            "email": user["email"],
            "username": user["username"],
            "created_at": user.get("created_at")
        })
    except Exception as e:
        logging.error(f"/auth/me error: {e}")
        return jsonify({"detail": "Failed to fetch user"}), 500

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    resp = jsonify({"msg": "Logged out successfully"})
    unset_jwt_cookies(resp)
    return resp

# ---------------------------
# MAIN ROUTES
# ---------------------------
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Smart Health Coach API is running',
        'groq_configured': api_key is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - works for both anonymous and logged-in users"""
    try:
        data = request.get_json(force=True)
        message = (data.get('message') or '').strip()
        category = data.get('category')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Try to get JWT user, fallback to session
        user_id = None
        try:
            verify_jwt_in_request_optional()
            user_id = get_jwt_identity()
        except:
            user_id = None

        if not user_id:
            session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
            user_id = session_id

        user_profile = load_profile_by_user_id(user_id)
        
        if not category:
            category = detect_intent(message)

        ai_response = generate_health_response(message, category, user_profile)
        log_interaction(user_id, message, ai_response, category)

        return jsonify({
            'response': ai_response,
            'category': category,
            'session_id': user_id,
            'success': True
        })
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({
            'error': 'Failed to process message',
            'success': False
        }), 500

@app.route('/api/profile', methods=['GET', 'POST'])
@jwt_required()
def user_profile():
    """Get or update user profile"""
    try:
        user_id = get_jwt_identity()
        
        if request.method == 'GET':
            profile = load_profile_by_user_id(user_id)
            return jsonify(profile)
        else:
            data = request.get_json(force=True)
            profile = load_profile_by_user_id(user_id)
            
            if 'profile' in data:
                profile['profile'].update(data['profile'])
            if 'preferences' in data:
                profile['preferences'].update(data['preferences'])
            
            save_profile_by_user_id(user_id, profile)
            logging.info(f"✅ Profile updated for user: {user_id}")
            
            return jsonify({
                'message': 'Profile updated successfully',
                'profile': profile,
                'success': True
            })
    except Exception as e:
        logging.error(f"Profile error: {e}")
        return jsonify({'error': 'Failed to update profile', 'success': False}), 500

@app.route('/api/metrics', methods=['GET', 'POST'])
@jwt_required()
def health_metrics():
    """Log or retrieve health metrics"""
    try:
        user_id = get_jwt_identity()
        profile = load_profile_by_user_id(user_id)
        
        if request.method == 'GET':
            return jsonify({
                'metrics': profile['history'].get('health_metrics', []),
                'success': True
            })
        else:
            data = request.get_json(force=True)
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': data.get('type'),
                'value': data.get('value'),
                'unit': data.get('unit')
            }
            profile['history']['health_metrics'].append(metric_entry)
            save_profile_by_user_id(user_id, profile)
            
            return jsonify({'message': 'Metric logged successfully', 'success': True})
    except Exception as e:
        logging.error(f"Metric logging error: {e}")
        return jsonify({'error': 'Failed to log metric', 'success': False}), 500

@app.route('/api/workout/generate', methods=['POST'])
@jwt_required()
def generate_workout():
    """Generate personalized workout plan"""
    try:
        user_id = get_jwt_identity()
        user_profile = load_profile_by_user_id(user_id)
        data = request.get_json(force=True)
        
        duration = data.get('duration', '30 minutes')
        focus = data.get('focus', 'full body')
        equipment = data.get('equipment', 'none')
        
        prompt = f"""Create a {duration} {focus} workout plan.
Available equipment: {equipment}

Provide 5-7 exercises with:
- Exercise name
- Sets and reps
- Brief form tips
- Rest periods

Format with clear structure."""
        
        response = generate_health_response(prompt, 'fitness', user_profile)
        
        # Save to history
        workout_entry = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'focus': focus,
            'equipment': equipment,
            'plan': response
        }
        user_profile['history']['workouts'].append(workout_entry)
        save_profile_by_user_id(user_id, user_profile)
        
        return jsonify({'workout_plan': response, 'success': True})
    except Exception as e:
        logging.error(f"Workout generation error: {e}")
        return jsonify({'error': 'Failed to generate workout', 'success': False}), 500

@app.route('/api/meal-plan/generate', methods=['POST'])
@jwt_required()
def generate_meal():
    """Generate personalized meal plan"""
    try:
        user_id = get_jwt_identity()
        user_profile = load_profile_by_user_id(user_id)
        data = request.get_json(force=True)
        
        meal_type = data.get('meal_type', 'full day')
        preferences = data.get('preferences', '')
        
        prompt = f"""Create a {meal_type} meal plan.
Preferences: {preferences}

Include:
- Meal names with ingredients
- Approximate calories and macros
- Preparation tips

Format with clear structure."""
        
        response = generate_health_response(prompt, 'nutrition', user_profile)
        
        # Save to history
        meal_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': meal_type,
            'preferences': preferences,
            'plan': response
        }
        user_profile['history']['meals'].append(meal_entry)
        save_profile_by_user_id(user_id, user_profile)
        
        return jsonify({'meal_plan': response, 'success': True})
    except Exception as e:
        logging.error(f"Meal plan generation error: {e}")
        return jsonify({'error': 'Failed to generate meal plan', 'success': False}), 500

@app.route('/api/goals', methods=['GET', 'POST'])
@jwt_required()
def manage_goals():
    """Create or retrieve goals"""
    try:
        user_id = get_jwt_identity()
        profile = load_profile_by_user_id(user_id)
        
        if request.method == 'GET':
            return jsonify({'goals': profile['history'].get('goals', []), 'success': True})
        else:
            data = request.get_json(force=True)
            goal = {
                'id': str(uuid.uuid4()),
                'title': data.get('title'),
                'description': data.get('description'),
                'target_date': data.get('target_date'),
                'category': data.get('category'),
                'created_at': datetime.now().isoformat(),
                'completed': False,
                'progress': 0
            }
            profile['history']['goals'].append(goal)
            save_profile_by_user_id(user_id, profile)
            
            logging.info(f"✅ Goal created for user {user_id}: {goal['title']}")
            return jsonify({'message': 'Goal created successfully', 'goal': goal, 'success': True})
    except Exception as e:
        logging.error(f"Goal creation error: {e}")
        return jsonify({'error': 'Failed to create goal', 'success': False}), 500

@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    """Get user analytics"""
    try:
        user_id = get_jwt_identity()
        profile = load_profile_by_user_id(user_id)
        
        metrics = profile['history'].get('health_metrics', [])
        goals = profile['history'].get('goals', [])
        
        analytics = {
            'total_workouts': len(profile['history'].get('workouts', [])),
            'total_meals_logged': len(profile['history'].get('meals', [])),
            'active_goals': len([g for g in goals if not g.get('completed', False)]),
            'completed_goals': len([g for g in goals if g.get('completed', False)]),
            'metrics_logged': len(metrics),
            'days_active': len(set(m['timestamp'][:10] for m in metrics if 'timestamp' in m))
        }
        
        return jsonify({'analytics': analytics, 'success': True})
    except Exception as e:
        logging.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to generate analytics', 'success': False}), 500

# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("💜 SMART HEALTH COACH AI - BACKEND SERVER")
    print("=" * 60)
    print(f"✅ Status: {'Ready' if client else '⚠️  API Key Missing'}")
    print(f"🔑 Groq API: {'Configured' if api_key else 'Not configured'}")
    print(f"🌐 Server: http://localhost:5001")
    print(f"🔐 JWT: {'Configured' if os.getenv('JWT_SECRET_KEY') else 'Using default (change this!)'}")
    print("\n📋 Available Endpoints:")
    print("   POST   /api/auth/register       - Register new user")
    print("   POST   /api/auth/login          - Login (returns access_token)")
    print("   GET    /api/auth/me             - Get current user")
    print("   POST   /api/auth/logout         - Logout")
    print("   POST   /api/chat                - Chat with AI coach")
    print("   GET    /api/health              - Health check")
    print("   GET/POST /api/profile           - Get/update profile")
    print("   GET/POST /api/metrics           - Health metrics")
    print("   POST   /api/workout/generate    - Generate workout")
    print("   POST   /api/meal-plan/generate  - Generate meal plan")
    print("   GET/POST /api/goals             - Manage goals")
    print("   GET    /api/analytics           - User analytics")
    print("\n💡 Setup Instructions:")
    print("   1. Create a .env file with:")
    print("      GROQ_API_KEY=your_groq_api_key_here")
    print("      JWT_SECRET_KEY=your_secret_key_here")
    print("   2. Install dependencies: pip install flask flask-cors flask-jwt-extended groq python-dotenv werkzeug")
    print("   3. Run: python app.py")
    print("\n📁 Data Storage:")
    print("   - Users: users.json")
    print("   - Profiles: user_data/")
    print("   - Logs: logs/")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
