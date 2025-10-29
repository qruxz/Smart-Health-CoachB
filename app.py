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
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

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

# Get Groq API key
api_key = os.getenv('GROQ_API_KEY')
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None
    logging.warning("⚠️  GROQ_API_KEY not found in environment variables")

# ======================= USER SESSION MANAGEMENT =======================

def get_user_id():
    """Generate or retrieve user session ID"""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

def get_user_profile(user_id: str) -> Dict:
    """Load user profile and history"""
    profile_file = user_data_dir / f'{user_id}.json'
    
    if profile_file.exists():
        with open(profile_file, 'r') as f:
            return json.load(f)
    
    # Default profile
    return {
        'user_id': user_id,
        'created_at': datetime.now().isoformat(),
        'profile': {
            'name': None,
            'age': None,
            'weight': None,
            'height': None,
            'fitness_level': None,
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

def save_user_profile(user_id: str, profile: Dict):
    """Save user profile"""
    profile_file = user_data_dir / f'{user_id}.json'
    with open(profile_file, 'w') as f:
        json.dump(profile, f, indent=2)

def log_interaction(user_id: str, message: str, response: str, category: str = None):
    """Log user interactions"""
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

# ======================= HEALTH COACH SYSTEM PROMPTS =======================

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

# ======================= AI RESPONSE GENERATION =======================

def generate_health_response(message: str, category: Optional[str], user_profile: Dict) -> str:
    """Generate AI response using Groq"""
    
    if not client:
        return "I'm currently unavailable. Please ensure the GROQ_API_KEY is configured properly. 🔧"
    
    # Select appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS['general'])
    
    # Build context from user profile
    user_context = f"""
USER PROFILE:
- Fitness Level: {user_profile['profile'].get('fitness_level', 'Not specified')}
- Health Goals: {', '.join(user_profile['profile'].get('health_goals', [])) or 'Not specified'}
- Dietary Preferences: {', '.join(user_profile['profile'].get('dietary_preferences', [])) or 'Not specified'}
- Medical Conditions: {', '.join(user_profile['profile'].get('medical_conditions', [])) or 'None reported'}
"""
    
    # Build conversation prompt
    prompt = f"""{system_prompt}

{user_context}

USER MESSAGE: {message}

INSTRUCTIONS:
- Provide a helpful, personalized response based on the user's profile and message
- Use markdown formatting for better readability (bold, lists, etc.)
- Keep responses concise but informative (aim for 150-300 words)
- Include actionable advice and specific recommendations
- Use emojis sparingly to make responses engaging
- If creating plans, use structured formats with clear sections
- Always encourage and motivate the user

Response:"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return f"I encountered an issue processing your request. Please try again! 💪"

# ======================= SMART FEATURES =======================

def detect_intent(message: str) -> str:
    """Detect user intent from message"""
    message_lower = message.lower()
    
    keywords = {
        'nutrition': ['meal', 'diet', 'food', 'eat', 'nutrition', 'calorie', 'recipe'],
        'fitness': ['workout', 'exercise', 'train', 'gym', 'fitness', 'muscle', 'cardio'],
        'schedule': ['schedule', 'plan', 'routine', 'time', 'organize', 'calendar'],
        'wellness': ['sleep', 'stress', 'mental', 'relax', 'meditation', 'wellness', 'tired'],
        'goals': ['goal', 'target', 'achieve', 'motivation', 'progress'],
        'analytics': ['track', 'progress', 'analyze', 'data', 'metrics', 'statistics']
    }
    
    for category, words in keywords.items():
        if any(word in message_lower for word in words):
            return category
    
    return 'general'

def generate_meal_plan(user_profile: Dict, preferences: str = "") -> Dict:
    """Generate a structured meal plan"""
    prompt = f"""Create a one-day meal plan with breakfast, lunch, dinner, and 2 snacks.
    
User preferences: {preferences}
Dietary restrictions: {', '.join(user_profile['profile'].get('dietary_preferences', []))}

Format as JSON with this structure:
{{
  "breakfast": {{"meal": "...", "calories": 000, "protein": 00}},
  "lunch": {{"meal": "...", "calories": 000, "protein": 00}},
  "dinner": {{"meal": "...", "calories": 000, "protein": 00}},
  "snacks": [{{"meal": "...", "calories": 000}}]
}}"""
    
    # This is a placeholder - you can enhance with actual Groq call
    return {
        "breakfast": {"meal": "Oatmeal with berries and almonds", "calories": 350, "protein": 12},
        "lunch": {"meal": "Grilled chicken salad with quinoa", "calories": 450, "protein": 35},
        "dinner": {"meal": "Salmon with roasted vegetables", "calories": 500, "protein": 40},
        "snacks": [
            {"meal": "Greek yogurt with honey", "calories": 150},
            {"meal": "Apple with almond butter", "calories": 200}
        ]
    }

# ======================= API ENDPOINTS =======================

@app.route('/')
def serve_frontend():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Smart Health Coach API is running',
        'groq_configured': api_key is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        category = data.get('category')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user session
        user_id = get_user_id()
        user_profile = get_user_profile(user_id)
        
        # Auto-detect category if not provided
        if not category:
            category = detect_intent(message)
        
        # Generate AI response
        ai_response = generate_health_response(message, category, user_profile)
        
        # Log interaction
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
def user_profile():
    """Get or update user profile"""
    user_id = get_user_id()
    
    if request.method == 'GET':
        profile = get_user_profile(user_id)
        return jsonify({
            'profile': profile,
            'success': True
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            profile = get_user_profile(user_id)
            
            # Update profile fields
            if 'profile' in data:
                profile['profile'].update(data['profile'])
            
            if 'preferences' in data:
                profile['preferences'].update(data['preferences'])
            
            save_user_profile(user_id, profile)
            
            return jsonify({
                'message': 'Profile updated successfully',
                'profile': profile,
                'success': True
            })
            
        except Exception as e:
            logging.error(f"Profile update error: {e}")
            return jsonify({
                'error': 'Failed to update profile',
                'success': False
            }), 500

@app.route('/api/metrics', methods=['GET', 'POST'])
def health_metrics():
    """Get or log health metrics"""
    user_id = get_user_id()
    profile = get_user_profile(user_id)
    
    if request.method == 'GET':
        return jsonify({
            'metrics': profile['history'].get('health_metrics', []),
            'success': True
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': data.get('type'),  # steps, calories, water, sleep
                'value': data.get('value'),
                'unit': data.get('unit')
            }
            
            profile['history']['health_metrics'].append(metric_entry)
            save_user_profile(user_id, profile)
            
            return jsonify({
                'message': 'Metric logged successfully',
                'success': True
            })
            
        except Exception as e:
            logging.error(f"Metric logging error: {e}")
            return jsonify({
                'error': 'Failed to log metric',
                'success': False
            }), 500

@app.route('/api/workout/generate', methods=['POST'])
def generate_workout():
    """Generate personalized workout plan"""
    try:
        user_id = get_user_id()
        user_profile = get_user_profile(user_id)
        data = request.get_json()
        
        duration = data.get('duration', '30 minutes')
        focus = data.get('focus', 'full body')
        equipment = data.get('equipment', 'none')
        
        prompt = f"""Create a {duration} {focus} workout plan.
        
User fitness level: {user_profile['profile'].get('fitness_level', 'beginner')}
Available equipment: {equipment}

Provide 5-7 exercises with:
- Exercise name
- Sets and reps
- Brief form tips
- Rest periods

Format with clear structure using markdown."""

        response = generate_health_response(prompt, 'fitness', user_profile)
        
        return jsonify({
            'workout_plan': response,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Workout generation error: {e}")
        return jsonify({
            'error': 'Failed to generate workout',
            'success': False
        }), 500

@app.route('/api/meal-plan/generate', methods=['POST'])
def generate_meal():
    """Generate personalized meal plan"""
    try:
        user_id = get_user_id()
        user_profile = get_user_profile(user_id)
        data = request.get_json()
        
        meal_type = data.get('meal_type', 'full day')
        preferences = data.get('preferences', '')
        
        prompt = f"""Create a {meal_type} meal plan.

User preferences: {preferences}
Dietary restrictions: {', '.join(user_profile['profile'].get('dietary_preferences', []))}
Health goals: {', '.join(user_profile['profile'].get('health_goals', []))}

Include:
- Meal names with ingredients
- Approximate calories and macros
- Preparation tips

Format with clear structure using markdown."""

        response = generate_health_response(prompt, 'nutrition', user_profile)
        
        return jsonify({
            'meal_plan': response,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Meal plan generation error: {e}")
        return jsonify({
            'error': 'Failed to generate meal plan',
            'success': False
        }), 500

@app.route('/api/goals', methods=['GET', 'POST'])
def manage_goals():
    """Get or create health goals"""
    user_id = get_user_id()
    profile = get_user_profile(user_id)
    
    if request.method == 'GET':
        return jsonify({
            'goals': profile['history'].get('goals', []),
            'success': True
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
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
            save_user_profile(user_id, profile)
            
            return jsonify({
                'message': 'Goal created successfully',
                'goal': goal,
                'success': True
            })
            
        except Exception as e:
            logging.error(f"Goal creation error: {e}")
            return jsonify({
                'error': 'Failed to create goal',
                'success': False
            }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get user analytics and insights"""
    try:
        user_id = get_user_id()
        profile = get_user_profile(user_id)
        
        # Calculate basic analytics
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
        
        return jsonify({
            'analytics': analytics,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Analytics error: {e}")
        return jsonify({
            'error': 'Failed to generate analytics',
            'success': False
        }), 500

# ======================= MAIN =======================

if __name__ == '__main__':
    print("=" * 60)
    print("🏃 SMART HEALTH COACH - BACKEND SERVER")
    print("=" * 60)
    print(f"✅ Status: {'Ready' if client else '⚠️  API Key Missing'}")
    print(f"🔑 Groq API: {'Configured' if api_key else 'Not configured'}")
    print(f"🌐 Server: http://localhost:5001")
    print("\n📋 Available Endpoints:")
    print("   POST   /api/chat                - Chat with health coach")
    print("   GET    /api/health              - Health check")
    print("   GET    /api/profile             - Get user profile")
    print("   POST   /api/profile             - Update profile")
    print("   GET    /api/metrics             - Get health metrics")
    print("   POST   /api/metrics             - Log health metrics")
    print("   POST   /api/workout/generate    - Generate workout plan")
    print("   POST   /api/meal-plan/generate  - Generate meal plan")
    print("   GET    /api/goals               - Get health goals")
    print("   POST   /api/goals               - Create new goal")
    print("   GET    /api/analytics           - Get user analytics")
    print("\n💡 Tips:")
    print("   - Set GROQ_API_KEY in .env file")
    print("   - Get free API key: https://console.groq.com/keys")
    print("   - Logs saved to: logs/ directory")
    print("   - User data saved to: user_data/ directory")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)