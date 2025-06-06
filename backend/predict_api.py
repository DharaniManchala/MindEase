
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# from flask_cors import CORS
# import json
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # Load your trained model (update the filename if needed)
# model = joblib.load('stress_model.pkl')
# def get_recommendations(level):
#     if level == "High":
#         return [
#             {"icon": "💧", "text": "Take regular breaks and hydrate."},
#             {"icon": "🧘", "text": "Practice deep breathing or meditation."},
#             {"icon": "📅", "text": "Organize your study schedule."},
#             {"icon": "🤝", "text": "Talk to a mentor or counselor."},
#             {"icon": "🚶", "text": "Go for a short walk to clear your mind."}
#         ]
#     elif level == "Medium":
#         return [
#             {"icon": "📋", "text": "Make a to-do list for your tasks."},
#             {"icon": "⏰", "text": "Set small, achievable goals."},
#             {"icon": "🎵", "text": "Listen to relaxing music."},
#             {"icon": "📞", "text": "Stay connected with friends."},
#             {"icon": "🍎", "text": "Eat healthy snacks."}
#         ]
#     else:  # Low
#         return [
#             {"icon": "👍", "text": "Keep up the good work!"},
#             {"icon": "🏃", "text": "Maintain your healthy habits."},
#             {"icon": "📚", "text": "Share your strategies with peers."},
#             {"icon": "🌳", "text": "Spend time outdoors."},
#             {"icon": "😊", "text": "Celebrate your achievements."}
#         ]

# def get_effects(level):
#     if level == "High":
#         return [
#             {"icon": "😫", "text": "May cause burnout or anxiety."},
#             {"icon": "😴", "text": "Can affect sleep quality."},
#             {"icon": "🤒", "text": "Weakens immune system."},
#             {"icon": "📉", "text": "Reduces academic performance."},
#             {"icon": "😔", "text": "Impacts mood and motivation."}
#         ]
#     elif level == "Medium":
#         return [
#             {"icon": "😕", "text": "Might feel distracted."},
#             {"icon": "😐", "text": "Occasional tiredness."},
#             {"icon": "📉", "text": "Slight drop in focus."},
#             {"icon": "😟", "text": "Mild worry or stress."},
#             {"icon": "🍫", "text": "Possible unhealthy snacking."}
#         ]
#     else:  # Low
#         return [
#             {"icon": "😃", "text": "You are managing stress well!"},
#             {"icon": "💪", "text": "Good energy and focus."},
#             {"icon": "🛌", "text": "Better sleep quality."},
#             {"icon": "📈", "text": "Improved academic results."},
#             {"icon": "😊", "text": "Positive mood and motivation."}
#         ]
# @app.route('/predict-stress', methods=['POST'])
# def predict_stress():
#     data = request.json
#     answers = data.get('answers')  # Should be a list of numbers, e.g. [5,2,4,3,4,2]
#     email = data.get('email')
#     if not answers or len(answers) != 6:
#         return jsonify({'error': 'Invalid input'}), 400

#     # Predict
#     pred = model.predict([answers])[0]
#     percent = int(np.mean(answers) / 5 * 100)
#     score = int(np.mean(answers) * 2)  # Example: scale to 10

#     # Example recommendations and effects
#     # recommendations = [
#     #     {"icon": "💤", "text": "Get enough sleep"},
#     #     {"icon": "🌞", "text": "Spend time outdoors"},
#     #     {"icon": "😌", "text": "Practice deep breathing exercises"}
#     # ]
#     # effects = [
#     #     {"icon": "📉", "text": "Decrease in productivity and concentration"},
#     #     {"icon": "⚠", "text": "Increased risk of health issues"}
#     # ]
#     recommendations = get_recommendations(pred)
#     effects = get_effects(pred)

#     # Optionally save high stress emails for future use
#     if pred == "High" and email:
#         with open('high_stress_emails.json', 'a') as f:
#             f.write(json.dumps({'email': email, 'date': str(datetime.now().date())}) + '\n')

#     return jsonify({
#         "score": score,
#         "percent": percent,
#         "level": pred,
#         "recommendations": recommendations,
#         "effects": effects,
#         "answers": answers  # Optionally return answers for graphing
#     })

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)
import pandas as pd

resources_df = pd.read_csv('resources.csv')

# def get_resources(level):
#     filtered = resources_df[resources_df['StressLevel'] == level]
#     resources = []
#     for _, row in filtered.iterrows():
#         resources.append({
#             "icon": row['Icon'],
#             "title": row['Title'],
#             "link": row['Link'],
#             "description": row['Description'],
#             "type": row['ResourceType']
#         })
#     return resources
def get_resources(level):
    filtered = resources_df[resources_df['StressLevel'] == level]
    resources = []
    type_counts = {"Book": 0, "Video": 0, "Activity": 0}
    max_per_type = 1

    for _, row in filtered.iterrows():
        rtype = row['ResourceType']
        if rtype in type_counts and type_counts[rtype] < max_per_type:
            resources.append({
                "icon": row['Icon'],
                "title": row['Title'],
                "link": row['Link'],
                "description": row['Description'],
                "type": rtype
            })
            type_counts[rtype] += 1
        # Stop if we have one of each
        if all(count == max_per_type for count in type_counts.values()):
            break
    return resources

# Load your trained model (update the filename if needed)
model = joblib.load('stress_model.pkl')

def get_recommendations(level):
    if level == "High":
        return [
            {"icon": "💧", "text": "Take regular breaks and hydrate."},
            {"icon": "🧘", "text": "Practice deep breathing or meditation."},
            {"icon": "📅", "text": "Organize your study schedule."},
            {"icon": "🤝", "text": "Talk to a mentor or counselor."},
            {"icon": "🚶", "text": "Go for a short walk to clear your mind."}
        ]
    elif level == "Medium":
        return [
            {"icon": "📋", "text": "Make a to-do list for your tasks."},
            {"icon": "⏰", "text": "Set small, achievable goals."},
            {"icon": "🎵", "text": "Listen to relaxing music."},
            {"icon": "📞", "text": "Stay connected with friends."},
            {"icon": "🍎", "text": "Eat healthy snacks."}
        ]
    else:  # Low
        return [
            {"icon": "👍", "text": "Keep up the good work!"},
            {"icon": "🏃", "text": "Maintain your healthy habits."},
            {"icon": "📚", "text": "Share your strategies with peers."},
            {"icon": "🌳", "text": "Spend time outdoors."},
            {"icon": "😊", "text": "Celebrate your achievements."}
        ]

def get_effects(level):
    if level == "High":
        return [
            {"icon": "😫", "text": "May cause burnout or anxiety."},
            {"icon": "😴", "text": "Can affect sleep quality."},
            {"icon": "🤒", "text": "Weakens immune system."},
            {"icon": "📉", "text": "Reduces academic performance."},
            {"icon": "😔", "text": "Impacts mood and motivation."}
        ]
    elif level == "Medium":
        return [
            {"icon": "😕", "text": "Might feel distracted."},
            {"icon": "😐", "text": "Occasional tiredness."},
            {"icon": "📉", "text": "Slight drop in focus."},
            {"icon": "😟", "text": "Mild worry or stress."},
            {"icon": "🍫", "text": "Possible unhealthy snacking."}
        ]
    else:  # Low
        return [
            {"icon": "😃", "text": "You are managing stress well!"},
            {"icon": "💪", "text": "Good energy and focus."},
            {"icon": "🛌", "text": "Better sleep quality."},
            {"icon": "📈", "text": "Improved academic results."},
            {"icon": "😊", "text": "Positive mood and motivation."}
        ]

@app.route('/predict-stress', methods=['POST'])
def predict_stress():
    data = request.json
    answers = data.get('answers')  # Should be a list of numbers, e.g. [5,2,4,3,4,2]
    email = data.get('email')
    if not answers or len(answers) != 6:
        return jsonify({'error': 'Invalid input'}), 400

    # Predict stress level
    pred = model.predict([answers])[0]
    percent = int(np.mean(answers) / 5 * 100)
    score = int(np.mean(answers) * 2)  # Example: scale to 10

    recommendations = get_recommendations(pred)
    effects = get_effects(pred)
    resources = get_resources(pred)

    # Save email for high stress cases
    if pred == "High" and email:
        with open('high_stress_emails.json', 'a') as f:
            f.write(json.dumps({'email': email, 'date': str(datetime.now().date())}) + '\n')

    # Save latest prediction info to a file for frontend use
    latest_prediction_data = {
        "score": score,
        "percent": percent,
        "level": pred,
        "recommendations": recommendations,
        "effects": effects,
        "resources": resources,
        "answers": answers,
        "timestamp": str(datetime.now())
    }

    with open('latest_prediction.json', 'w') as f:
        json.dump(latest_prediction_data, f)

    return jsonify({
        "score": score,
        "percent": percent,
        "level": pred,
        "recommendations": recommendations,
        "effects": effects,
        "resources": resources,
        "answers": answers  # Optionally return answers for graphing
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
