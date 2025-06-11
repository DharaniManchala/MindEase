from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import json
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load all resources and models at startup
resources = {
    "academic": pd.read_csv('resources.csv'),
    "family": pd.read_csv('famresources.csv'),
    "illness": pd.read_csv('illnessresources.csv'),
    "work": pd.read_csv('workstressresources.csv'),
    "relationship": pd.read_csv('relationresources.csv'),
    "timemanagement": pd.read_csv('timemanagementresources.csv'),
    "other": pd.read_csv('otherresources.csv')
}
models = {
    "academic": joblib.load('stress_model.pkl'),
    "family": joblib.load('family_stress_model.pkl'),
    "illness": joblib.load('illness_stress_model.pkl'),
    "work": joblib.load('work_stress_model.pkl'),
    "relationship": joblib.load('relationship_stress_model.pkl'),
    "timemanagement": joblib.load('timemanagement_stress_model.pkl'),
    "other": joblib.load('other_stress_model.pkl')
}
rec_files = {
    "academic": 'recommandations.csv',
    "family": 'famrecommendation.csv',
    "illness": 'illnessrecommendation.csv',
    "work": 'workstressrecommendation.csv',
    "relationship": 'relationshiprecommendation.csv',
    "timemanagement": 'timemanagementworkloadrecommendation.csv',
    "other": 'otherrecommendation.csv'
}

def get_resources(level, stress_type):
    df = resources[stress_type]
    # Robust column mapping
    colmap = {
        'type': None,
        'title': None,
        'link': None,
        'description': None,
        'icon': None
    }
    for col in df.columns:
        lcol = col.lower()
        if 'type' in lcol and not colmap['type']:
            colmap['type'] = col
        elif 'title' in lcol and not colmap['title']:
            colmap['title'] = col
        elif 'link' in lcol and not colmap['link']:
            colmap['link'] = col
        elif 'desc' in lcol and not colmap['description']:
            colmap['description'] = col
        elif 'icon' in lcol and not colmap['icon']:
            colmap['icon'] = col
    for k in colmap:
        if not colmap[k]:
            colmap[k] = k.capitalize() if k != 'icon' else 'Icon'
    if 'StressLevel' in df.columns:
        filtered = df[df['StressLevel'] == level]
    else:
        filtered = df
    resources_list = []
    type_counts = {"Book": 0, "Video": 0, "Activity": 0}
    max_per_type = 1
    for _, row in filtered.iterrows():
        rtype = row.get(colmap['type'], '')
        if rtype in type_counts and type_counts[rtype] < max_per_type:
            resources_list.append({
                "icon": row.get(colmap['icon'], ''),
                "title": row.get(colmap['title'], ''),
                "link": row.get(colmap['link'], ''),
                "description": row.get(colmap['description'], ''),
                "type": rtype
            })
            type_counts[rtype] += 1
        if all(count == max_per_type for count in type_counts.values()):
            break
    return resources_list

def get_recommendation(user_answers, stress_type):
    rec_df = pd.read_csv(rec_files[stress_type])
    match = rec_df
    for i, ans in enumerate(user_answers):
        match = match[(match[f'Q{i+1}'] == ans) | (match[f'Q{i+1}'] == '*')]
    if not match.empty:
        row = match.iloc[0]
        return {
            "recommendation": row['Recommendation'],
            "rec_icon": row['RecIcon'],
            "effect": row['Effect'],
            "effect_icon": row['EffectIcon']
        }
    else:
        return {
            "recommendation": "Try to identify your main stressors and seek support.",
            "rec_icon": "📖",
            "effect": "Unmanaged stress can impact your well-being.",
            "effect_icon": "⚠️"
        }

@app.route('/predict-stress', methods=['POST'])
def predict_stress():
    data = request.json
    answers = data.get('answers')  # List of numbers
    email = data.get('email')
    stress_type = data.get('type', 'academic').lower()  # "academic" or "family"

    # Map expected answer length for each type
    expected_len_map = {
        "academic": 6,
        "family": 5,
        "illness": 6,
        "work": 6,
        "relationship": 6,
        "timemanagement": 6,
        "other": 6
    }
    expected_len = expected_len_map.get(stress_type, 6)
    if stress_type not in models:
        return jsonify({'error': 'Invalid stress type'}), 400
    if not answers or len(answers) != expected_len:
        return jsonify({'error': f'Invalid input: expected {expected_len} answers'}), 400

    # Robust mapping for all domains except academic/family
    mapping_options = {
        # 'illness': [
        #     ['Yes', 'No'],
        #     ['Never', 'Sometimes', 'Often', 'Always'],
        #     ['Not at all', 'Slightly', 'Moderately', 'Severely'],
        #     ['Yes', 'No', 'Occasionally'],
        #     ['Yes', 'No', 'Somewhat'],
        #     ['Very hopeful', 'Neutral', 'Not hopeful']
        # ],
            'illness': [
        ['No', 'Somewhat', 'Yes'],
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
        ['Not at all', 'Slightly', 'Moderately', 'Quite a bit', 'Severely'],
        ['Yes', 'Occasionally', 'No'],
        ['Yes', 'Somewhat', 'No'],
        ['Very hopeful', 'Neutral', 'Not hopeful']
    ],

           'work': [
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],      # Q1
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],      # Q2
        ['Very well', 'Well', 'Neutral', 'Poorly', 'Very poorly'],# Q3
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],      # Q4
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],      # Q5
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']       # Q6
    ],

    
       
            'relationship': [
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],           # Q1: Never = low stress, Always = high stress
        ['Confident', 'Neutral', 'Anxious', 'Very stressed'],          # Q2: Confident = low stress, Very stressed = high stress
        ['Low', 'Moderate', 'High'],                                   # Q3: Low = low stress, High = high stress
        ['Never', 'Occasionally', 'Sometimes', 'Frequently', 'Always'],# Q4: Never = low stress, Always = high stress
        ['Supported', 'Yes', 'No'],                                   # Q5: Supported = low stress, No = high stress
        ['Very well', 'Well', 'Neutral', 'Poorly', 'Very poorly']      # Q6: Very well = low stress, Very poorly = high stress
    ],

    
            'timemanagement': [
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],           # Q1
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],           # Q2
        ['Very well', 'Well', 'Neutral', 'Poorly', 'Very poorly'],     # Q3
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],           # Q4
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],           # Q5
        ['Very well', 'Well', 'Neutral', 'Poorly', 'Very poorly']      # Q6
    ],
        'other': [
        ['Never', 'Sometimes', 'Often', 'Always'],
        ['Yes', 'No', 'Prefer not to say'],
        ['Never', 'Occasionally', 'Frequently', 'Almost daily'],
        ['No', 'Sometimes', 'Often', 'Always'],
        ['Not at all', 'A little', 'Significantly', 'Extremely'],
        ['Yes', 'No', 'Maybe']
    ]
    }
    mapping_mapq = {
        
    'illness': [
        {'No':1, 'Somewhat':3, 'Yes':5},
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},
        {'Not at all':1, 'Slightly':2, 'Moderately':3, 'Quite a bit':4, 'Severely':5},
        {'Yes':1, 'Occasionally':3, 'No':5},
        {'Yes':1, 'Somewhat':3, 'No':5},
        {'Very hopeful':1, 'Neutral':3, 'Not hopeful':5}
    ],
    'work': [
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q1
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q2
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5},        # Q3
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q4
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q5
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5}               # Q6
    ],
        'relationship': [
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},                # Q1
        {'Confident':1, 'Neutral':3, 'Anxious':4, 'Very stressed':5},                 # Q2
        {'Low':1, 'Moderate':3, 'High':5},                                            # Q3
        {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Frequently':4, 'Always':5},     # Q4
        {'Supported':1, 'Yes':3, 'No':5},                                             # Q5
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}           # Q6
    ],
            'timemanagement': [
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q1
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q2
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5},        # Q3
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q4
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q5
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}         # Q6
    ],
            'other': [
        {'Never':1, 'Sometimes':2, 'Often':3, 'Always':5},                        # Q1
        {'Yes':1, 'No':3, 'Prefer not to say':5},                                 # Q2
        {'Never':1, 'Occasionally':2, 'Frequently':3, 'Almost daily':5},          # Q3
        {'No':1, 'Sometimes':2, 'Often':3, 'Always':5},                           # Q4
        {'Not at all':1, 'A little':2, 'Significantly':3, 'Extremely':5},         # Q5
        {'Yes':1, 'No':3, 'Maybe':5}                                              # Q6
    ]
    }
    # Map family answers to numbers before prediction
    if stress_type == 'family':
        family_map_q = [
            {'Always':1, 'Often':2, 'Sometimes':3, 'Rarely':4, 'Never':5},      # Q1: More support = lower stress
            {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},      # Q2: More disturbance = higher stress
            {'Yes':1, 'Not sure':3, 'No':5},                                    # Q3: Yes = low stress, No = high stress
            {'No':1, 'Not sure':3, 'Yes':5},                                    # Q4: No = low stress, Yes = high stress
            {'Very well':1, 'Somewhat':2, 'Not much':4, 'Not at all':5}         # Q5: Very well = low stress, Not at all = high stress
        ]
        answers_text = data.get('answers_text', [])
        if answers_text and len(answers_text) == 5:
            answers_for_model = [family_map_q[i].get(answers_text[i], 1) for i in range(5)]
        else:
            # fallback: try to map numbers to options (should not happen if frontend is correct)
            options = [
                ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'],
                ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
                ['Yes', 'Not sure', 'No'],
                ['No', 'Not sure', 'Yes'],
                ['Very well', 'Somewhat', 'Not much', 'Not at all']
            ]
            answers_for_model = [
                family_map_q[i].get(options[i][answers[i]-1], 1) if 1 <= answers[i] <= len(options[i]) else 1
                for i in range(5)
            ]
    else:
        if stress_type in mapping_options and all(isinstance(a, int) for a in answers):
            options = mapping_options[stress_type]
            map_q = mapping_mapq[stress_type]
            # Use answers_text from frontend if present and valid
            answers_text = data.get('answers_text')
            if answers_text and len(answers_text) == len(answers):
                # Use provided text directly
                answers_for_model = [map_q[i].get(answers_text[i], 1) for i in range(len(answers))]
            else:
                # Fallback: derive text from indices, but handle out-of-range safely
                answers_text = []
                for i, a in enumerate(answers):
                    if 1 <= a <= len(options[i]):
                        answers_text.append(options[i][a-1])
                    else:
                        answers_text.append(list(map_q[i].keys())[0])  # fallback to first option
                answers_for_model = [map_q[i].get(answers_text[i], 1) for i in range(len(answers))]
        else:
            answers_for_model = answers

    # Predict
    model = models[stress_type]
    # For family, use DataFrame with correct feature names as used in training (Q1n, Q2n, ...), else just a list
    if stress_type == 'family':
        family_feature_names = ['Q1n', 'Q2n', 'Q3n', 'Q4n', 'Q5n']
        X_input = pd.DataFrame([answers_for_model], columns=family_feature_names)
        pred = model.predict(X_input)[0]
    else:
        pred = model.predict([answers_for_model])[0]
   # percent = int(np.mean(answers_for_model) / 5 * 100)
    percent = int((np.mean(answers_for_model) - 1) / 4 * 100)

    # score = int(np.mean(answers_for_model) * 2)
    score = int((np.mean(answers_for_model) - 1) / 4 * 10)

    # Recommendation and resources
    recommendations = [get_recommendation(data.get('answers_text', []), stress_type)]
    effects = [{"icon": recommendations[0]["effect_icon"], "text": recommendations[0]["effect"]}]
    resources_list = get_resources(pred, stress_type)

    # Save email for high stress
    if pred == "High" and email:
        with open('high_stress_emails.json', 'a') as f:
            f.write(json.dumps({'email': email, 'date': str(datetime.now().date()), 'type': stress_type}) + '\n')

    # Save latest prediction info
    latest_prediction_data = {
        "score": score,
        "percent": percent,
        "level": pred,
        "recommendations": recommendations,
        "effects": effects,
        "resources": resources_list,
        "answers": answers,
        "timestamp": str(datetime.now())
    }
    # Save to a unique file for each stress type for debugging, but not used by frontend
    latest_json_map = {
        "academic": 'latest_prediction.json',
        "family": 'latest_family_prediction.json',
        "illness": 'latest_illness_prediction.json',
        "work": 'latest_work_prediction.json',
        "relationship": 'latest_relationship_prediction.json',
        "timemanagement": 'latest_timemanagement_prediction.json',
        "other": 'latest_other_prediction.json'
    }
    latest_json = latest_json_map.get(stress_type, 'latest_prediction.json')
    with open(latest_json, 'w') as f:
        json.dump(latest_prediction_data, f)

    return jsonify({
        "score": score,
        "percent": percent,
        "level": pred,
        "recommendations": recommendations,
        "effects": effects,
        "resources": resources_list,
        "answers": answers
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)