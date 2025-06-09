import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import joblib
import json
import plotly.graph_objects as go
import os

def train_and_test(
    answers_file, resources_file, rec_file, model_file, latest_json, 
    question_cols, map_q, n_questions, default_answers, label_func, mode_name
):
    print(f"\n--- {mode_name.upper()} STRESS TRAINING & TESTING ---")
    
    # Ensure all file paths are relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    answers_file = os.path.join(base_dir, answers_file)
    resources_file = os.path.join(base_dir, resources_file)
    rec_file = os.path.join(base_dir, rec_file)
    model_file = os.path.join(base_dir, model_file)
    latest_json = os.path.join(base_dir, latest_json)
    
    resources_df = pd.read_csv(resources_file)
    rec_df = pd.read_csv(rec_file)
    df = pd.read_csv(answers_file, header=None)
    # If CSV has extra StressLevel column, drop it before assigning columns
    # If CSV has header, skip header row and set columns accordingly
    if df.shape[1] == len(question_cols) + 1:
        # Assume last column is StressLevel, drop it
        df = df.iloc[:, :len(question_cols)]
        df.columns = question_cols
    elif df.shape[1] == len(question_cols):
        df.columns = question_cols
    else:
        raise ValueError(f"Length mismatch: CSV has {df.shape[1]} columns, expected {len(question_cols)} (questions only) or {len(question_cols)+1} (with StressLevel)")
    # Map answers to numbers (LOGICAL, EXPLICIT, PER-QUESTION MAPPING)
    for i in range(n_questions):
        if mode_name == 'family':
            if i == 0:
                df[f'Q{i+1}n'] = df[f'Q{i+1}'].map({'Never':5, 'Rarely':4, 'Sometimes':3, 'Often':2, 'Always':1})
            elif i == 1:
                df[f'Q{i+1}n'] = df[f'Q{i+1}'].map({'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5})
            elif i == 2:
                df[f'Q{i+1}n'] = df[f'Q{i+1}'].map({'No':5, 'Not sure':3, 'Yes':1})
            elif i == 3:
                df[f'Q{i+1}n'] = df[f'Q{i+1}'].map({'No':1, 'Not sure':3, 'Yes':5})
            elif i == 4:
                df[f'Q{i+1}n'] = df[f'Q{i+1}'].map({'Very well':1, 'Somewhat':2, 'Not much':4, 'Not at all':5})
        else:
            df[f'Q{i+1}n'] = df[f'Q{i+1}'].map(map_q[i])

    # If Timestamp column missing, add dummy timestamps for plotting
    if 'Timestamp' not in df.columns:
        import datetime
        base = datetime.datetime.today()
        df['Timestamp'] = [base - datetime.timedelta(days=x) for x in range(len(df))][::-1]

    # Calculate average & assign stress level
    df['Average'] = df[[f'Q{i+1}n' for i in range(n_questions)]].mean(axis=1)
    df['StressLevel'] = df['Average'].apply(label_func)

    print(df[[*question_cols, 'Average', 'StressLevel']].head())

    X = df[[f'Q{i+1}n' for i in range(n_questions)]]
    y = df['StressLevel']

    # Recommendation function
    def get_recommendation(user_answers):
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

    def get_resources_for_level(level):
        # Handle possible column name variations and missing columns gracefully
        colmap = {
            'type': None,
            'title': None,
            'link': None,
            'description': None,
            'icon': None
        }
        # Try to find the best match for each expected column
        for col in resources_df.columns:
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
        # Fallbacks if not found
        for k in colmap:
            if not colmap[k]:
                colmap[k] = k.capitalize() if k != 'icon' else 'Icon'
        if 'StressLevel' in resources_df.columns:
            filtered = resources_df[resources_df['StressLevel'] == level]
        else:
            filtered = resources_df
        resources = []
        for _, row in filtered.iterrows():
            resources.append({
                "type": row.get(colmap['type'], ''),
                "title": row.get(colmap['title'], ''),
                "link": row.get(colmap['link'], ''),
                "description": row.get(colmap['description'], ''),
                "icon": row.get(colmap['icon'], '')
            })
        return resources

    # Train/test split & model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Model Test Accuracy: {accuracy*100:.2f}%")

    # Predict for new answers (example)
    print("Predict all High:", clf.predict([[5]*n_questions]))
    print("Predict all Medium:", clf.predict([[3]*n_questions]))
    print("Predict all Low:", clf.predict([[1]*n_questions]))
    print(df['StressLevel'].value_counts())

    # Visualization (optional)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        color_map = {'Low':'#4caf50', 'Medium':'#ffeb3b', 'High':'#f44336'}
        df_valid = df.dropna(subset=['Timestamp'])
        plt.figure(figsize=(10,4))
        plt.scatter(df_valid['Timestamp'], df_valid['Average'], c=df_valid['StressLevel'].map(color_map))
        plt.xlabel('Time')
        plt.ylabel('Average Score')
        plt.title(f'{mode_name.capitalize()} Stress Level Over Time')
        plt.show()

    last_avg = df['Average'].iloc[-1]
    percent = last_avg / 5 * 100

    plt.figure(figsize=(4,4))
    plt.pie([percent, 100-percent], labels=[f'Stress {percent:.1f}%', ''], startangle=90, colors=['#f44336','#e0e0e0'], wedgeprops={'width':0.3})
    plt.title(f'{mode_name.capitalize()} Stress Level (Last User)')
    plt.show()

    gauge_percent = percent
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gauge_percent,
        title = {'text': f"{mode_name.capitalize()} Stress Level (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#f44336"},
            'steps': [
                {'range': [0, 40], 'color': "#4caf50"},
                {'range': [40, 70], 'color': "#ffeb3b"},
                {'range': [70, 100], 'color': "#f44336"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_percent}
        }
    ))
    fig.show()

    # Get latest answers (simulate or from JSON)
    try:
        with open(latest_json, 'r') as f:
            latest = json.load(f)
        latest_level = latest.get('level')
        latest_percent = latest.get('percent')
        latest_answers = latest.get('answers', None)
    except Exception as e:
        latest_level = None
        latest_percent = None
        latest_answers = None

    if latest_answers and all(isinstance(ans, str) for ans in latest_answers):
        user_answers = [ans.strip() for ans in latest_answers]
    else:
        user_answers = default_answers

    # Map user_answers to numbers for prediction
    try:
        user_answers_num = [map_q[i][ans] for i, ans in enumerate(user_answers)]
    except KeyError as e:
        print(f"KeyError mapping user_answers: {user_answers}")
        print(f"Mapping keys: {[list(m.keys()) for m in map_q]}")
        raise

    user_level = clf.predict([user_answers_num])[0]

    rec = get_recommendation(user_answers)
    resources = get_resources_for_level(user_level)

    # Build and save the response
    response = {
        "level": user_level,
        "percent": percent,
        "recommendations": [rec],
        "effects": [{"icon": rec["effect_icon"], "text": rec["effect"]}],
        "resources": resources,
        "answers": user_answers
    }

    with open(latest_json, 'w') as f:
        json.dump(response, f)

    print(f"Recommendation: {rec['rec_icon']} {rec['recommendation']}")
    print(f"Effect: {rec['effect_icon']} {rec['effect']}")
    print(f"Resources for this level: {resources}")

    # Save the trained model to a file
    joblib.dump(clf, model_file)
    print(f"Model saved as {model_file}")

# --------- Academic Stress ---------
train_and_test(
    answers_file='analyze_stress_answers1.csv',
    resources_file='resources.csv',
    rec_file='recommandations.csv',
    model_file='stress_model.pkl',
    latest_json='latest_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},
        {'Very Confident':1, 'Confident':2, 'Neutral':3, 'Anxious':4, 'Very Anxious':5},
        {'Not at all':1, 'A little':2, 'Moderate':3, 'Quite a bit':4, 'Extremely':5},
        {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Frequently':4, 'Always':5},
        {'Very supported':1, 'Supported':2, 'Neutral':3, 'Unsure':4, 'Not at all':5},
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}
    ],
    n_questions=6,
    default_answers=['Always', 'Very Anxious', 'Extremely', 'Always', 'Not at all', 'Very poorly'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="academic"
)

# # --------- Family Stress ---------
# train_and_test(
#     answers_file='famstress.csv',
#     resources_file='famresources.csv',
#     rec_file='famrecommendation.csv',
#     model_file='family_stress_model.pkl',
#     latest_json='latest_family_prediction.json',
#     question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],  # No Timestamp or StressLevel in CSV
#     map_q=[
#         {'Never':5, 'Rarely':4, 'Sometimes':3, 'Often':2, 'Always':1},  # Q1: More support = lower stress
#         {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},  # Q2: More arguments = higher stress
#         {'No':5, 'Not sure':3, 'Yes':1},  # Q3: 'No' = high stress, 'Yes' = low stress
#         {'No':1, 'Not sure':3, 'Yes':5},  # Q4: 'No' = low stress, 'Yes' = high stress
#         {'Very well':1, 'Somewhat':2, 'Not much':4, 'Not at all':5}    # Q5: 'Very well' = low stress, 'Not at all' = high stress
#     ],
#     n_questions=5,
#     default_answers=['Never', 'Always', 'No', 'Yes', 'Not at all'],
#     label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
#     mode_name="family"
# )
# --------- Family Stress ---------
train_and_test(
    answers_file='famstress.csv',
    resources_file='famresources.csv',
    rec_file='famrecommendation.csv',
    model_file='family_stress_model.pkl',
    latest_json='latest_family_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],  # No Timestamp or StressLevel in CSV
    map_q=[
        {'Always':1, 'Often':2, 'Sometimes':3, 'Rarely':4, 'Never':5},      # Q1: More support = lower stress
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},      # Q2: More disturbance = higher stress
        {'Yes':1, 'Not sure':3, 'No':5},                                    # Q3: Yes = low stress, No = high stress
        {'No':1, 'Not sure':3, 'Yes':5},                                    # Q4: No = low stress, Yes = high stress
        {'Very well':1, 'Somewhat':2, 'Not much':4, 'Not at all':5}         # Q5: Very well = low stress, Not at all = high stress
    ],
    n_questions=5,
    default_answers=['Always', 'Always', 'Yes', 'No', 'Very well'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="family"
)

# # --------- Illness Stress ---------
# train_and_test(
#     answers_file='illnessstress.csv',
#     resources_file='illnessresources.csv',
#     rec_file='illnessrecommendation.csv',
#     model_file='illness_stress_model.pkl',
#     latest_json='latest_illness_prediction.json',
#     question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
#     map_q=[
#         {'No': 1, 'Somewhat': 3, 'Yes': 5},  # Q1: Yes = 5 (high stress), No = 1 (low)
#         {'Never': 1, 'Sometimes': 3, 'Often': 4, 'Always': 5},  # Q2: Always = 5
#         {'Not at all': 1, 'Slightly': 2, 'Moderately': 3, 'Severely': 5},  # Q3: Severely = 5
#         {'Yes': 1, 'Occasionally': 3, 'No': 5},  # Q4: No = 5 (high stress), Yes = 1 (low)
#         {'Yes': 1, 'Somewhat': 3, 'No': 5},  # Q5: No = 5 (high stress), Yes = 1 (low)
#         {'Very hopeful': 1, 'Neutral': 3, 'Not hopeful': 5}  # Q6: Not hopeful = 5
#     ],
#     n_questions=6,
#     default_answers=['Yes', 'Always', 'Severely', 'No', 'No', 'Not hopeful'],
#     label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
#     mode_name="illness"
# )

# --------- Illness Stress ---------
train_and_test(
    answers_file='illnessstress.csv',
    resources_file='illnessresources.csv',
    rec_file='illnessrecommendation.csv',
    model_file='illness_stress_model.pkl',
    latest_json='latest_illness_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'No': 1, 'Somewhat': 3, 'Yes': 5},  # Q1: No illness = low stress, Yes = high stress
        {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},  # Q2: Never interferes = low stress, Always = high stress
        {'Not at all': 1, 'Slightly': 2, 'Moderately': 3, 'Quite a bit': 4, 'Severely': 5},  # Q3: Not at all = low stress, Severely = high stress
        {'Yes': 1, 'Occasionally': 3, 'No': 5},  # Q4: Yes = low stress, No = high stress
        {'Yes': 1, 'Somewhat': 3, 'No': 5},  # Q5: Yes = low stress, No = high stress
        {'Very hopeful': 1, 'Neutral': 3, 'Not hopeful': 5}  # Q6: Very hopeful = low stress, Not hopeful = high stress
    ],
    n_questions=6,
    default_answers=['No', 'Never', 'Not at all', 'Yes', 'Yes', 'Very hopeful'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="illness"
)

# --------- Work Stress ---------
# --------- Work Stress ---------
train_and_test(
    answers_file='workstress.csv',
    resources_file='workstressresources.csv',
    rec_file='workstressrecommendation.csv',
    model_file='work_stress_model.pkl',
    latest_json='latest_work_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q1
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q2
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5},        # Q3
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q4
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q5
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5}               # Q6
    ],
    n_questions=6,
    default_answers=['Never', 'Never', 'Very well', 'Never', 'Never', 'Never'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="work"
)

# --------- Relationship Stress ---------
train_and_test(
    answers_file='relationstress.csv',
    resources_file='relationresources.csv',
    rec_file='relationshiprecommendation.csv',
    model_file='relationship_stress_model.pkl',
    latest_json='latest_relationship_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},                # Q1: Never = low stress, Always = high stress
        {'Confident':1, 'Neutral':3, 'Anxious':4, 'Very stressed':5},                 # Q2: Confident = low stress, Very stressed = high stress
        {'Low':1, 'Moderate':3, 'High':5},                                            # Q3: Low = low stress, High = high stress
        {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Frequently':4, 'Always':5},     # Q4: Never = low stress, Always = high stress
        {'Supported':1, 'Yes':3, 'No':5},                                             # Q5: Supported = low stress, No = high stress
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}           # Q6: Very well = low stress, Very poorly = high stress
    ],
    n_questions=6,
    default_answers=['Never', 'Confident', 'Low', 'Never', 'Supported', 'Very well'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="relationship"
)
# --------- Time Management Stress ---------
train_and_test(
    answers_file='timemanagestress.csv',
    resources_file='timemanagementresources.csv',
    rec_file='timemanagementworkloadrecommendation.csv',
    model_file='timemanagement_stress_model.pkl',
    latest_json='latest_timemanagement_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q1
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q2
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5},        # Q3
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q4
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},              # Q5
        {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}         # Q6
    ],
    n_questions=6,
    default_answers=['Never', 'Never', 'Very well', 'Never', 'Never', 'Very well'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="timemanagement"
)
# --------- Other Stress ---------
train_and_test(
    answers_file='othersstress.csv',
    resources_file='otherresources.csv',
    rec_file='otherrecommendation.csv',
    model_file='other_stress_model.pkl',
    latest_json='latest_other_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp'],
    map_q=[
        {'Never':1, 'Sometimes':2, 'Often':3, 'Always':5},                        # Q1
        {'Yes':1, 'No':3, 'Prefer not to say':5},                                 # Q2
        {'Never':1, 'Occasionally':2, 'Frequently':3, 'Almost daily':5},          # Q3
        {'No':1, 'Sometimes':2, 'Often':3, 'Always':5},                           # Q4
        {'Not at all':1, 'A little':2, 'Significantly':3, 'Extremely':5},         # Q5
        {'Yes':1, 'No':3, 'Maybe':5}                                              # Q6
    ],
    n_questions=6,
    default_answers=['Never', 'Yes', 'Never', 'No', 'Not at all', 'Yes'],
    label_func=lambda avg: 'High' if avg >= 4 else ('Medium' if avg >= 2.5 else 'Low'),
    mode_name="other"
)