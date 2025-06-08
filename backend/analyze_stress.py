
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
# import joblib
# import json
# import plotly.graph_objects as go

# # Load resources and recommendations CSVs
# resources_df = pd.read_csv('resources.csv')
# rec_df = pd.read_csv('recommandations.csv')

# # 1. Load the answers CSV file (no header in your file)
# df = pd.read_csv('analyze_stress_answers1.csv', header=None)
# df.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp']

# # 2. Map answers to numeric values
# map_q1 = {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5}
# map_q2 = {'Very Confident':1, 'Confident':2, 'Neutral':3, 'Anxious':4, 'Very Anxious':5}
# map_q3 = {'Not at all':1, 'A little':2, 'Moderate':3, 'Quite a bit':4, 'Extremely':5}
# map_q4 = {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Frequently':4, 'Always':5}
# map_q5 = {'Very supported':1, 'Supported':2, 'Neutral':3, 'Unsure':4, 'Not at all':5}
# map_q6 = {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}

# df['Q1n'] = df['Q1'].map(map_q1)
# df['Q2n'] = df['Q2'].map(map_q2)
# df['Q3n'] = df['Q3'].map(map_q3)
# df['Q4n'] = df['Q4'].map(map_q4)
# df['Q5n'] = df['Q5'].map(map_q5)
# df['Q6n'] = df['Q6'].map(map_q6)

# # 3. Calculate average score for each row
# df['Average'] = df[['Q1n', 'Q2n', 'Q3n', 'Q4n', 'Q5n', 'Q6n']].mean(axis=1)

# # 4. Assign stress level based on average
# def stress_label(avg):
#     if avg <= 2:
#         return 'Low'
#     elif avg < 4:
#         return 'Medium'
#     else:
#         return 'High'

# df['StressLevel'] = df['Average'].apply(stress_label)

# print(df[['Q1','Q2','Q3','Q4','Q5','Q6','Average','StressLevel']].head())

# # 5. Prepare features (X) and labels (y)
# X = df[['Q1n', 'Q2n', 'Q3n', 'Q4n', 'Q5n', 'Q6n']]
# y = df['StressLevel']

# # Function to get recommendation/effect based on answers
# def get_academic_recommendation(user_answers):
#     match = rec_df
#     for i, ans in enumerate(user_answers):
#         match = match[(match[f'Q{i+1}'] == ans) | (match[f'Q{i+1}'] == '*')]
#     if not match.empty:
#         row = match.iloc[0]
#         return {
#             "recommendation": row['Recommendation'],
#             "rec_icon": row['RecIcon'],
#             "effect": row['Effect'],
#             "effect_icon": row['EffectIcon']
#         }
#     else:
#         return {
#             "recommendation": "Try to identify your main academic stressors and seek support.",
#             "rec_icon": "📖",
#             "effect": "Unmanaged academic stress can impact your grades and well-being.",
#             "effect_icon": "⚠️"
#         }

# # Function to get resources for a given stress level
# def get_resources_for_level(level):
#     filtered = resources_df[resources_df['StressLevel'] == level]
#     resources = []
#     for _, row in filtered.iterrows():
#         resources.append({
#             "type": row['ResourceType'],
#             "title": row['Title'],
#             "link": row['Link'],
#             "description": row['Description'],
#             "icon": row['Icon']
#         })
#     return resources

# # 6. Split into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 7. Train a Decision Tree Classifier
# clf = DecisionTreeClassifier(max_depth=3)
# clf.fit(X_train, y_train)

# # 8. Test accuracy
# accuracy = clf.score(X_test, y_test)
# print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# # 9. Predict for new answers (example)
# new_answers = [[5, 4, 4, 3, 2, 4]]
# prediction = clf.predict(new_answers)
# print("\nPredicted Stress Level for new answers:", prediction[0])
# print("Predict all High:", clf.predict([[5,5,5,5,5,5]]))
# print("Predict all Medium:", clf.predict([[3,3,3,3,3,3]]))
# print("Predict all Low:", clf.predict([[1,1,1,1,1,1]]))
# print(df['StressLevel'].value_counts())

# # 10. Load the latest prediction from the JSON file for graphs
# try:
#     with open('latest_prediction.json', 'r') as f:
#         latest = json.load(f)
#     latest_level = latest['level']
#     latest_percent = latest['percent']
#     latest_answers = latest.get('answers', None)
# except Exception as e:
#     print(f"Could not load latest prediction data: {e}")
#     latest_level = None
#     latest_percent = None
#     latest_answers = None

# # 11. Visualization (optional, can comment out if not needed)
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
# color_map = {'Low':'#4caf50', 'Medium':'#ffeb3b', 'High':'#f44336'}
# df_valid = df.dropna(subset=['Timestamp'])

# plt.figure(figsize=(10,4))
# plt.scatter(df_valid['Timestamp'], df_valid['Average'], c=df_valid['StressLevel'].map(color_map))
# plt.xlabel('Time')
# plt.ylabel('Average Score')
# plt.title('Stress Level Over Time')
# plt.show()

# last_avg = df['Average'].iloc[-1]
# percent = last_avg / 5 * 100

# plt.figure(figsize=(4,4))
# plt.pie([percent, 100-percent], labels=[f'Stress {percent:.1f}%', ''], startangle=90, colors=['#f44336','#e0e0e0'], wedgeprops={'width':0.3})
# plt.title('Stress Level (Last User)')
# plt.show()

# gauge_percent = percent
# fig = go.Figure(go.Indicator(
#     mode = "gauge+number",
#     value = gauge_percent,
#     title = {'text': "Your Stress Level (%)"},
#     gauge = {
#         'axis': {'range': [0, 100]},
#         'bar': {'color': "#f44336"},
#         'steps': [
#             {'range': [0, 40], 'color': "#4caf50"},
#             {'range': [40, 70], 'color': "#ffeb3b"},
#             {'range': [70, 100], 'color': "#f44336"}
#         ],
#         'threshold': {
#             'line': {'color': "black", 'width': 4},
#             'thickness': 0.75,
#             'value': gauge_percent}
#     }
# ))
# fig.show()


# if latest_answers and all(isinstance(ans, str) for ans in latest_answers):
#     user_answers = [ans.strip() for ans in latest_answers]
# else:
#     user_answers = ['Always', 'Very Anxious', 'Extremely', 'Always', 'Not at all', 'Very poorly']

# # Map user_answers to numbers for prediction
# answer_maps = [map_q1, map_q2, map_q3, map_q4, map_q5, map_q6]
# try:
#     user_answers_num = [answer_maps[i][ans] for i, ans in enumerate(user_answers)]
# except KeyError as e:
#     print(f"KeyError mapping user_answers: {user_answers}")
#     print(f"Mapping keys: {[list(m.keys()) for m in answer_maps]}")
#     raise

# user_level = clf.predict([user_answers_num])[0]

# rec = get_academic_recommendation(user_answers)
# resources = get_resources_for_level(user_level)

# # Build and save the response
# response = {
#     "level": user_level,
#     "percent": percent,
#     "recommendations": [rec],
#     "effects": [{"icon": rec["effect_icon"], "text": rec["effect"]}],
#     "resources": resources,
#     "answers": user_answers
# }

# with open('latest_prediction.json', 'w') as f:
#     json.dump(response, f)

# print(f"Recommendation: {rec['rec_icon']} {rec['recommendation']}")
# print(f"Effect: {rec['effect_icon']} {rec['effect']}")
# print(f"Resources for this level: {resources}")

# # 13. Save the trained model to a file
# joblib.dump(clf, 'stress_model.pkl')
# print("Model saved as stress_model.pkl")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import joblib
import json
import plotly.graph_objects as go

def train_and_test(
    answers_file, resources_file, rec_file, model_file, latest_json, 
    question_cols, map_q, n_questions, default_answers, label_func, mode_name
):
    print(f"\n--- {mode_name.upper()} STRESS TRAINING & TESTING ---")
    resources_df = pd.read_csv(resources_file)
    rec_df = pd.read_csv(rec_file)
    df = pd.read_csv(answers_file, header=None)
    df.columns = question_cols

    # Map answers to numbers
    for i in range(n_questions):
        df[f'Q{i+1}n'] = df[f'Q{i+1}'].map(map_q[i])

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
        if 'StressLevel' in resources_df.columns:
            filtered = resources_df[resources_df['StressLevel'] == level]
        else:
            filtered = resources_df
        resources = []
        for _, row in filtered.iterrows():
            resources.append({
                "type": row['ResourceType'],
                "title": row['Title'],
                "link": row['Link'],
                "description": row['Description'],
                "icon": row['Icon']
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
    label_func=lambda avg: 'Low' if avg <= 2 else ('Medium' if avg < 4 else 'High'),
    mode_name="academic"
)

# --------- Family Stress ---------
train_and_test(
    answers_file='famstress.csv',
    resources_file='famresources.csv',
    rec_file='famrecommendation.csv',
    model_file='family_stress_model.pkl',
    latest_json='latest_family_prediction.json',
    question_cols=['Q1', 'Q2', 'Q3', 'Q4', 'Q5','StressLevel'],
    map_q=[
        {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5},
        {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Often':4, 'Always':5},
        {'No':1, 'Yes':2, 'Not sure':3},
        {'No':1, 'Yes':2, 'Not sure':3},
        {'Very well':1, 'Somewhat':2, 'Not much':3, 'Not at all':4}
    ],
    n_questions=5,
    default_answers=['Always', 'Often', 'Yes', 'Yes', 'Somewhat'],
    label_func=lambda avg: 'Low' if avg <= 2 else ('Medium' if avg < 3.5 else 'High'),
    mode_name="family"
)