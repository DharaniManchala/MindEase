



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
# # Map user_answers to numbers for prediction
# answer_maps = [map_q1, map_q2, map_q3, map_q4, map_q5, map_q6]
# user_answers_num = [answer_maps[i][ans] for i, ans in enumerate(user_answers)]
# user_level = clf.predict([user_answers_num])[0]

# rec = get_academic_recommendation(user_answers)
# resources = get_resources_for_level(user_level)
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
# clf = DecisionTreeClassifier(max_depth=3)  # max_depth helps generalization
# clf.fit(X_train, y_train)

# # 8. Test accuracy
# accuracy = clf.score(X_test, y_test)
# print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# # 9. Predict for new answers (example)
# # Example: [Always, Anxious, Quite a bit, Sometimes, Supported, Poorly] -> [5, 4, 4, 3, 2, 4]
# new_answers = [[5, 4, 4, 3, 2, 4]]
# prediction = clf.predict(new_answers)
# print("\nPredicted Stress Level for new answers:", prediction[0])
# user_level = prediction[0]  # or however you get the user's stress level
# resources = get_resources_for_level(user_level)
# print("Resources for this level:", resources)

# # 10. Test model directly with all-High, all-Medium, all-Low
# print("Predict all High:", clf.predict([[5,5,5,5,5,5]]))    # Should be High
# print("Predict all Medium:", clf.predict([[3,3,3,3,3,3]]))  # Should be Medium
# print("Predict all Low:", clf.predict([[1,1,1,1,1,1]]))     # Should be Low
# print(df['StressLevel'].value_counts())

# # 11. Load the latest prediction from the JSON file for graphs
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

# # 12. Pie chart for the predicted stress percentage (latest user)
# # (Optional: Uncomment if you want to show the pie chart)
# # if latest_percent is not None:
# #     plt.figure(figsize=(4,4))
# #     plt.pie([latest_percent, 100-latest_percent], 
# #             labels=[f'Stress {latest_percent:.1f}%', ''], 
# #             startangle=90, 
# #             colors=['#f44336','#e0e0e0'], 
# #             wedgeprops={'width':0.3})
# #     plt.title('Stress Level (Last User)')
# #     plt.show()

# # 13. Time graph: Stress level over time
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', infer_datetime_format=True)
# color_map = {'Low':'#4caf50', 'Medium':'#ffeb3b', 'High':'#f44336'}

# # Remove rows with invalid/missing timestamps
# df_valid = df.dropna(subset=['Timestamp'])

# plt.figure(figsize=(10,4))
# plt.scatter(df_valid['Timestamp'], df_valid['Average'], c=df_valid['StressLevel'].map(color_map))
# plt.xlabel('Time')
# plt.ylabel('Average Score')
# plt.title('Stress Level Over Time')
# plt.show()

# # 14. Circle progress bar for a sample user's stress percentage (last row)
# last_avg = df['Average'].iloc[-1]
# percent = last_avg / 5 * 100

# plt.figure(figsize=(4,4))
# plt.pie([percent, 100-percent], labels=[f'Stress {percent:.1f}%', ''], startangle=90, colors=['#f44336','#e0e0e0'], wedgeprops={'width':0.3})
# plt.title('Stress Level (Last User)')
# plt.show()

# # 15. Gauge Chart (Speedometer) for Stress Percent (requires plotly)
# gauge_percent = percent  # Use the last user's percent, or set your own

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

# # 16. Dynamic Recommendation and Effect based on user's answers
# if latest_answers:
#     user_answers = latest_answers
# else:
#     user_answers = ['Always', 'Very Anxious', 'Extremely', 'Always', 'Not at all', 'Very poorly']

# rec = get_academic_recommendation(user_answers)
# resources = get_resources_for_level(user_level)

# # Now build and save the response
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

# # 17. Save the trained model to a file
# joblib.dump(clf, 'stress_model.pkl')
# print("Model saved as stress_model.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import joblib
import json
import plotly.graph_objects as go

# Load resources and recommendations CSVs
resources_df = pd.read_csv('resources.csv')
rec_df = pd.read_csv('recommandations.csv')

# 1. Load the answers CSV file (no header in your file)
df = pd.read_csv('analyze_stress_answers1.csv', header=None)
df.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Timestamp']

# 2. Map answers to numeric values
map_q1 = {'Never':1, 'Rarely':2, 'Sometimes':3, 'Often':4, 'Always':5}
map_q2 = {'Very Confident':1, 'Confident':2, 'Neutral':3, 'Anxious':4, 'Very Anxious':5}
map_q3 = {'Not at all':1, 'A little':2, 'Moderate':3, 'Quite a bit':4, 'Extremely':5}
map_q4 = {'Never':1, 'Occasionally':2, 'Sometimes':3, 'Frequently':4, 'Always':5}
map_q5 = {'Very supported':1, 'Supported':2, 'Neutral':3, 'Unsure':4, 'Not at all':5}
map_q6 = {'Very well':1, 'Well':2, 'Neutral':3, 'Poorly':4, 'Very poorly':5}

df['Q1n'] = df['Q1'].map(map_q1)
df['Q2n'] = df['Q2'].map(map_q2)
df['Q3n'] = df['Q3'].map(map_q3)
df['Q4n'] = df['Q4'].map(map_q4)
df['Q5n'] = df['Q5'].map(map_q5)
df['Q6n'] = df['Q6'].map(map_q6)

# 3. Calculate average score for each row
df['Average'] = df[['Q1n', 'Q2n', 'Q3n', 'Q4n', 'Q5n', 'Q6n']].mean(axis=1)

# 4. Assign stress level based on average
def stress_label(avg):
    if avg <= 2:
        return 'Low'
    elif avg < 4:
        return 'Medium'
    else:
        return 'High'

df['StressLevel'] = df['Average'].apply(stress_label)

print(df[['Q1','Q2','Q3','Q4','Q5','Q6','Average','StressLevel']].head())

# 5. Prepare features (X) and labels (y)
X = df[['Q1n', 'Q2n', 'Q3n', 'Q4n', 'Q5n', 'Q6n']]
y = df['StressLevel']

# Function to get recommendation/effect based on answers
def get_academic_recommendation(user_answers):
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
            "recommendation": "Try to identify your main academic stressors and seek support.",
            "rec_icon": "📖",
            "effect": "Unmanaged academic stress can impact your grades and well-being.",
            "effect_icon": "⚠️"
        }

# Function to get resources for a given stress level
def get_resources_for_level(level):
    filtered = resources_df[resources_df['StressLevel'] == level]
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

# 6. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# 8. Test accuracy
accuracy = clf.score(X_test, y_test)
print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# 9. Predict for new answers (example)
new_answers = [[5, 4, 4, 3, 2, 4]]
prediction = clf.predict(new_answers)
print("\nPredicted Stress Level for new answers:", prediction[0])
print("Predict all High:", clf.predict([[5,5,5,5,5,5]]))
print("Predict all Medium:", clf.predict([[3,3,3,3,3,3]]))
print("Predict all Low:", clf.predict([[1,1,1,1,1,1]]))
print(df['StressLevel'].value_counts())

# 10. Load the latest prediction from the JSON file for graphs
try:
    with open('latest_prediction.json', 'r') as f:
        latest = json.load(f)
    latest_level = latest['level']
    latest_percent = latest['percent']
    latest_answers = latest.get('answers', None)
except Exception as e:
    print(f"Could not load latest prediction data: {e}")
    latest_level = None
    latest_percent = None
    latest_answers = None

# 11. Visualization (optional, can comment out if not needed)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
color_map = {'Low':'#4caf50', 'Medium':'#ffeb3b', 'High':'#f44336'}
df_valid = df.dropna(subset=['Timestamp'])

plt.figure(figsize=(10,4))
plt.scatter(df_valid['Timestamp'], df_valid['Average'], c=df_valid['StressLevel'].map(color_map))
plt.xlabel('Time')
plt.ylabel('Average Score')
plt.title('Stress Level Over Time')
plt.show()

last_avg = df['Average'].iloc[-1]
percent = last_avg / 5 * 100

plt.figure(figsize=(4,4))
plt.pie([percent, 100-percent], labels=[f'Stress {percent:.1f}%', ''], startangle=90, colors=['#f44336','#e0e0e0'], wedgeprops={'width':0.3})
plt.title('Stress Level (Last User)')
plt.show()

gauge_percent = percent
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = gauge_percent,
    title = {'text': "Your Stress Level (%)"},
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

# 12. Dynamic Recommendation, Effect, and Resources for the latest user
# if latest_answers:
#     user_answers = latest_answers
# else:
#     user_answers = ['Always', 'Very Anxious', 'Extremely', 'Always', 'Not at all', 'Very poorly']

# # Clean up answers (remove extra spaces)
# user_answers = [ans.strip() for ans in user_answers]
if latest_answers and all(isinstance(ans, str) for ans in latest_answers):
    user_answers = [ans.strip() for ans in latest_answers]
else:
    user_answers = ['Always', 'Very Anxious', 'Extremely', 'Always', 'Not at all', 'Very poorly']

# Map user_answers to numbers for prediction
answer_maps = [map_q1, map_q2, map_q3, map_q4, map_q5, map_q6]
try:
    user_answers_num = [answer_maps[i][ans] for i, ans in enumerate(user_answers)]
except KeyError as e:
    print(f"KeyError mapping user_answers: {user_answers}")
    print(f"Mapping keys: {[list(m.keys()) for m in answer_maps]}")
    raise

user_level = clf.predict([user_answers_num])[0]

rec = get_academic_recommendation(user_answers)
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

with open('latest_prediction.json', 'w') as f:
    json.dump(response, f)

print(f"Recommendation: {rec['rec_icon']} {rec['recommendation']}")
print(f"Effect: {rec['effect_icon']} {rec['effect']}")
print(f"Resources for this level: {resources}")

# 13. Save the trained model to a file
joblib.dump(clf, 'stress_model.pkl')
print("Model saved as stress_model.pkl")