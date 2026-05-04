import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

print(df.head())

X = df["text"]
y = df["intent"]


vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

responses = {
    "fees": "Fee details are available in the accounts section.",
    "courses": "Various UG and PG courses are available.",
    "timetable": "Timetable is available on the website.",
    "exams": "Exam schedule will be announced soon.",
    "results": "Results will be published online.",
    "faculty_details": "Faculty details are available on the college portal.",
    "events": "Check the notice board for upcoming events.",
    "hostel_info": "Hostel facilities are available for students.",
    "placement_info": "Placement cell provides job opportunities.",
    "scholarships": "Scholarships are available based on eligibility."
}

def chatbot(msg):
    msg_vec = vectorizer.transform([msg])
    intent = model.predict(msg_vec)[0]
    return responses[intent]

# Test
print(chatbot("when is exam"))

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break
    print("Bot:", chatbot(user))
