import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
from fer import FER
import numpy as np

# Global variables for model and vectorizer
vec = None
mod = None

def IMDBrev():
    global vec, mod
    try:
        # Load dataset
        data = pd.read_csv(r"C:\Users\ASHI\Desktop\Personal Info\Python\PROJECT\IMDB Dataset.csv")
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

        x = data['review']
        y = data['sentiment']

        # Train-test split
        xt, xtest, yt, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
        vec = TfidfVectorizer(stop_words='english', max_features=5000)
        Xt_tfidf = vec.fit_transform(xt)
        xtest_tfidf = vec.transform(xtest)

        # Train model
        mod = MultinomialNB()
        mod.fit(Xt_tfidf, yt)

        # Evaluate model
        ypred = mod.predict(Xt_tfidf)
        acc = accuracy_score(yt, ypred)
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(yt, ypred))

        # Plot precision scores
        rep = classification_report(yt, ypred, output_dict=True)
        classes = ['Negative', 'Positive']
        precision_scores = [rep['0']['precision'], rep['1']['precision']]

        plt.figure(figsize=(8, 5))
        plt.bar(classes, precision_scores, color=['blue', 'green'])
        plt.title('Precision for Each Class')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Precision Score')
        plt.ylim(0, 1)

        for i, v in enumerate(precision_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

        plt.show()

    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
        vec, mod = None, None

def facsentana():
    ed = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam not accessible. Exiting...")
            break

        em = ed.top_emotion(frame)
        if em is not None:
            emotion, score = em
            if emotion is not None and score is not None:
                cv2.putText(frame, f"{emotion} ({score:.2f})", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "No emotions detected", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No emotions detected", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Facial Sentiment Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predictext():
    global vec, mod

    # Check if model and vectorizer are initialized
    if vec is None or mod is None:
        print("Training model with IMDB data...Please Wait...")
        IMDBrev()
        if vec is None or mod is None:
            print("Failed to train the model. Exiting...")
            return

    sent = input("Enter a sentence to analyze: ")

    try:
        sent_tfidf = vec.transform([sent])
        prediction = mod.predict(sent_tfidf)[0]

        if prediction == 1:
            print("The Sentence is of Positive Note.")
        else:
            print("The Sentence is of Negative Note.")
    except Exception as e:
        print(f"Error during prediction: {e}")
# Main menu
print("Choose the type of sentiment analysis:")
print("Press 1 for IMDB Reviews Analysis")
print("Press 2 for Facial Sentiment Analysis")
print("Press 3 to Enter a Sentence for Sentiment Analysis")

try:
    ch = int(input("Enter Your Choice: "))

    if ch == 1:
        IMDBrev()
    elif ch == 2:
        print("Press 'x' to exit webcam.")
        facsentana()
    elif ch == 3:
        predictext()
    else:
        print("Invalid choice. Try entering 1, 2, or 3.")
except ValueError:
    print("Invalid input. Please enter a number (1, 2, or 3).")











    

    

