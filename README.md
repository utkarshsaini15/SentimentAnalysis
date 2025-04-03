# Sentiment Analysis Project

This project provides a multi-functional sentiment analysis tool that includes:

1. **IMDB Reviews Sentiment Analysis**: Analyzing the sentiment of a dataset of movie reviews.
2. **Facial Sentiment Analysis**: Detecting emotions using a webcam feed.
3. **Custom Text Sentiment Prediction**: Allowing users to input a sentence and predicting its sentiment.

## Features

### IMDB Reviews Sentiment Analysis

- Loads and processes an IMDB movie reviews dataset.
- Converts text reviews into numerical data using **TF-IDF Vectorization**.
- Trains a **Naive Bayes Classifier** for sentiment classification.
- Outputs model accuracy and a classification report.
- Visualizes precision scores for positive and negative sentiment classes.

### Facial Sentiment Analysis

- Detects facial emotions in real-time using a webcam.
- Uses the **FER (Facial Expression Recognition)** library for emotion detection.
- Displays the detected emotion and confidence score on the video feed.

### Custom Text Sentiment Prediction

- Allows users to input custom text for sentiment analysis.
- Uses the pre-trained Naive Bayes model to classify the text as positive or negative.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.7 or later
- pip (Python package manager)
- Webcam (for Facial Sentiment Analysis)

### Required Libraries

Install the required libraries by running the following command:

```bash
pip install pandas scikit-learn matplotlib opencv-python-headless fer numpy
```

## File Structure

- **IMDB Dataset**: Ensure the IMDB dataset (`IMDB Dataset.csv`) is available at the specified path in the code.
- **Python Script**: Contains the main code for sentiment analysis.

## Usage

1. **Run the Script**

   ```bash
   python sentiment_analysis.py
   ```

2. **Select an Option**

   - Press `1` for IMDB Reviews Analysis.
   - Press `2` for Facial Sentiment Analysis. Press `x` to exit the webcam feed.
   - Press `3` to analyze the sentiment of a custom sentence.

### Dataset

For IMDB Reviews Sentiment Analysis, ensure you have the dataset file (`IMDB Dataset.csv`) in the correct path. Update the path in the script if needed:

```python
data = pd.read_csv(r"C:\Users\ASHI\Desktop\Personal Info\Python\PROJECT\IMDB Dataset.csv")
```

### Output

- IMDB Reviews Analysis: Displays accuracy, classification report, and a bar chart of precision scores.
- Facial Sentiment Analysis: Real-time emotion detection displayed on a webcam feed.
- Custom Text Analysis: Outputs whether the input text is positive or negative.

## Troubleshooting

- **FileNotFoundError**: Ensure the IMDB dataset path is correct.
- **Webcam Issues**: Check if your webcam is functional and accessible.
- **Missing Libraries**: Install missing libraries using pip.

## Example

### IMDB Reviews Sentiment Analysis

```bash
Choose the type of sentiment analysis:
Press 1 for IMDB Reviews Analysis
Press 2 for Facial Sentiment Analysis
Press 3 to Enter a Sentence for Sentiment Analysis
Enter Your Choice: 1

Accuracy: 0.91
Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.93      0.91      1000
           1       0.94      0.90      0.92      1000

   accuracy                           0.91      2000
  macro avg       0.91      0.91      0.91      2000
weighted avg       0.91      0.91      0.91      2000
```

### Facial Sentiment Analysis

- The detected emotion (e.g., `Happy (0.85)`) is displayed on the webcam feed.

### Custom Text Sentiment Prediction

```bash
Enter a sentence to analyze: The movie was fantastic!
The Sentence is of Positive Note.
```

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Acknowledgments

- **FER Library** for facial emotion detection.
- **Scikit-learn** for machine learning utilities.
- **Matplotlib** for visualization.
- **IMDB Dataset** for providing labeled movie reviews.


