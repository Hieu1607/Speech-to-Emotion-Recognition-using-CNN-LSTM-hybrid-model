# Speech-to-Emotion-Recognition-using-CNN-LSTM-hybrid-model

This project aims to build a speech emotion recognition system using a hybrid CNN-LSTM model. The model will be trained on a combination of several publicly available datasets to classify emotions from audio recordings.

## Project Structure

```
.
├── data
│   ├── processed
│   └── raw
│       ├── crema_d
│       ├── ravdess
│       ├── savee
│       └── tess
├── model
├── notebooks
│   ├── data_preprocess.ipynb
│   └── download_data.ipynb
├── src
│   ├── model
│   └── preprocess
│       └── preprocess_raw_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

-   **data/**: Contains the raw and processed datasets.
    -   `raw/`: Stores the original, untouched audio files from the datasets.
    -   `processed/`: Will store the processed data, such as extracted features, ready for model training.
-   **model/**: Will contain the trained model files.
-   **notebooks/**: Jupyter notebooks for data exploration, preprocessing, and model experimentation.
-   **src/**: Source code for the project.
    -   `model/`: Will contain the definition of the CNN-LSTM model.
    -   `preprocess/`: Scripts for data preprocessing.
-   **requirements.txt**: A list of the Python libraries required to run this project.

## Tech Stack

The project is primarily built using Python and the following libraries:

-   **Data Processing and Analysis**:
    -   `numpy`
    -   `pandas`
    -   `librosa` (for audio processing)
-   **Machine Learning and Deep Learning**:
    -   `scikit-learn`
    -   `tensorflow`
    -   `torch`
-   **Data Download**:
    -   `kaggle`
-   **Notebooks and Visualization**:
    -   `notebook`
    -   `ipython`
    -   `matplotlib`
    -   `seaborn`

## Datasets

This project uses a combination of the following popular emotional speech datasets:

-   [**Ravdess**](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio): The Ryerson Audio-Visual Database of Emotional Speech and Song.
-   [**Crema-D**](https://www.kaggle.com/datasets/ejlok1/cremad): The Crowd-sourced Emotional Multimodal Actors Dataset.
-   [**Tess**](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess): The Toronto Emotional Speech Set.
-   [**Savee**](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee): The Surrey Audio-Visual Expressed Emotion database.

## What We Did

-   **Data Acquisition**: Set up scripts to download the required datasets from Kaggle.
-   **Data Preprocessing**:
    -   Loaded audio files from the Ravdess, Crema-D, Tess, and Savee datasets.
    -   Extracted emotions from the file names.
    -   Combined the metadata from all datasets into a single DataFrame.

## What We Are Going to Do

-   **Feature Extraction**: Extract relevant audio features from the raw audio files. Common features for speech emotion recognition include MFCCs, Chroma, and Mel-spectrograms.
-   **Data Augmentation**: Apply data augmentation techniques to the audio data to increase the diversity of the training set and prevent overfitting.
-   **Model Building**: Implement the hybrid CNN-LSTM model architecture.
-   **Model Training**: Train the model on the preprocessed and augmented data.
-aluation**: Evaluate the model's performance on a held-out test set.
-   **Inference**: Build a simple application or script to perform emotion recognition on new audio files.
