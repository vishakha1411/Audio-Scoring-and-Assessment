## **Speech Quality Scoring System** 🎵📊

A machine learning pipeline for automatically scoring speech audio quality (1.0-5.0 scale). Extracts acoustic features using OpenAI's Whisper model and predicts quality scores using ensemble learning.

## **Features**
- **Audio Processing**: Uses Whisper-small for feature extraction
- **Multi-Model Approach**: XGBoost, CatBoost, Random Forest, KNN, MLP
- **Ensemble Learning**: Weighted blending of top-performing models
- **Imbalanced Data Handling**: Oversampling for minority classes
- **High Accuracy**: ~96% on validation data

## **Results**
- **Best Model**: XGBoost Regressor (96.0% accuracy)
- **Feature Dimension**: 768 (Whisper encoder output)
- **Output**: Scores in 0.5 increments (1.0, 1.5, ..., 5.0)

## **Tech Stack**
- Python, PyTorch, Librosa
- Scikit-learn, XGBoost, CatBoost
- Transformers (Hugging Face)
