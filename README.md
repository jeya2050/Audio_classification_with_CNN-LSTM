# Music Genre Classification

## Project Description  
This is a deep learning project that utilizes **CNN** and **LSTM** models for music genre classification. The models are trained on the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which contains audio samples from 10 distinct music genres.  

## Dataset Details  
- The **GTZAN Dataset** consists of 10 genres: *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock*.  
- Each genre includes **100 samples**, making a total of 1,000 audio samples.  
- Each audio file is **30 seconds long**, stored in `.wav` format.  

## Model Training  
- The models were trained for **200 epochs**, ensuring robust performance.  
- The final weights are saved in the `saved_models` directory as `.keras` files.  
- The saved models can be directly loaded for predictions without retraining.  

## Features  
- **Convolutional Neural Network (CNN):** Extracts spatial features from audio spectrograms.  
- **Long Short-Term Memory (LSTM):** Captures temporal dependencies in the audio signals.  
- **Streamlit Web Interface:** An interactive platform to upload audio files and predict their genres in real time.  

## How to Use  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```  

2. Set up a Python virtual environment:  
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```  

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

4. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

## Demo  
Check out the demo video below showcasing the model's performance and the Streamlit interface:  
![Demo Video](./path_to_your_demo_video.mp4)  

## How to Predict  
1. Simply upload a 30-second `.wav` audio file via the Streamlit interface.  
2. The model will classify the audio into one of the 10 genres and display the result.  

## Pretrained Models  
- Pretrained models are stored in the `saved_models` directory.  
- To use the pretrained models:  
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('saved_models/cnn_lstm_model.keras')
   predictions = model.predict(audio_data)
   ```  

## References  
- Dataset: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- Frameworks Used: TensorFlow, Streamlit  

---  
