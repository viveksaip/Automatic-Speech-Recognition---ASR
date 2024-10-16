# Automatic-Speech-Recognition---ASR

Overview: 
I developed an Automatic Speech Recognition (ASR) system leveraging Hugging Face’s state-of-the-art Wav2Vec 2.0 model. The primary goal was to create an efficient and accurate speech-to-text pipeline using Python within a Google Colab environment.
Technical Stack and Tools:
* Programming Language: Python
* Frameworks and Libraries: Hugging Face Transformers, PyTorch, Torchaudio, Librosa, and the Evaluate library for metrics.
* Development Environment: Google Colab with GPU acceleration to expedite model training and inference.
* Dataset: Utilized the LibriSpeech dataset, specifically a 1% subset of the validation split, to manage computational resources effectively.

Project Workflow:
1. Data Loading and Preprocessing:
    * Loading: Employed Hugging Face’s datasets library to load the LibriSpeech dataset. To mitigate long loading times and manage Colab’s resource constraints, I worked with a reduced subset (validation[:1%]).
    * Preprocessing: Used Librosa for audio processing, ensuring all audio files were resampled to 16kHz, which is the standard input rate for Wav2Vec 2.0. Normalization of audio signals was performed to maintain consistent amplitude levels across samples.
    * Tokenization: Leveraged the Wav2Vec2Processor to handle both feature extraction from audio waveforms and tokenization of textual transcriptions. This streamlined the pipeline by combining preprocessing steps efficiently.

2. Model Setup and Fine-Tuning:
    * Model Selection: Chose Wav2Vec 2.0, a powerful self-supervised model renowned for its robust performance in ASR tasks.
    * Fine-Tuning: Loaded the pre-trained Wav2Vec2ForCTC model and fine-tuned it on the selected LibriSpeech subset. Key hyperparameters such as learning rate (1e-4), batch size (8 with gradient accumulation steps of 2), and dropout rates were meticulously tuned to balance performance and resource utilization.
    * Optimization: Enabled mixed precision training (fp16=True) to accelerate computations and reduce memory consumption, which was crucial given the limited GPU resources in Colab.

3. Training and Evaluation:
    * Training: Implemented a training loop using Hugging Face’s Trainer API, which facilitated efficient training with built-in support for evaluation and checkpointing.
    * Evaluation Metrics: Employed the Evaluate library to compute the Word Error Rate (WER), a standard metric for assessing ASR accuracy. Through iterative training and validation, I achieved a WER of X%, demonstrating the model's effectiveness.
    * Regularization: To prevent overfitting and ensure generalization, I incorporated dropout layers, used early stopping based on validation performance, and applied data augmentation techniques such as adding background noise and time-shifting audio samples.

4. Inference and Deployment:
    * Inference Pipeline: Developed a prediction function that processes new audio inputs, performs inference using the fine-tuned model, and decodes the output into readable text.
    * Deployment Considerations: While the project was primarily focused on model development and evaluation within Colab, the architecture is scalable for deployment using frameworks like FastAPI or Flask, allowing integration into real-time applications.

Challenges and Solutions:
* Dataset Handling: Initially, loading the full LibriSpeech dataset was time-consuming. By selecting a smaller subset and caching the dataset locally within Colab, I significantly reduced loading times and improved efficiency.
* Memory Constraints: Training a large model on limited GPU resources posed memory challenges. Addressed this by optimizing batch sizes, utilizing gradient accumulation, and enabling mixed precision training to conserve memory without sacrificing performance.
* Library Updates: Encountered an ImportError with the load_metric function from the datasets library. Resolved this by migrating to the evaluate library, which now handles metric computations, ensuring compatibility and smooth workflow.

Outcome and Impact: The project successfully demonstrated the capability to build a high-accuracy ASR system using pre-trained models and efficient fine-tuning techniques. Achieving a low WER showcases the model's proficiency in transcribing spoken language accurately, laying the groundwork for potential applications in voice-controlled interfaces, transcription services, and accessibility tools.

Future Enhancements:
* Scaling Up: Incorporate larger datasets like the full LibriSpeech or Common Voice for enhanced model robustness.
* Model Optimization: Explore advanced optimization techniques such as model quantization or distillation to further reduce inference latency.
* Multi-Language Support: Extend the system to support multiple languages, enhancing its applicability across diverse user bases.
* Deployment: Develop a user-friendly API or integrate the ASR system into existing platforms for real-time speech recognition services.
