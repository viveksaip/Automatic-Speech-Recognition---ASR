#Step 2: Install and Import Necessary Libraries
# Install necessary libraries
!pip install --upgrade transformers datasets torchaudio librosa evaluate

# Import libraries
import torch
import torchaudio
from datasets import load_dataset
import evaluate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
from google.colab import files


#Step 3: Load and Explore the Dataset
# Load a subset of the LibriSpeech dataset
dataset = load_dataset("librispeech_asr", "clean", split="validation[:1%]")  # Adjust the slice as needed

print(f"Number of samples: {len(dataset)}")
print(dataset[0])


#Step 4: Preprocess the Data
# Load the pre-trained processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Define the sampling rate expected by the model
sampling_rate = processor.feature_extractor.sampling_rate

def speech_file_to_array_fn(batch):
    # Load audio
    speech_array, _ = librosa.load(batch["file"], sr=sampling_rate)
    batch["speech"] = speech_array
    return batch

# Apply the function to load speech
dataset = dataset.map(speech_file_to_array_fn)

# Display an example
print(dataset[0]["speech"])



#Step 5: Tokenize the Transcriptions
def prepare_dataset(batch):
    # Tokenize the speech
    batch["input_values"] = processor(batch["speech"], sampling_rate=sampling_rate).input_values[0]
    # Tokenize the labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

# Apply the function to tokenize the dataset
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Display tokenized inputs
print(dataset[0]["input_values"])
print(dataset[0]["labels"])


#Step 6: Define the Data Collator
from transformers import DataCollatorCTCWithPadding

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


#Step 7: Load the Pre-trained Wav2Vec 2.0 Model
# Load the pre-trained Wav2Vec2 model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)


#Step 8: Define Metrics Using evaluate
# Load the WER metric using the evaluate library
wer = evaluate.load("wer")

def compute_metrics(pred):
    # Get the predicted IDs
    pred_ids = np.argmax(pred.predictions, axis=-1)
    # Decode the IDs to text
    pred_str = processor.batch_decode(pred_ids)
    
    # Replace -100 with the pad token for labels
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode the labels to text
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    
    # Compute WER
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}



#Step 9: Define Training Arguments and Initialize Trainer (Optional)
Note: Fine-tuning requires substantial computational resources. If you encounter resource limitations, consider skipping fine-tuning or using a smaller subset.

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-libri-clean",
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,  # Enable mixed precision
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=1e-4,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset,
    eval_dataset=dataset,  # Ideally, use separate train and eval sets
)


#Step 10: Start Training (Optional)

# Start training
trainer.train()

#If you choose not to fine-tune, you can proceed directly to evaluation and inference.

#Step 11: Evaluate the Model

# Evaluate the model
evaluation = trainer.evaluate()
print(f"WER: {evaluation['wer']}")

#Step 12: Make Predictions on New Audio Samples

def transcribe(file_path):
    # Load audio
    speech, _ = librosa.load(file_path, sr=sampling_rate)
    # Process the audio
    input_values = processor(speech, return_tensors="pt", padding="longest").input_values
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# Example using a file from the dataset
sample = dataset[0]
sample_file = sample["file"]

# Transcribe the sample audio
transcription = transcribe(sample_file)
print(f"Original Text: {sample['text'].lower()}")
print(f"Transcribed Text: {transcription}")

# Upload and transcribe your own audio files
uploaded = files.upload()

for fn in uploaded.keys():
    transcription = transcribe(fn)
    print(f"Transcribed Text for {fn}: {transcription}")


