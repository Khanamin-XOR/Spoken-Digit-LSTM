# Spoken Digit Recognition with LSTMs — Raw Audio vs Spectrograms

A controlled 2 × 2 experiment in audio classification: LSTMs trained on the **[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)** to predict spoken digits 0 – 9, comparing **raw waveforms vs spectrograms** as input and the effect of **audio data augmentation**.

## Results

| Model | Input | Training data | Val F1 (Micro) |
|---|---|---|---:|
| Model 1 | Raw waveform | 1,600 samples | 10.00 % *(random)* |
| **Model 2** | **Spectrogram** | **1,600 samples** | **97.50 %** |
| Model 3 | Raw waveform | 14,400 (9× augmented) | 10.00 % *(random)* |
| **Model 4** | **Spectrogram** | **14,400 (9× augmented)** | **98.83 %** |

### What the results say

- **Input representation dominates data volume.** Raw waveforms refused to learn — LSTM validation accuracy stuck at chance (10 % across 10 classes), even with 9× augmented data (Models 1 and 3). Switching to spectrograms produced a 97.5 % F1 on the *smaller* dataset (Model 2). The lesson: for audio, *what you feed the model matters more than how much you feed it.*
- **Augmentation compounds on a good representation.** On top of spectrograms, time stretching + pitch shifting (Model 4) pushed validation F1 from 97.5 % → 98.83 % — a relative reduction in error of more than 50 %.

## Method

### Data
- **Free Spoken Digit Dataset** — 2,000 `.wav` recordings of digits 0 – 9
- 80 / 20 train / validation split (1,600 / 400)

### Preprocessing pipeline
1. Audio loading via **librosa** at 22,050 Hz
2. Duration analysis: 99 th percentile of clips is ≤ 0.8 sec
3. Padding / truncation to 17,640 samples (= 0.8 × 22050) with explicit **masking vectors** so the LSTM doesn't backpropagate through pad tokens
4. (For spectrogram models) librosa STFT to convert each clip into a time-frequency matrix

### Models
- **Models 1 & 3 (raw waveform):** masked LSTM over the (17640,) sequence, dense head, softmax over 10 classes
- **Models 2 & 4 (spectrogram):** LSTM returning output at every time step → average pooling across time steps → dense head → softmax

All models trained with sparse-categorical-crossentropy, Adam, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint on best val_loss.

### Audio augmentation (Models 3 & 4)
Two transforms applied stochastically to expand the training set 9×:

- **Time stretching:** scaling the audio playback speed by ±30 % without altering pitch
- **Pitch shifting:** shifting the entire signal up or down by half-steps

## Tech stack

Python 3 · TensorFlow / Keras · librosa · NumPy · pandas · scikit-learn · matplotlib · Jupyter

## Run

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib jupyter
jupyter notebook Speech_detection_Solution.ipynb
```

**Note on data:** the `recordings/` directory of the Free Spoken Digit Dataset is not committed here. Original dataset: https://github.com/Jakobovski/free-spoken-digit-dataset. The notebook contains preserved training outputs across all 4 models, so the validation F1 values and per-epoch loss curves are visible end-to-end without re-execution.

## Takeaway

The strongest signal in this project isn't either of the 98 % models — it's the **negative result** on the raw-waveform models. They sit in the table on purpose: they're the controlled comparison that turns a result ("spectrograms work") into evidence ("spectrograms are the reason this works"). The augmentation lift comes second, on top of the right representation.

## Related projects

This repo is part of an ML series exploring different problem domains from scratch:
- [PCA-From-Scratch](https://github.com/Khanamin-XOR/PCA-From-Scratch) · [ML-Metrics-From-Scratch](https://github.com/Khanamin-XOR/ML-Metrics-From-Scratch) · [Microsoft-Malware-Detection](https://github.com/Khanamin-XOR/Microsoft-Malware-Detection) · [Movie-Recommender-Matrix-Factorization](https://github.com/Khanamin-XOR/Movie-Recommender-Matrix-Factorization)

## License

MIT
