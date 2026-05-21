# SignSpeak — Python training pipeline

## Install
```bash
pip install -r requirements.txt
```

## 1. Collect dataset
```bash
python collect_data.py
```
Records 30 samples × 30 frames per class from your webcam. Press Enter to
start each take. Output: `dataset/<class>/<idx>/sequence.npy` shaped
`(30, 126)` — 21 landmarks × 3 coords × 2 hands.

## 2. Train on Google Colab (recommended)
Upload `dataset/` and `train_lstm.py` to a Colab notebook with a T4 GPU runtime:
```bash
!pip install -q tensorflow scikit-learn
!python train_lstm.py
```
~15–30 min for 100 classes. Produces `best_model.keras` and `labels.npy`.

## 3. Export for the browser
```bash
pip install tensorflowjs
python export_tfjs.py
```

## 4. Deploy into the React app
```
tfjs_model_quantized/model.json          → public/model/model.json
tfjs_model_quantized/group1-shard*.bin   → public/model/
labels.json                              → public/labels.json
```
Reload the page — the demo banner disappears and live inference takes over.