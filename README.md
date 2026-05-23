# SignSpeak — AI Sign Language Translator

A real-time sign language recognition app that runs entirely in your browser. No server required. No training data needed for basic signs. Train custom gestures locally with TensorFlow.js.

![SignSpeak](https://img.shields.io/badge/TensorFlow.js-LSTM-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![React](https://img.shields.io/badge/React-19-blue)

## What it does

- **Pre-trained sign recognition** — Instantly recognizes 12 common signs (hello, yes, no, thank you, i love you, peace, ok, stop, point, call me, rock, fist) using hand geometry rules on MediaPipe landmarks.
- **Custom gesture training** — Record your own gesture samples and train an LSTM neural network locally in the browser with TensorFlow.js.
- **Sentence builder** — Chain recognized signs into full sentences.
- **Text-to-Speech** — Speak recognized sentences aloud using the Web Speech API.
- **No backend needed** — Everything runs client-side. Your camera never leaves your device.

## Try it live

[Open the app](https://id-preview--a6299034-dec3-47e4-a509-c6d007e7ab87.lovable.app)

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | TanStack Start (React 19 + Vite + file-based routing) |
| ML | TensorFlow.js (client-side LSTM training + inference) |
| Hand tracking | MediaPipe Hands (21 landmarks per hand) |
| Styling | Tailwind CSS v4 + shadcn/ui |
| Speech | Web Speech API (browser-native TTS) |

## Quick start

```bash
# Install dependencies
bun install

# Run dev server
bun dev
```

Open `http://localhost:3000` and allow camera access.

## How to use

### Pre-trained mode (no training)
1. Open the **Live** page.
2. Click **Start camera**.
3. Select **Pre-trained signs** mode.
4. Show your hand — recognized signs appear instantly.

### Custom training mode
1. Go to **Train** page.
2. Add gesture labels (e.g., `hello`, `goodbye`).
3. Click **Start camera**, then **Record sample** for each gesture.
4. Record at least **5 samples per gesture** (8–15 recommended).
5. Click **Train model** — training runs locally in your browser.
6. Go back to **Live** and switch to **Trained model** mode.

Your trained model and samples are saved in the browser (IndexedDB + localStorage) and persist across reloads.

## Project structure

```
src/
├── routes/
│   ├── index.tsx          # Live recognition page
│   ├── train.tsx           # In-browser gesture trainer
│   └── about.tsx           # About / project info
├── hooks/
│   ├── useHandTracking.ts  # MediaPipe + TF.js inference hook
│   └── useSentenceBuilder.ts # Sentence + TTS logic
├── lib/
│   ├── heuristicRecognizer.ts  # Pre-trained rule-based recognizer
│   └── gestureStore.ts     # IndexedDB model storage + training
├── components/ui/          # shadcn/ui components
├── styles.css             # Tailwind v4 + custom tokens
```

## Python pipeline (optional)

For users who prefer Python training or want to export to TF.js:

```bash
cd python
pip install -r requirements.txt

# 1. Record data with webcam
python collect_data.py

# 2. Train in Google Colab (see python/train_lstm.py)

# 3. Export to TF.js
python export_tfjs.py
```

Drop exported files into `public/model/` to load them in the app.

## Supported pre-trained signs

| Sign | Hand shape |
|------|-----------|
| hello | Open hand, all 5 fingers extended |
| yes | Thumb up, other fingers folded |
| no | Thumb down, other fingers folded |
| thank you | (heuristic) |
| i love you | Thumb + index + pinky extended |
| peace | Index + middle extended (V sign) |
| ok | Thumb + index touching, others extended |
| stop | All 4 fingers extended (no thumb) |
| point | Index finger only |
| call me | Thumb + pinky extended |
| rock | Index + pinky extended |
| fist | All fingers folded |

## Browser requirements

- Webcam access (HTTPS or localhost)
- WebGL for TensorFlow.js acceleration
- Web Speech API for TTS (Chrome/Edge/Safari recommended)

## License

MIT
