import * as tf from "@tensorflow/tfjs";

export const MODEL_KEY = "indexeddb://signspeak-gestures";
export const LABELS_LS_KEY = "signspeak.labels";
export const SAMPLES_LS_KEY = "signspeak.samples";

export const SEQ_LENGTH = 30;
export const FEATURE_LEN = 126;

export type Sample = { label: string; sequence: number[][] };

export function loadLabels(): string[] {
  try {
    const raw = localStorage.getItem(LABELS_LS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((v) => typeof v === "string") : [];
  } catch {
    return [];
  }
}

export function saveLabels(labels: string[]) {
  localStorage.setItem(LABELS_LS_KEY, JSON.stringify(labels));
}

export function loadSamples(): Sample[] {
  try {
    const raw = localStorage.getItem(SAMPLES_LS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (sample): sample is Sample =>
        typeof sample?.label === "string" &&
        Array.isArray(sample?.sequence) &&
        sample.sequence.length === SEQ_LENGTH,
    );
  } catch {
    return [];
  }
}

export function saveSamples(samples: Sample[]) {
  localStorage.setItem(SAMPLES_LS_KEY, JSON.stringify(samples));
}

export async function hasTrainedModel(): Promise<boolean> {
  try {
    const list = await tf.io.listModels();
    return Object.prototype.hasOwnProperty.call(list, MODEL_KEY);
  } catch {
    return false;
  }
}

export async function deleteTrainedModel() {
  try {
    await tf.io.removeModel(MODEL_KEY);
  } catch {
    /* noop */
  }
}

export function buildModel(numClasses: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: 96,
      returnSequences: true,
      inputShape: [SEQ_LENGTH, FEATURE_LEN],
    }),
  );
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.lstm({ units: 64 }));
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(8e-4),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

function jitterSequence(sequence: number[][], amount = 0.012) {
  return sequence.map((frame) => frame.map((value) => value + (Math.random() - 0.5) * amount));
}

function augmentSamples(samples: Sample[]) {
  const augmented: Sample[] = [...samples];
  for (const sample of samples) {
    augmented.push({ label: sample.label, sequence: jitterSequence(sample.sequence) });
  }
  return augmented;
}

export async function trainModel(
  samples: Sample[],
  labels: string[],
  onEpoch: (epoch: number, logs: { loss: number; acc: number }) => void,
  epochs = 50,
): Promise<tf.LayersModel> {
  if (labels.length < 2) throw new Error("Add at least two gesture labels.");
  const validSamples = samples.filter(
    (sample) => labels.includes(sample.label) && sample.sequence.length === SEQ_LENGTH,
  );
  if (validSamples.length < labels.length * 5) {
    throw new Error("Record at least 5 good samples for every gesture.");
  }

  const trainingSamples = augmentSamples(validSamples);
  const labelIdx = new Map(labels.map((l, i) => [l, i]));
  const xs = tf.tensor3d(trainingSamples.map((s) => s.sequence));
  const ysFlat = trainingSamples.map((s) => labelIdx.get(s.label) ?? 0);
  const ys = tf.oneHot(tf.tensor1d(ysFlat, "int32"), labels.length);

  const model = buildModel(labels.length);
  await model.fit(xs, ys, {
    epochs,
    batchSize: Math.min(16, Math.max(2, Math.floor(trainingSamples.length / 4))),
    shuffle: true,
    validationSplit: trainingSamples.length >= 40 ? 0.15 : 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        onEpoch(epoch + 1, {
          loss: Number(logs?.loss ?? 0),
          acc: Number(logs?.acc ?? logs?.accuracy ?? 0),
        });
      },
    },
  });
  xs.dispose();
  ys.dispose();
  await model.save(MODEL_KEY);
  return model;
}
