import * as tf from "@tensorflow/tfjs";

export const MODEL_KEY = "indexeddb://signspeak-gestures";
export const LABELS_LS_KEY = "signspeak.labels";
export const SAMPLES_LS_KEY = "signspeak.samples"; // optional cache

export const SEQ_LENGTH = 30;
export const FEATURE_LEN = 126;

export type Sample = { label: string; sequence: number[][] };

export function loadLabels(): string[] {
  try {
    const raw = localStorage.getItem(LABELS_LS_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

export function saveLabels(labels: string[]) {
  localStorage.setItem(LABELS_LS_KEY, JSON.stringify(labels));
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
      units: 64,
      returnSequences: true,
      inputShape: [SEQ_LENGTH, FEATURE_LEN],
    }),
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.lstm({ units: 64 }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(
    tf.layers.dense({ units: numClasses, activation: "softmax" }),
  );
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export async function trainModel(
  samples: Sample[],
  labels: string[],
  onEpoch: (epoch: number, logs: { loss: number; acc: number }) => void,
  epochs = 40,
): Promise<tf.LayersModel> {
  const labelIdx = new Map(labels.map((l, i) => [l, i]));
  const xs = tf.tensor3d(samples.map((s) => s.sequence));
  const ysFlat = samples.map((s) => labelIdx.get(s.label) ?? 0);
  const ys = tf.oneHot(tf.tensor1d(ysFlat, "int32"), labels.length);

  const model = buildModel(labels.length);
  await model.fit(xs, ys, {
    epochs,
    batchSize: Math.min(16, Math.max(2, Math.floor(samples.length / 4))),
    shuffle: true,
    validationSplit: samples.length >= 20 ? 0.15 : 0,
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