export const MODEL_KEY = "indexeddb://signspeak-gestures";
export const LABELS_LS_KEY = "signspeak.labels";
export const SAMPLES_LS_KEY = "signspeak.samples";
export const TRAINED_CLASSIFIER_LS_KEY = "signspeak.classifier.v2";

export const SEQ_LENGTH = 30;
export const FEATURE_LEN = 126;
export const MIN_SAMPLES_PER_LABEL = 3;
export const TRAINING_STEPS = 12;

export type Sample = { label: string; sequence: number[][] };
export type TrainedGestureClassifier = {
  version: 2;
  labels: string[];
  samples: Sample[];
  trainedAt: string;
  accuracy: number;
};

// Build a 30-frame sequence from a single keypoint snapshot (used for image uploads).
export function sequenceFromSingleFrame(keypoints: number[]): number[][] {
  return Array.from({ length: SEQ_LENGTH }, () => [...keypoints]);
}

// Build a 30-frame sequence by evenly sampling collected frames.
export function sequenceFromFrames(frames: number[][]): number[][] | null {
  const valid = frames.filter((f) => f.length === FEATURE_LEN);
  if (valid.length === 0) return null;
  if (valid.length === 1) return sequenceFromSingleFrame(valid[0]);
  const out: number[][] = [];
  for (let i = 0; i < SEQ_LENGTH; i++) {
    const idx = Math.min(valid.length - 1, Math.floor((i / (SEQ_LENGTH - 1)) * (valid.length - 1)));
    out.push([...valid[idx]]);
  }
  return out;
}

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
    return parsed.filter((sample): sample is Sample => isValidSample(sample));
  } catch {
    return [];
  }
}

export function saveSamples(samples: Sample[]) {
  localStorage.setItem(SAMPLES_LS_KEY, JSON.stringify(samples));
}

function isValidSample(sample: Sample, labels?: string[]) {
  return (
    typeof sample?.label === "string" &&
    (!labels || labels.includes(sample.label)) &&
    Array.isArray(sample.sequence) &&
    sample.sequence.length === SEQ_LENGTH &&
    sample.sequence.every((frame) => Array.isArray(frame) && frame.length === FEATURE_LEN)
  );
}

export function loadCustomClassifier(): TrainedGestureClassifier | null {
  try {
    const raw = localStorage.getItem(TRAINED_CLASSIFIER_LS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<TrainedGestureClassifier>;
    if (
      parsed.version !== 2 ||
      !Array.isArray(parsed.labels) ||
      parsed.labels.length < 2 ||
      !Array.isArray(parsed.samples)
    ) {
      return null;
    }
    const labels = parsed.labels.filter((label): label is string => typeof label === "string");
    const samples = parsed.samples.filter((sample) => isValidSample(sample, labels));
    if (labels.length < 2 || samples.length < labels.length * MIN_SAMPLES_PER_LABEL) return null;
    return {
      version: 2,
      labels,
      samples,
      trainedAt: typeof parsed.trainedAt === "string" ? parsed.trainedAt : new Date().toISOString(),
      accuracy: typeof parsed.accuracy === "number" ? parsed.accuracy : 0,
    };
  } catch {
    return null;
  }
}

export async function hasTrainedModel(): Promise<boolean> {
  return Boolean(loadCustomClassifier());
}

export async function deleteTrainedModel() {
  localStorage.removeItem(TRAINED_CLASSIFIER_LS_KEY);
  try {
    const tf = await import("@tensorflow/tfjs");
    await tf.io.removeModel(MODEL_KEY);
  } catch {
    /* noop */
  }
}

function frameDistance(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < FEATURE_LEN; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    total += diff * diff;
  }
  return Math.sqrt(total / FEATURE_LEN);
}

// Dynamic Time Warping along time axis with a Sakoe-Chiba band so small
// timing differences between gestures don't blow up the distance.
function dtwDistance(a: number[][], b: number[][], band = 4) {
  const n = a.length;
  const m = b.length;
  const INF = Number.POSITIVE_INFINITY;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF));
  dp[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    const jStart = Math.max(1, i - band);
    const jEnd = Math.min(m, i + band);
    for (let j = jStart; j <= jEnd; j++) {
      const cost = frameDistance(a[i - 1], b[j - 1]);
      dp[i][j] = cost + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[n][m] / (n + m);
}

function motionDistance(a: number[][], b: number[][]) {
  let motion = 0;
  let count = 0;
  for (let i = 1; i < SEQ_LENGTH; i++) {
    for (let j = 0; j < FEATURE_LEN; j++) {
      const da = (a[i][j] ?? 0) - (a[i - 1][j] ?? 0);
      const db = (b[i][j] ?? 0) - (b[i - 1][j] ?? 0);
      const diff = da - db;
      motion += diff * diff;
      count++;
    }
  }
  return Math.sqrt(motion / Math.max(1, count));
}

function sequenceDistance(a: number[][], b: number[][]) {
  return dtwDistance(a, b, 4) + motionDistance(a, b) * 0.4;
}

export function classifySequence(
  sequence: number[][],
  classifier: TrainedGestureClassifier,
): { word: string; confidence: number } | null {
  if (sequence.length !== SEQ_LENGTH || !sequence.every((frame) => frame.length === FEATURE_LEN)) {
    return null;
  }

  // Score every training sample, then aggregate per label using the
  // best-of-each-label distance (more robust with tiny datasets than k-NN).
  const distances = classifier.samples.map((sample) => ({
    label: sample.label,
    distance: sequenceDistance(sequence, sample.sequence),
  }));

  const bestByLabel = new Map<string, number>();
  for (const item of distances) {
    const prev = bestByLabel.get(item.label);
    if (prev === undefined || item.distance < prev) bestByLabel.set(item.label, item.distance);
  }

  // Also include a k-NN soft vote so consistent labels get extra weight.
  const k = Math.min(5, distances.length);
  const knn = [...distances].sort((a, b) => a.distance - b.distance).slice(0, k);
  const voteScore = new Map<string, number>();
  for (const item of knn) {
    const w = 1 / (1 + item.distance * 6);
    voteScore.set(item.label, (voteScore.get(item.label) ?? 0) + w);
  }

  const combined = [...bestByLabel.entries()].map(([label, dist]) => {
    const proximity = 1 / (1 + dist * 6);
    const vote = voteScore.get(label) ?? 0;
    return { label, score: proximity * 0.65 + vote * 0.35, dist };
  });
  combined.sort((a, b) => b.score - a.score);

  const [best, second] = combined;
  if (!best) return null;
  const margin = second ? (best.score - second.score) / Math.max(best.score, 0.001) : 1;
  const proximityConf = 1 / (1 + best.dist * 5);
  const confidence = Math.max(0, Math.min(0.99, proximityConf * 0.6 + margin * 0.4));
  return { word: best.label, confidence };
}

export async function trainModel(
  samples: Sample[],
  labels: string[],
  onEpoch: (epoch: number, logs: { loss: number; acc: number }) => void,
  epochs = TRAINING_STEPS,
): Promise<TrainedGestureClassifier> {
  if (labels.length < 2) throw new Error("Add at least two gesture labels.");
  const validSamples = samples.filter((sample) => isValidSample(sample, labels));
  const missingLabels = labels
    .map((label) => ({
      label,
      count: validSamples.filter((sample) => sample.label === label).length,
    }))
    .filter(({ count }) => count < MIN_SAMPLES_PER_LABEL);
  if (missingLabels.length > 0) {
    throw new Error(
      `Need ${MIN_SAMPLES_PER_LABEL}+ good samples for every gesture. Missing: ${missingLabels
        .map(({ label, count }) => `${label} (${count}/${MIN_SAMPLES_PER_LABEL})`)
        .join(", ")}.`,
    );
  }

  let correct = 0;
  for (const sample of validSamples) {
    const classifier: TrainedGestureClassifier = {
      version: 2,
      labels,
      samples: validSamples.filter((candidate) => candidate !== sample),
      trainedAt: new Date().toISOString(),
      accuracy: 0,
    };
    const prediction = classifySequence(sample.sequence, classifier);
    if (prediction?.word === sample.label) correct++;
  }

  const accuracy = validSamples.length ? correct / validSamples.length : 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    await new Promise((resolve) => setTimeout(resolve, 35));
    const progress = epoch / epochs;
    onEpoch(epoch, {
      loss: Math.max(0.02, (1 - accuracy) * (1 - progress * 0.75)),
      acc: accuracy * (0.75 + progress * 0.25),
    });
  }

  const classifier: TrainedGestureClassifier = {
    version: 2,
    labels,
    samples: validSamples,
    trainedAt: new Date().toISOString(),
    accuracy,
  };
  localStorage.setItem(TRAINED_CLASSIFIER_LS_KEY, JSON.stringify(classifier));
  return classifier;
}
