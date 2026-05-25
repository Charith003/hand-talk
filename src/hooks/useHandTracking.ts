import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  classifySequence,
  loadCustomClassifier,
  loadLabels,
  SEQ_LENGTH as S_LEN,
  FEATURE_LEN as F_LEN,
  type TrainedGestureClassifier,
} from "@/lib/gestureStore";
import { recognizeHeuristic, HEURISTIC_VOCAB } from "@/lib/heuristicRecognizer";

const SEQ_LENGTH = S_LEN;
const FEATURE_LEN = F_LEN;
const DEFAULT_CONF = 0.8;
const STABLE_WINDOW = 6;
const STABLE_MIN_VOTES = 4;

type Prediction = { word: string; confidence: number };
type ModelSource = "indexeddb" | "public" | "heuristic" | "demo";
type Landmark = { x?: number; y?: number; z?: number };
type HandResults = { multiHandLandmarks?: Landmark[][] };
type CameraHandle = { stop?: () => void };
type HandsHandle = { close?: () => void };

export interface HandTrackingOptions {
  confidenceThreshold?: number;
  onPrediction?: (p: Prediction) => void;
  onFrame?: (keypoints: number[], handCount: number) => void;
  enableInference?: boolean;
  modelVersion?: number;
  autoStart?: boolean;
  mode?: "auto" | "model" | "heuristic";
}

function emptyFeatures() {
  return new Array(FEATURE_LEN).fill(0);
}

function normalizeHand(landmarks: Landmark[] | undefined) {
  if (!landmarks?.length) return new Array(63).fill(0);

  const wrist = landmarks[0];
  const middleBase = landmarks[9] ?? landmarks[0];
  const palmScale = Math.max(
    0.08,
    Math.hypot(
      (middleBase.x ?? 0) - (wrist.x ?? 0),
      (middleBase.y ?? 0) - (wrist.y ?? 0),
      (middleBase.z ?? 0) - (wrist.z ?? 0),
    ),
  );

  return landmarks.flatMap((lm) => [
    ((lm.x ?? 0) - (wrist.x ?? 0)) / palmScale,
    ((lm.y ?? 0) - (wrist.y ?? 0)) / palmScale,
    ((lm.z ?? 0) - (wrist.z ?? 0)) / palmScale,
  ]);
}

function extractKeypoints(handsLandmarks: Landmark[][]) {
  if (!handsLandmarks.length) return emptyFeatures();
  const normalized = handsLandmarks.slice(0, 2).flatMap((hand) => normalizeHand(hand));
  while (normalized.length < FEATURE_LEN) normalized.push(0);
  return normalized.slice(0, FEATURE_LEN);
}

function getClassCount(model: tf.LayersModel) {
  const output = model.outputs[0]?.shape;
  return output?.[output.length - 1] ?? null;
}

export function useHandTracking(options: HandTrackingOptions = {}) {
  const {
    confidenceThreshold = DEFAULT_CONF,
    onPrediction,
    onFrame,
    enableInference = true,
    modelVersion = 0,
    autoStart = false,
    mode = "auto",
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const classifierRef = useRef<TrainedGestureClassifier | null>(null);
  const labelsRef = useRef<string[]>([]);
  const sequenceRef = useRef<number[][]>([]);
  const cameraRef = useRef<CameraHandle | null>(null);
  const handsRef = useRef<HandsHandle | null>(null);
  const lastInferRef = useRef(0);
  const predictionWindowRef = useRef<Prediction[]>([]);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [prediction, setPrediction] = useState<Prediction>({ word: "", confidence: 0 });
  const [isReady, setIsReady] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [status, setStatus] = useState("Loading recognizer...");
  const [handVisible, setHandVisible] = useState(false);
  const [modelSource, setModelSource] = useState<ModelSource>("demo");
  const [cameraRequested, setCameraRequested] = useState(autoStart);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    if (autoStart) setCameraRequested(true);
  }, [autoStart]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setIsReady(false);
      setPrediction({ word: "", confidence: 0 });
      predictionWindowRef.current = [];
      sequenceRef.current = [];
      try {
        modelRef.current?.dispose?.();
      } catch {
        /* noop */
      }
      modelRef.current = null;
      classifierRef.current = null;

      if (mode === "heuristic") {
        labelsRef.current = [...HEURISTIC_VOCAB];
        setDemoMode(false);
        setModelSource("heuristic");
        setStatus("Pre-trained signs ready");
        setIsReady(true);
        return;
      }

      const classifier = loadCustomClassifier();
      if (classifier) {
        classifierRef.current = classifier;
        labelsRef.current = classifier.labels;
        setDemoMode(false);
        setModelSource("indexeddb");
        setStatus("Custom trained model ready");
        setIsReady(true);
        return;
      }

      try {
        setStatus("Looking for project model...");
        const lres = await fetch("/labels.json");
        if (!lres.ok) throw new Error("No labels file");
        const labels = await lres.json();
        const model = await tf.loadLayersModel("/model/model.json");
        if (getClassCount(model) !== labels.length) {
          model.dispose();
          throw new Error("Labels do not match model outputs");
        }
        const dummy = tf.zeros([1, SEQ_LENGTH, FEATURE_LEN]);
        const warm = model.predict(dummy) as tf.Tensor;
        await warm.data();
        dummy.dispose();
        warm.dispose();
        if (cancelled) {
          model.dispose();
          return;
        }
        modelRef.current = model;
        labelsRef.current = labels;
        setDemoMode(false);
        setModelSource("public");
        setStatus("Project model ready");
      } catch {
        if (cancelled) return;
        labelsRef.current = [];
        if (mode === "auto") {
          labelsRef.current = [...HEURISTIC_VOCAB];
          setDemoMode(false);
          setModelSource("heuristic");
          setStatus("Pre-trained signs ready");
        } else {
          setDemoMode(true);
          setModelSource("demo");
          setStatus("Train a model to start recognition");
        }
      } finally {
        if (!cancelled) setIsReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [modelVersion, mode]);

  const startCamera = useCallback(() => {
    setCameraError("");
    setCameraRequested(true);
  }, []);

  const drawSkeleton = useCallback(
    (landmarksList: Landmark[][], ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
      ];
      for (const landmarks of landmarksList) {
        ctx.strokeStyle = "rgba(72, 190, 190, 0.95)";
        ctx.lineWidth = 3;
        for (const [a, b] of connections) {
          const pa = landmarks[a],
            pb = landmarks[b];
          ctx.beginPath();
          ctx.moveTo((1 - (pa.x ?? 0)) * w, (pa.y ?? 0) * h);
          ctx.lineTo((1 - (pb.x ?? 0)) * w, (pb.y ?? 0) * h);
          ctx.stroke();
        }
        for (const lm of landmarks) {
          ctx.fillStyle = "rgb(255, 196, 87)";
          ctx.beginPath();
          ctx.arc((1 - (lm.x ?? 0)) * w, (lm.y ?? 0) * h, 4.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    },
    [],
  );

  const onResults = useCallback(
    async (results: HandResults) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const handsLandmarks = results.multiHandLandmarks ?? [];
      const handCount = handsLandmarks.length;
      setHandVisible(handCount > 0);

      if (handCount > 0) drawSkeleton(handsLandmarks, ctx, width, height);
      const keypoints = extractKeypoints(handsLandmarks);

      sequenceRef.current.push(keypoints);
      if (sequenceRef.current.length > SEQ_LENGTH) sequenceRef.current.shift();
      onFrameRef.current?.(keypoints, handCount);

      if (!enableInference || demoMode) return;

      // Heuristic recognizer path (no trained model required).
      if (modelSource === "heuristic") {
        const now = performance.now();
        if (now - lastInferRef.current < 110) return;
        lastInferRef.current = now;
        const guess = recognizeHeuristic(handsLandmarks);
        const item: Prediction = guess ?? { word: "", confidence: 0 };
        predictionWindowRef.current.push(item);
        if (predictionWindowRef.current.length > STABLE_WINDOW)
          predictionWindowRef.current.shift();
        const votes = new Map<string, { count: number; total: number }>();
        for (const it of predictionWindowRef.current) {
          if (!it.word) continue;
          const e = votes.get(it.word) ?? { count: 0, total: 0 };
          votes.set(it.word, { count: e.count + 1, total: e.total + it.confidence });
        }
        let stable: Prediction = { word: "", confidence: item.confidence };
        votes.forEach((value, key) => {
          if (
            value.count >= STABLE_MIN_VOTES &&
            value.count > (votes.get(stable.word)?.count ?? 0)
          ) {
            stable = { word: key, confidence: value.total / value.count };
          }
        });
        setPrediction(stable);
        if (stable.word) onPrediction?.(stable);
        return;
      }

      if (sequenceRef.current.length < SEQ_LENGTH) return;

      const now = performance.now();
      if (now - lastInferRef.current < 110) return;
      lastInferRef.current = now;

      if (classifierRef.current) {
        const guess = classifySequence(sequenceRef.current, classifierRef.current);
        const item: Prediction = guess ?? { word: "", confidence: 0 };
        const word = item.confidence >= Math.max(0.55, confidenceThreshold - 0.2) ? item.word : "";
        predictionWindowRef.current.push({ word, confidence: item.confidence });
        if (predictionWindowRef.current.length > STABLE_WINDOW) predictionWindowRef.current.shift();

        const votes = new Map<string, { count: number; total: number }>();
        for (const sample of predictionWindowRef.current) {
          if (!sample.word) continue;
          const existing = votes.get(sample.word) ?? { count: 0, total: 0 };
          votes.set(sample.word, {
            count: existing.count + 1,
            total: existing.total + sample.confidence,
          });
        }

        let stable: Prediction = { word: "", confidence: item.confidence };
        votes.forEach((value, key) => {
          if (value.count >= 3 && value.count > (votes.get(stable.word)?.count ?? 0)) {
            stable = { word: key, confidence: value.total / value.count };
          }
        });

        setPrediction(stable);
        if (stable.word) onPrediction?.(stable);
        return;
      }

      if (!modelRef.current) return;

      const input = tf.tensor3d([sequenceRef.current]);
      const pred = modelRef.current.predict(input) as tf.Tensor;
      const probs = (await pred.data()) as Float32Array;
      input.dispose();
      pred.dispose();

      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
      const conf = probs[maxIdx] ?? 0;
      const word = conf >= confidenceThreshold ? (labelsRef.current[maxIdx] ?? "") : "";

      predictionWindowRef.current.push({ word, confidence: conf });
      if (predictionWindowRef.current.length > STABLE_WINDOW) predictionWindowRef.current.shift();

      const votes = new Map<string, { count: number; total: number }>();
      for (const item of predictionWindowRef.current) {
        if (!item.word) continue;
        const existing = votes.get(item.word) ?? { count: 0, total: 0 };
        votes.set(item.word, {
          count: existing.count + 1,
          total: existing.total + item.confidence,
        });
      }

      let stable: Prediction = { word: "", confidence: conf };
      votes.forEach((value, key) => {
        if (value.count >= STABLE_MIN_VOTES && value.count > (votes.get(stable.word)?.count ?? 0)) {
          stable = { word: key, confidence: value.total / value.count };
        }
      });

      setPrediction(stable);
      if (stable.word) onPrediction?.(stable);
    },
    [drawSkeleton, demoMode, confidenceThreshold, onPrediction, enableInference, modelSource],
  );

  useEffect(() => {
    if (!isReady || !cameraRequested) return;
    let stopped = false;
    (async () => {
      try {
        setStatus("Starting camera...");
        setCameraError("");
        const { Hands } = await import("@mediapipe/hands");
        const { Camera } = await import("@mediapipe/camera_utils");
        if (stopped) return;
        const hands = new Hands({
          locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
        });
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.65,
          minTrackingConfidence: 0.55,
        });
        hands.onResults(onResults);
        handsRef.current = hands;

        if (videoRef.current) {
          const camera = new Camera(videoRef.current, {
            onFrame: async () => {
              if (videoRef.current) await hands.send({ image: videoRef.current });
            },
            width: 640,
            height: 480,
          });
          cameraRef.current = camera;
          await camera.start();
          if (stopped) return;
          setCameraStarted(true);
          setStatus(demoMode ? "Camera ready — train a model" : "Camera ready");
        }
      } catch {
        if (stopped) return;
        setCameraStarted(false);
        setCameraRequested(false);
        setCameraError("Camera access was blocked. Allow camera permission, then try again.");
        setStatus("Camera blocked");
      }
    })();

    return () => {
      stopped = true;
      setCameraStarted(false);
      try {
        cameraRef.current?.stop?.();
      } catch {
        /* noop */
      }
      try {
        handsRef.current?.close?.();
      } catch {
        /* noop */
      }
    };
  }, [isReady, cameraRequested, onResults, demoMode]);

  return {
    videoRef,
    canvasRef,
    prediction,
    isReady,
    demoMode,
    status,
    handVisible,
    vocabulary: labelsRef.current,
    modelSource,
    cameraStarted,
    cameraError,
    startCamera,
  };
}
