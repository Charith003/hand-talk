import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  MODEL_KEY,
  loadLabels,
  SEQ_LENGTH as S_LEN,
  FEATURE_LEN as F_LEN,
} from "@/lib/gestureStore";

const SEQ_LENGTH = S_LEN;
const FEATURE_LEN = F_LEN;
const DEFAULT_CONF = 0.85;

type Prediction = { word: string; confidence: number };

export interface HandTrackingOptions {
  confidenceThreshold?: number;
  onPrediction?: (p: Prediction) => void;
  onFrame?: (keypoints: number[], handCount: number) => void;
  enableInference?: boolean;
  modelVersion?: number;
}

export function useHandTracking(options: HandTrackingOptions = {}) {
  const {
    confidenceThreshold = DEFAULT_CONF,
    onPrediction,
    onFrame,
    enableInference = true,
    modelVersion = 0,
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const labelsRef = useRef<string[]>([]);
  const sequenceRef = useRef<number[][]>([]);
  const cameraRef = useRef<any>(null);
  const handsRef = useRef<any>(null);
  const lastInferRef = useRef(0);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [prediction, setPrediction] = useState<Prediction>({ word: "", confidence: 0 });
  const [isReady, setIsReady] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [status, setStatus] = useState("Initializing...");
  const [handVisible, setHandVisible] = useState(false);
  const [modelSource, setModelSource] = useState<"indexeddb" | "public" | "demo">("demo");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setIsReady(false);
      try { modelRef.current?.dispose?.(); } catch { /* noop */ }
      modelRef.current = null;

      // 1) Prefer in-browser trained model in IndexedDB
      try {
        const list = await tf.io.listModels();
        if (list[MODEL_KEY]) {
          const stored = loadLabels();
          if (stored.length > 0) {
            setStatus("Loading your trained model...");
            const model = await tf.loadLayersModel(MODEL_KEY);
            const dummy = tf.zeros([1, SEQ_LENGTH, FEATURE_LEN]);
            const warm = model.predict(dummy) as tf.Tensor;
            await warm.data();
            dummy.dispose();
            warm.dispose();
            if (cancelled) { model.dispose(); return; }
            modelRef.current = model;
            labelsRef.current = stored;
            setDemoMode(false);
            setModelSource("indexeddb");
            setStatus("Your trained model is live");
            setIsReady(true);
            return;
          }
        }
      } catch { /* fall through */ }

      // 2) Try /public/model
      try {
        setStatus("Loading labels...");
        const lres = await fetch("/labels.json");
        if (!lres.ok) throw new Error("no labels");
        labelsRef.current = await lres.json();

        setStatus("Loading TF.js model...");
        const model = await tf.loadLayersModel("/model/model.json");
        const dummy = tf.zeros([1, SEQ_LENGTH, FEATURE_LEN]);
        const warm = model.predict(dummy) as tf.Tensor;
        await warm.data();
        dummy.dispose();
        warm.dispose();
        if (cancelled) return;
        modelRef.current = model;
        setDemoMode(false);
        setModelSource("public");
        setStatus("Model ready");
      } catch {
        if (cancelled) return;
        labelsRef.current = [
          "hello", "thank you", "yes", "no", "please",
          "sorry", "help", "water", "food", "love",
        ];
        setDemoMode(true);
        setModelSource("demo");
        setStatus("Demo mode — train your own gestures");
      } finally {
        if (!cancelled) setIsReady(true);
      }
    })();
    return () => { cancelled = true; };
  }, [modelVersion]);

  const drawSkeleton = useCallback(
    (landmarksList: any[], ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const CONN = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20],
      ];
      for (const landmarks of landmarksList) {
        ctx.strokeStyle = "rgba(140, 110, 255, 0.9)";
        ctx.lineWidth = 3;
        for (const [a, b] of CONN) {
          const pa = landmarks[a], pb = landmarks[b];
          ctx.beginPath();
          ctx.moveTo((1 - pa.x) * w, pa.y * h);
          ctx.lineTo((1 - pb.x) * w, pb.y * h);
          ctx.stroke();
        }
        for (const lm of landmarks) {
          ctx.fillStyle = "rgb(80, 220, 170)";
          ctx.beginPath();
          ctx.arc((1 - lm.x) * w, lm.y * h, 4.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    },
    [],
  );

  const onResults = useCallback(
    async (results: any) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      let keypoints = new Array(FEATURE_LEN).fill(0);
      const handsLandmarks = results.multiHandLandmarks ?? [];
      setHandVisible(handsLandmarks.length > 0);

      if (handsLandmarks.length > 0) {
        drawSkeleton(handsLandmarks, ctx, width, height);
        const h1 = handsLandmarks[0].flatMap((lm: any) => [lm.x, lm.y, lm.z]);
        const h2 = handsLandmarks[1]
          ? handsLandmarks[1].flatMap((lm: any) => [lm.x, lm.y, lm.z])
          : new Array(63).fill(0);
        keypoints = [...h1, ...h2].slice(0, FEATURE_LEN);
      }

      sequenceRef.current.push(keypoints);
      if (sequenceRef.current.length > SEQ_LENGTH) sequenceRef.current.shift();
      onFrameRef.current?.(keypoints, handsLandmarks.length);

      if (!enableInference) return;
      if (sequenceRef.current.length < SEQ_LENGTH) return;

      const now = performance.now();
      if (now - lastInferRef.current < 100) return;
      lastInferRef.current = now;

      if (modelRef.current && !demoMode) {
        const input = tf.tensor3d([sequenceRef.current]);
        const pred = modelRef.current.predict(input) as tf.Tensor;
        const probs = (await pred.data()) as Float32Array;
        input.dispose();
        pred.dispose();
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
        const conf = probs[maxIdx];
        const word = labelsRef.current[maxIdx] ?? "";
        const next = { word: conf > confidenceThreshold ? word : "", confidence: conf };
        setPrediction(next);
        if (next.word) onPrediction?.(next);
      } else if (demoMode && handsLandmarks.length > 0) {
        const sum = keypoints.reduce((a: number, b: number) => a + Math.abs(b), 0);
        const idx = Math.floor(sum * 7) % labelsRef.current.length;
        const conf = 0.88 + (Math.sin(now / 800) + 1) * 0.05;
        const word = labelsRef.current[idx];
        const next = { word, confidence: Math.min(0.99, conf) };
        setPrediction(next);
        onPrediction?.(next);
      }
    },
    [drawSkeleton, demoMode, confidenceThreshold, onPrediction, enableInference],
  );

  useEffect(() => {
    if (!isReady) return;
    let stopped = false;
    (async () => {
      const { Hands } = await import("@mediapipe/hands");
      const { Camera } = await import("@mediapipe/camera_utils");
      if (stopped) return;
      const hands = new Hands({
        locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
      });
      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
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
        try {
          await camera.start();
        } catch {
          setStatus("Camera permission denied");
        }
      }
    })();

    return () => {
      stopped = true;
      try { cameraRef.current?.stop?.(); } catch { /* noop */ }
      try { handsRef.current?.close?.(); } catch { /* noop */ }
    };
  }, [isReady, onResults]);

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
  };
}