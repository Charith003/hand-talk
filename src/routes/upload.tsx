import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { CheckCircle2, FileImage, FileVideo, Loader2, Trash2 } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import {
  FEATURE_LEN,
  MIN_SAMPLES_PER_LABEL,
  TRAINING_STEPS,
  loadLabels,
  saveLabels,
  loadSamples,
  saveSamples,
  sequenceFromFrames,
  trainModel,
  type Sample,
} from "@/lib/gestureStore";

export const Route = createFileRoute("/upload")({
  component: UploadPage,
  head: () => ({
    meta: [
      { title: "Upload Dataset — SignSpeak" },
      {
        name: "description",
        content:
          "Upload images or videos of signs, auto-extract hand landmarks, and train a recognition model — no live recording needed.",
      },
    ],
  }),
});

type Landmark = { x?: number; y?: number; z?: number };
type HandResults = { multiHandLandmarks?: Landmark[][] };
type HandsHandle = {
  setOptions: (opts: Record<string, number>) => void;
  onResults: (cb: (r: HandResults) => void) => void;
  send: (input: {
    image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;
  }) => Promise<void>;
  close?: () => void;
};

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

function extractKeypoints(hands: Landmark[][]) {
  const out: number[] = [];
  for (const h of hands.slice(0, 2)) out.push(...normalizeHand(h));
  while (out.length < FEATURE_LEN) out.push(0);
  return out.slice(0, FEATURE_LEN);
}

function UploadPage() {
  const [labels, setLabels] = useState<string[]>(() => loadLabels());
  const [samples, setSamples] = useState<Sample[]>(() => loadSamples());
  const [activeLabel, setActiveLabel] = useState<string>(() => loadLabels()[0] ?? "");
  const [newLabel, setNewLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number; file: string }>({
    current: 0,
    total: 0,
    file: "",
  });
  const [message, setMessage] = useState("");
  const [training, setTraining] = useState(false);
  const [trainLog, setTrainLog] = useState<{ epoch: number; loss: number; acc: number } | null>(
    null,
  );

  const handsRef = useRef<HandsHandle | null>(null);
  const lastResultsRef = useRef<HandResults | null>(null as HandResults | null);

  useEffect(() => saveLabels(labels), [labels]);
  useEffect(() => saveSamples(samples), [samples]);

  async function ensureHands(): Promise<HandsHandle> {
    if (handsRef.current) return handsRef.current;
    const { Hands } = await import("@mediapipe/hands");
    const h = new Hands({
      locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
    }) as unknown as HandsHandle;
    h.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    h.onResults((r) => {
      lastResultsRef.current = r;
    });
    handsRef.current = h;
    return h;
  }

  function addLabel() {
    const v = newLabel.trim().toLowerCase().replace(/\s+/g, " ");
    if (!v || labels.includes(v)) return;
    setLabels((p) => [...p, v]);
    setActiveLabel(v);
    setNewLabel("");
  }

  function removeLabel(label: string) {
    setLabels((p) => p.filter((l) => l !== label));
    setSamples((p) => p.filter((s) => s.label !== label));
    if (activeLabel === label) setActiveLabel(labels.find((l) => l !== label) ?? "");
  }

  async function processImage(file: File): Promise<number[][] | null> {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error("image load failed"));
        img.src = url;
      });
      const h = await ensureHands();
      lastResultsRef.current = null as HandResults | null;
      await h.send({ image: img });
      const res: HandResults | null = lastResultsRef.current;
      const hands = res?.multiHandLandmarks ?? [];
      if (!hands.length) return null;
      const kp = extractKeypoints(hands);
      return sequenceFromFrames([kp]);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function processVideo(file: File): Promise<number[][] | null> {
    const url = URL.createObjectURL(file);
    try {
      const video = document.createElement("video");
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      await new Promise<void>((res, rej) => {
        video.onloadedmetadata = () => res();
        video.onerror = () => rej(new Error("video load failed"));
      });
      const duration = Math.max(0.5, video.duration || 1);
      const h = await ensureHands();
      const frames: number[][] = [];
      const steps = 30;
      for (let i = 0; i < steps; i++) {
        const t = (i / (steps - 1)) * duration;
        await new Promise<void>((res) => {
          video.onseeked = () => res();
          video.currentTime = Math.min(duration - 0.01, t);
        });
        lastResultsRef.current = null as HandResults | null;
        await h.send({ image: video });
        const res: HandResults | null = lastResultsRef.current;
        const hands = res?.multiHandLandmarks ?? [];
        if (hands.length) frames.push(extractKeypoints(hands));
      }
      return sequenceFromFrames(frames);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function onFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!activeLabel) {
      setMessage("Pick a gesture label first.");
      return;
    }
    if (!files.length) return;
    setBusy(true);
    setMessage("");
    let added = 0;
    let skipped = 0;
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      setProgress({ current: i + 1, total: files.length, file: f.name });
      try {
        const seq = f.type.startsWith("video/") ? await processVideo(f) : await processImage(f);
        if (seq) {
          setSamples((prev) => [...prev, { label: activeLabel, sequence: seq }]);
          added++;
        } else {
          skipped++;
        }
      } catch {
        skipped++;
      }
    }
    setBusy(false);
    setProgress({ current: 0, total: 0, file: "" });
    setMessage(
      `Added ${added} sample${added === 1 ? "" : "s"} for "${activeLabel}"` +
        (skipped ? ` · skipped ${skipped} (no hand detected)` : ""),
    );
  }

  async function handleTrain() {
    const counts = labels.map((l) => samples.filter((s) => s.label === l).length);
    const minPer = counts.length ? Math.min(...counts) : 0;
    if (labels.length < 2 || minPer < MIN_SAMPLES_PER_LABEL) {
      setMessage(
        `Need 2+ labels with at least ${MIN_SAMPLES_PER_LABEL} samples each before training.`,
      );
      return;
    }
    setTraining(true);
    setTrainLog(null);
    setMessage("Training started...");
    try {
      await trainModel(samples, labels, (epoch, logs) => setTrainLog({ epoch, ...logs }), TRAINING_STEPS);
      setMessage("Model trained and saved. Open the live recognizer to test it.");
    } catch (err) {
      setMessage(`Training failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTraining(false);
    }
  }

  const countsByLabel = labels.map((label) => ({
    label,
    count: samples.filter((s) => s.label === label).length,
  }));

  return (
    <main className="min-h-screen bg-background text-foreground">
      <AppHeader title="Dataset Upload" subtitle="images · videos · auto-train" />

      <section className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:py-10">
        <div className="mb-6">
          <h1 className="text-3xl font-semibold sm:text-4xl">
            Train from uploaded images or videos
          </h1>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            Pick a gesture label, then upload <strong>images</strong> (one sign per image) or
            <strong> short videos</strong> (one sign per clip). Hand landmarks are extracted in your
            browser — no upload to any server.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
          <section className="rounded-3xl border border-border bg-card p-6 shadow-sm">
            <p className="text-xs uppercase text-muted-foreground">1 · Choose label</p>
            <div className="mt-3 flex gap-2">
              <input
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") addLabel();
                }}
                placeholder="e.g. hello"
                className="min-w-0 flex-1 rounded-xl border border-input bg-background px-3 py-3 text-sm outline-none focus:ring-2 focus:ring-ring"
              />
              <button
                type="button"
                onClick={addLabel}
                className="rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
              >
                Add label
              </button>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {countsByLabel.map(({ label, count }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setActiveLabel(label)}
                  className={`group inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${
                    activeLabel === label
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border bg-background text-muted-foreground hover:bg-accent"
                  }`}
                >
                  <span className="font-medium">{label}</span>
                  <span className="text-xs opacity-70">· {count}</span>
                  <span
                    role="button"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      removeLabel(label);
                    }}
                    className="rounded p-0.5 opacity-60 hover:bg-destructive/10 hover:text-destructive hover:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </span>
                </button>
              ))}
              {labels.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  Add at least two labels (e.g. hello and yes).
                </p>
              )}
            </div>

            <p className="mt-8 text-xs uppercase text-muted-foreground">
              2 · Upload images or videos
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Selected: <span className="font-medium text-foreground">{activeLabel || "none"}</span>
            </p>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label
                className={`flex cursor-pointer items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-border px-4 py-8 text-sm font-medium transition hover:border-primary hover:bg-primary/5 ${!activeLabel || busy ? "pointer-events-none opacity-50" : ""}`}
              >
                <FileImage className="h-5 w-5" />
                Upload images
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={onFiles}
                  disabled={!activeLabel || busy}
                />
              </label>
              <label
                className={`flex cursor-pointer items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-border px-4 py-8 text-sm font-medium transition hover:border-primary hover:bg-primary/5 ${!activeLabel || busy ? "pointer-events-none opacity-50" : ""}`}
              >
                <FileVideo className="h-5 w-5" />
                Upload videos
                <input
                  type="file"
                  accept="video/*"
                  multiple
                  className="hidden"
                  onChange={onFiles}
                  disabled={!activeLabel || busy}
                />
              </label>
            </div>

            {busy && (
              <div className="mt-4 rounded-xl border border-border bg-secondary p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing {progress.current}/{progress.total}:{" "}
                  <span className="truncate">{progress.file}</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-background">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${(progress.current / Math.max(1, progress.total)) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {message && !busy && (
              <p className="mt-4 rounded-xl bg-secondary p-3 text-sm text-secondary-foreground">
                {message}
              </p>
            )}
          </section>

          <aside className="grid gap-4">
            <section className="rounded-3xl border-2 border-primary/40 bg-card p-5 shadow-md ring-1 ring-primary/10">
              <p className="text-xs uppercase text-primary">3 · Train model</p>
              <button
                type="button"
                onClick={handleTrain}
                disabled={training}
                className="mt-3 w-full rounded-xl bg-primary px-4 py-3 text-base font-semibold text-primary-foreground transition hover:opacity-90 disabled:pointer-events-none disabled:opacity-50"
              >
                {training ? `Training step ${trainLog?.epoch ?? 0}/${TRAINING_STEPS}` : "Train model now"}
              </button>
              {!training && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Need 2+ labels and {MIN_SAMPLES_PER_LABEL}+ samples each. Lowest currently:{" "}
                  {labels.length ? Math.min(...countsByLabel.map((item) => item.count)) : 0}.
                </p>
              )}
              {trainLog && (
                <div className="mt-3 rounded-xl bg-secondary p-3 text-xs text-secondary-foreground">
                  <p>
                    Epoch <span className="font-mono">{trainLog.epoch}</span>
                  </p>
                  <p>
                    Loss: <span className="font-mono">{trainLog.loss.toFixed(4)}</span>
                  </p>
                  <p>
                    Accuracy: <span className="font-mono">{(trainLog.acc * 100).toFixed(1)}%</span>
                  </p>
                </div>
              )}
              {message.includes("trained") && (
                <Link
                  to="/"
                  className="mt-3 flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm font-semibold transition hover:bg-accent"
                >
                  <CheckCircle2 className="h-4 w-4" /> Open live recognizer
                </Link>
              )}
            </section>

            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <p className="text-xs uppercase text-muted-foreground">Tips for best accuracy</p>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                <li>• Use 5–15 examples per label.</li>
                <li>• Vary background and lighting slightly.</li>
                <li>• Make sure the hand fills the frame.</li>
                <li>• Short videos (1–3 s) work best.</li>
                <li>• Images with no detected hand are skipped.</li>
              </ul>
            </section>
          </aside>
        </div>
      </section>
    </main>
  );
}
