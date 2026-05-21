import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { useHandTracking } from "@/hooks/useHandTracking";
import {
  SEQ_LENGTH,
  FEATURE_LEN,
  loadLabels,
  saveLabels,
  trainModel,
  deleteTrainedModel,
  type Sample,
} from "@/lib/gestureStore";

export const Route = createFileRoute("/train")({
  component: TrainPage,
  head: () => ({
    meta: [
      { title: "Train Your Gestures — SignSpeak" },
      {
        name: "description",
        content:
          "Record gesture samples in your browser and train a TensorFlow.js LSTM model live — no Python, no Colab.",
      },
    ],
  }),
});

function TrainPage() {
  const [labels, setLabels] = useState<string[]>(() => loadLabels());
  const [newLabel, setNewLabel] = useState("");
  const [activeLabel, setActiveLabel] = useState<string>("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [training, setTraining] = useState(false);
  const [trainLog, setTrainLog] = useState<{ epoch: number; loss: number; acc: number } | null>(null);
  const [trainHistory, setTrainHistory] = useState<{ epoch: number; loss: number; acc: number }[]>([]);
  const [modelVersion, setModelVersion] = useState(0);
  const [message, setMessage] = useState<string>("");

  const bufferRef = useRef<number[][]>([]);
  const recordingRef = useRef(false);

  useEffect(() => { recordingRef.current = recording; }, [recording]);
  useEffect(() => { saveLabels(labels); }, [labels]);

  const onFrame = useCallback((keypoints: number[]) => {
    if (!recordingRef.current) return;
    bufferRef.current.push(keypoints);
    if (bufferRef.current.length >= SEQ_LENGTH) {
      const seq = bufferRef.current.slice(0, SEQ_LENGTH);
      bufferRef.current = [];
      recordingRef.current = false;
      setRecording(false);
      setSamples((prev) => [...prev, { label: activeLabelRef.current, sequence: seq }]);
    }
  }, []);

  // Keep activeLabel accessible inside ref-driven callback
  const activeLabelRef = useRef(activeLabel);
  useEffect(() => { activeLabelRef.current = activeLabel; }, [activeLabel]);

  const { videoRef, canvasRef, handVisible, status, isReady } = useHandTracking({
    enableInference: false,
    onFrame,
  });

  const addLabel = () => {
    const v = newLabel.trim().toLowerCase();
    if (!v || labels.includes(v)) return;
    const next = [...labels, v];
    setLabels(next);
    setActiveLabel(v);
    setNewLabel("");
  };

  const removeLabel = (l: string) => {
    setLabels((prev) => prev.filter((x) => x !== l));
    setSamples((prev) => prev.filter((s) => s.label !== l));
    if (activeLabel === l) setActiveLabel("");
  };

  const startRecord = async () => {
    if (!activeLabel || recording) return;
    setMessage("");
    // 3-2-1 countdown then capture 30 frames
    for (let i = 3; i >= 1; i--) {
      setCountdown(i);
      await new Promise((r) => setTimeout(r, 700));
    }
    setCountdown(0);
    bufferRef.current = [];
    setRecording(true);
  };

  const countsByLabel = labels.map((l) => ({
    label: l,
    count: samples.filter((s) => s.label === l).length,
  }));
  const minPerLabel = countsByLabel.length
    ? Math.min(...countsByLabel.map((c) => c.count))
    : 0;
  const canTrain = labels.length >= 2 && minPerLabel >= 5 && !training;

  const handleTrain = async () => {
    if (!canTrain) return;
    setTraining(true);
    setTrainLog(null);
    setTrainHistory([]);
    setMessage("");
    try {
      await trainModel(
        samples,
        labels,
        (epoch, logs) => {
          setTrainLog({ epoch, loss: logs.loss, acc: logs.acc });
          setTrainHistory((prev) => [...prev, { epoch, ...logs }]);
        },
        40,
      );
      setMessage("Model trained and saved. Go back to the live page to try it.");
      setModelVersion((v) => v + 1);
    } catch (e: any) {
      setMessage(`Training failed: ${e?.message ?? e}`);
    } finally {
      setTraining(false);
    }
  };

  const resetAll = async () => {
    if (!confirm("Delete all samples, labels, and the trained model?")) return;
    setSamples([]);
    setLabels([]);
    setActiveLabel("");
    setTrainHistory([]);
    setTrainLog(null);
    saveLabels([]);
    await deleteTrainedModel();
    setModelVersion((v) => v + 1);
    setMessage("Cleared.");
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg" style={{ background: "var(--gradient-primary)" }} />
            <span className="font-semibold tracking-tight">SignSpeak</span>
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link to="/" className="text-muted-foreground hover:text-foreground">Live</Link>
            <Link to="/about" className="text-muted-foreground hover:text-foreground">About</Link>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 pt-10 pb-6">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Train your own gestures in the browser
        </h1>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          Add gesture labels, record 5+ samples each (30 frames per sample), then train a small
          LSTM on this page using TensorFlow.js. Everything runs locally — no uploads,
          no Python required. {modelVersion > 0 && <span className="text-foreground">Model v{modelVersion} saved to IndexedDB.</span>}
        </p>
      </section>

      <section className="mx-auto grid max-w-6xl gap-6 px-6 pb-10 lg:grid-cols-[1fr_360px]">
        {/* Camera stage */}
        <div
          className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-border bg-black"
          style={{ boxShadow: "var(--shadow-elegant)" }}
        >
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
            autoPlay muted playsInline
          />
          <canvas ref={canvasRef} width={640} height={480} className="absolute inset-0 h-full w-full" />

          <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs text-white backdrop-blur">
            <span className={`h-2 w-2 rounded-full ${isReady ? (handVisible ? "bg-emerald-400" : "bg-amber-400") : "bg-zinc-400"}`} />
            {status}
          </div>

          {countdown > 0 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-8xl font-bold text-white drop-shadow-lg">{countdown}</div>
            </div>
          )}
          {recording && (
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full border border-red-500/40 bg-red-500/90 px-5 py-2 text-sm font-medium text-white">
              <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
              Recording {bufferRef.current.length}/{SEQ_LENGTH}
            </div>
          )}
          {!activeLabel && !recording && countdown === 0 && (
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-white/20 bg-black/60 px-5 py-2 text-sm text-white">
              Pick or add a gesture to begin
            </div>
          )}
        </div>

        {/* Side panel */}
        <aside className="flex flex-col gap-4">
          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">1 · Gestures</p>
            <div className="mt-3 flex gap-2">
              <input
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") addLabel(); }}
                placeholder="e.g. hello"
                className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/30"
              />
              <button
                onClick={addLabel}
                className="rounded-lg px-3 py-2 text-sm font-medium text-primary-foreground"
                style={{ background: "var(--gradient-primary)" }}
              >
                Add
              </button>
            </div>
            <ul className="mt-3 space-y-1.5">
              {countsByLabel.map(({ label, count }) => (
                <li
                  key={label}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${
                    activeLabel === label ? "border-primary bg-primary/10" : "border-border"
                  }`}
                >
                  <button onClick={() => setActiveLabel(label)} className="flex-1 text-left">
                    <span className="font-medium">{label}</span>
                    <span className="ml-2 text-xs text-muted-foreground">{count} samples</span>
                  </button>
                  <button
                    onClick={() => removeLabel(label)}
                    className="text-xs text-muted-foreground hover:text-destructive"
                  >
                    remove
                  </button>
                </li>
              ))}
              {labels.length === 0 && (
                <li className="text-xs text-muted-foreground">
                  Add at least 2 gestures to begin (e.g. "hello", "yes").
                </li>
              )}
            </ul>
          </div>

          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">2 · Record samples</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Selected: <span className="font-medium text-foreground">{activeLabel || "—"}</span>
            </p>
            <button
              onClick={startRecord}
              disabled={!activeLabel || recording || !isReady}
              className="mt-3 w-full rounded-lg px-4 py-2.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
              style={{ background: "var(--gradient-primary)" }}
            >
              {recording ? "Recording…" : "Record sample (30 frames)"}
            </button>
            <p className="mt-2 text-xs text-muted-foreground">
              Aim for 8–15 samples per gesture, varying angle and distance.
            </p>
          </div>

          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">3 · Train</p>
            <button
              onClick={handleTrain}
              disabled={!canTrain}
              className="mt-3 w-full rounded-lg px-4 py-2.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
              style={{ background: "var(--gradient-primary)" }}
            >
              {training ? `Training… epoch ${trainLog?.epoch ?? 0}/40` : "Train model"}
            </button>
            {!canTrain && !training && (
              <p className="mt-2 text-xs text-muted-foreground">
                Need ≥ 2 labels and ≥ 5 samples per label. Current min: {minPerLabel}.
              </p>
            )}
            {trainLog && (
              <div className="mt-3 rounded-lg bg-muted/40 p-3 text-xs">
                <p>Epoch <span className="font-mono">{trainLog.epoch}</span></p>
                <p>Loss: <span className="font-mono">{trainLog.loss.toFixed(4)}</span></p>
                <p>Accuracy: <span className="font-mono">{(trainLog.acc * 100).toFixed(1)}%</span></p>
              </div>
            )}
            {message && (
              <p className="mt-3 text-xs text-foreground">{message}</p>
            )}
            {message.includes("trained") && (
              <Link
                to="/"
                className="mt-3 block rounded-lg border border-border px-3 py-2 text-center text-sm hover:bg-accent"
              >
                Open live recognizer →
              </Link>
            )}
            <button
              onClick={resetAll}
              className="mt-2 w-full rounded-lg border border-border px-3 py-2 text-xs text-muted-foreground hover:bg-accent"
            >
              Reset everything
            </button>
          </div>
        </aside>
      </section>

      {trainHistory.length > 0 && (
        <section className="mx-auto max-w-6xl px-6 pb-16">
          <h2 className="text-xl font-semibold">Training curve</h2>
          <div className="mt-3 grid grid-cols-10 gap-0.5 sm:grid-cols-20 md:grid-cols-40">
            {trainHistory.map((h) => (
              <div
                key={h.epoch}
                title={`epoch ${h.epoch} · acc ${(h.acc * 100).toFixed(1)}%`}
                className="h-16 rounded-sm bg-primary/20"
                style={{ background: `linear-gradient(to top, var(--primary) ${h.acc * 100}%, transparent ${h.acc * 100}%)` }}
              />
            ))}
          </div>
        </section>
      )}
    </main>
  );
}