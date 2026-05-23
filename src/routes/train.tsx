import { createFileRoute, Link } from "@tanstack/react-router";
import { CheckCircle2, Play, Plus, Trash2, Video, Upload, Radio, Brain } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useHandTracking } from "@/hooks/useHandTracking";
import {
  SEQ_LENGTH,
  loadLabels,
  saveLabels,
  loadSamples,
  saveSamples,
  trainModel,
  deleteTrainedModel,
  type Sample,
} from "@/lib/gestureStore";

export const Route = createFileRoute("/train")({
  component: TrainPage,
  head: () => ({
    meta: [
      { title: "Train Gestures — SignSpeak" },
      {
        name: "description",
        content:
          "Record gesture samples in your browser and train a TensorFlow.js sign recognition model locally.",
      },
    ],
  }),
});

function TrainPage() {
  const [labels, setLabels] = useState<string[]>(() => loadLabels());
  const [newLabel, setNewLabel] = useState("");
  const [activeLabel, setActiveLabel] = useState<string>(() => loadLabels()[0] ?? "");
  const [samples, setSamples] = useState<Sample[]>(() => loadSamples());
  const [recording, setRecording] = useState(false);
  const [recordedFrames, setRecordedFrames] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [training, setTraining] = useState(false);
  const [trainLog, setTrainLog] = useState<{ epoch: number; loss: number; acc: number } | null>(
    null,
  );
  const [trainHistory, setTrainHistory] = useState<{ epoch: number; loss: number; acc: number }[]>(
    [],
  );
  const [modelVersion, setModelVersion] = useState(0);
  const [message, setMessage] = useState<string>("");

  const bufferRef = useRef<number[][]>([]);
  const recordingRef = useRef(false);
  const activeLabelRef = useRef(activeLabel);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);
  useEffect(() => {
    activeLabelRef.current = activeLabel;
  }, [activeLabel]);
  useEffect(() => {
    saveLabels(labels);
  }, [labels]);
  useEffect(() => {
    saveSamples(samples);
  }, [samples]);

  const onFrame = useCallback((keypoints: number[], handCount: number) => {
    if (!recordingRef.current || handCount === 0) return;
    bufferRef.current.push(keypoints);
    setRecordedFrames(bufferRef.current.length);
    if (bufferRef.current.length >= SEQ_LENGTH) {
      const seq = bufferRef.current.slice(0, SEQ_LENGTH);
      bufferRef.current = [];
      recordingRef.current = false;
      setRecording(false);
      setRecordedFrames(0);
      setSamples((prev) => [...prev, { label: activeLabelRef.current, sequence: seq }]);
      setMessage(`Saved sample for ${activeLabelRef.current}.`);
    }
  }, []);

  const {
    videoRef,
    canvasRef,
    handVisible,
    status,
    isReady,
    cameraStarted,
    cameraError,
    startCamera,
  } = useHandTracking({
    enableInference: false,
    onFrame,
    modelVersion,
  });

  const addLabel = () => {
    const value = newLabel.trim().toLowerCase().replace(/\s+/g, " ");
    if (!value || labels.includes(value)) return;
    setLabels((prev) => [...prev, value]);
    setActiveLabel(value);
    setNewLabel("");
    setMessage(`Gesture "${value}" added.`);
  };

  const removeLabel = (label: string) => {
    setLabels((prev) => prev.filter((item) => item !== label));
    setSamples((prev) => prev.filter((sample) => sample.label !== label));
    if (activeLabel === label) setActiveLabel(labels.find((item) => item !== label) ?? "");
  };

  const startRecord = async () => {
    if (!activeLabel || recording || countdown > 0 || !cameraStarted) return;
    if (!handVisible) {
      setMessage("Show your hand clearly before recording.");
      return;
    }
    setMessage("");
    for (let i = 3; i >= 1; i--) {
      setCountdown(i);
      await new Promise((resolve) => setTimeout(resolve, 650));
    }
    setCountdown(0);
    bufferRef.current = [];
    setRecordedFrames(0);
    recordingRef.current = true;
    setRecording(true);
  };

  const countsByLabel = labels.map((label) => ({
    label,
    count: samples.filter((sample) => sample.label === label).length,
  }));
  const minPerLabel = countsByLabel.length
    ? Math.min(...countsByLabel.map((item) => item.count))
    : 0;
  const MIN_PER_LABEL = 3;
  const canTrain = labels.length >= 2 && minPerLabel >= MIN_PER_LABEL && !training;

  const handleTrain = async () => {
    if (!canTrain) return;
    setTraining(true);
    setTrainLog(null);
    setTrainHistory([]);
    setMessage("Training started. Keep this tab open.");
    try {
      await trainModel(
        samples,
        labels,
        (epoch, logs) => {
          setTrainLog({ epoch, loss: logs.loss, acc: logs.acc });
          setTrainHistory((prev) => [...prev, { epoch, ...logs }]);
        },
        50,
      );
      setMessage("Model trained and saved. Open the live recognizer to test it.");
      setModelVersion((v) => v + 1);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      setMessage(`Training failed: ${errorMessage}`);
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
    saveSamples([]);
    await deleteTrainedModel();
    setModelVersion((v) => v + 1);
    setMessage("Everything has been cleared.");
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-card/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
          <Link to="/" className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Video className="h-5 w-5" />
            </span>
            <span>
              <span className="block text-base font-semibold">Gesture Trainer</span>
              <span className="block text-xs text-muted-foreground">local browser training</span>
            </span>
          </Link>
          <nav className="flex items-center gap-2 text-sm">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
            >
              <Radio className="h-4 w-4" /> Live
            </Link>
            <Link
              to="/train"
              className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-2 font-semibold text-primary-foreground transition hover:opacity-90"
            >
              <Brain className="h-4 w-4" /> Train
            </Link>
            <Link
              to="/upload"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
            >
              <Upload className="h-4 w-4" /> Upload dataset
            </Link>
            <Link
              to="/about"
              className="rounded-lg px-3 py-2 text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
            >
              About
            </Link>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:py-10">
        <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
          <div>
            <h1 className="text-3xl font-semibold sm:text-4xl">Train accurate custom gestures</h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              Add gestures, record clear samples, then train locally. For better accuracy, use the
              same lighting and camera position during training and live recognition.
            </p>
          </div>
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-xl border border-border bg-card px-4 py-3 text-sm font-semibold transition hover:bg-accent"
          >
            Test live recognizer
          </Link>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_400px]">
          <section className="overflow-hidden rounded-3xl border border-border bg-stage text-stage-foreground shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stage-foreground/10 px-5 py-4">
              <div>
                <h2 className="text-xl font-semibold">Camera capture</h2>
                <p className="text-sm text-stage-foreground/70">{status}</p>
              </div>
              <div className="rounded-full border border-stage-foreground/10 bg-stage-foreground/10 px-3 py-1.5 text-xs">
                {recording
                  ? `Recording ${recordedFrames}/${SEQ_LENGTH}`
                  : handVisible
                    ? "Hand detected"
                    : "Show hand"}
              </div>
            </div>

            <div className="relative aspect-[4/3] bg-stage sm:aspect-video lg:aspect-[4/3]">
              <video
                ref={videoRef}
                className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
                autoPlay
                muted
                playsInline
              />
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute inset-0 h-full w-full"
              />

              {!cameraStarted && (
                <div className="absolute inset-0 flex items-center justify-center bg-stage/85 px-6 text-center backdrop-blur-sm">
                  <div className="max-w-md">
                    <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
                      <Play className="h-7 w-7" />
                    </div>
                    <h3 className="mt-5 text-2xl font-semibold">Start camera to record</h3>
                    <p className="mt-2 text-sm text-stage-foreground/70">
                      The camera now starts only from this button, so browser permission works
                      reliably.
                    </p>
                    {cameraError && (
                      <p className="mt-3 rounded-xl bg-stage-foreground/10 p-3 text-sm">
                        {cameraError}
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={startCamera}
                      className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
                    >
                      <Play className="h-4 w-4" />
                      Start camera
                    </button>
                  </div>
                </div>
              )}

              {countdown > 0 && (
                <div className="absolute inset-0 flex items-center justify-center bg-stage/40">
                  <div className="text-8xl font-bold text-stage-foreground drop-shadow-lg">
                    {countdown}
                  </div>
                </div>
              )}

              {recording && (
                <div className="absolute bottom-5 left-1/2 w-[min(90%,360px)] -translate-x-1/2 rounded-2xl bg-stage-foreground p-4 text-center text-stage shadow-xl">
                  <p className="font-semibold">Hold the sign steady</p>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-stage/15">
                    <div
                      className="h-full rounded-full bg-primary transition-all"
                      style={{ width: `${Math.min(100, (recordedFrames / SEQ_LENGTH) * 100)}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </section>

          <aside className="grid gap-4">
            <section className="rounded-3xl border-2 border-primary/40 bg-card p-5 shadow-md ring-1 ring-primary/10">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs uppercase text-primary">Train model</p>
                <span className="rounded-full bg-secondary px-2.5 py-0.5 text-xs text-secondary-foreground">
                  {labels.length} labels · {samples.length} samples
                </span>
              </div>
              <button
                type="button"
                onClick={handleTrain}
                disabled={!canTrain}
                className="mt-3 w-full rounded-xl bg-primary px-4 py-3 text-base font-semibold text-primary-foreground transition hover:opacity-90 disabled:pointer-events-none disabled:opacity-50"
              >
                {training
                  ? `Training epoch ${trainLog?.epoch ?? 0}/50`
                  : canTrain
                    ? "Train model now"
                    : "Train model"}
              </button>
              {!canTrain && !training && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Need 2+ labels and {MIN_PER_LABEL}+ samples each. Lowest currently: {minPerLabel}.
                </p>
              )}
              {trainLog && (
                <div className="mt-3 rounded-xl bg-secondary p-3 text-xs text-secondary-foreground">
                  <p>Epoch <span className="font-mono">{trainLog.epoch}</span></p>
                  <p>Loss: <span className="font-mono">{trainLog.loss.toFixed(4)}</span></p>
                  <p>Accuracy: <span className="font-mono">{(trainLog.acc * 100).toFixed(1)}%</span></p>
                </div>
              )}
              {message && (
                <p className="mt-3 rounded-xl bg-secondary p-3 text-xs text-secondary-foreground">
                  {message}
                </p>
              )}
              {message.includes("trained") && (
                <Link
                  to="/"
                  className="mt-3 flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm font-semibold transition hover:bg-accent"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  Open live recognizer
                </Link>
              )}
            </section>

            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <p className="text-xs uppercase text-muted-foreground">1 · Gesture labels</p>
              <div className="mt-3 flex gap-2">
                <input
                  value={newLabel}
                  onChange={(e) => setNewLabel(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") addLabel();
                  }}
                  placeholder="e.g. hello"
                  className="min-w-0 flex-1 rounded-xl border border-input bg-background px-3 py-3 text-sm outline-none transition focus:ring-2 focus:ring-ring"
                />
                <button
                  type="button"
                  onClick={addLabel}
                  className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
                >
                  <Plus className="h-4 w-4" />
                  Add
                </button>
              </div>
              <ul className="mt-4 space-y-2">
                {countsByLabel.map(({ label, count }) => (
                  <li
                    key={label}
                    className={`flex items-center justify-between gap-2 rounded-xl border p-2 text-sm ${
                      activeLabel === label ? "border-primary bg-primary/10" : "border-border"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => setActiveLabel(label)}
                      className="min-w-0 flex-1 px-2 py-1 text-left"
                    >
                      <span className="block truncate font-medium">{label}</span>
                      <span className="text-xs text-muted-foreground">
                        {count} / 5 minimum samples
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => removeLabel(label)}
                      className="rounded-lg p-2 text-muted-foreground transition hover:bg-accent hover:text-destructive"
                      aria-label={`Remove ${label}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </li>
                ))}
                {labels.length === 0 && (
                  <li className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
                    Add at least two gestures such as hello and yes.
                  </li>
                )}
              </ul>
            </section>

            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <p className="text-xs uppercase text-muted-foreground">2 · Record sample</p>
              <p className="mt-2 text-sm text-muted-foreground">
                Selected:{" "}
                <span className="font-medium text-foreground">{activeLabel || "none"}</span>
              </p>
              <button
                type="button"
                onClick={startRecord}
                disabled={!activeLabel || recording || countdown > 0 || !cameraStarted || !isReady}
                className="mt-4 w-full rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:pointer-events-none disabled:opacity-50"
              >
                {recording ? "Recording…" : countdown > 0 ? "Get ready…" : "Record 30-frame sample"}
              </button>
              <p className="mt-3 text-xs text-muted-foreground">
                Record at least {MIN_PER_LABEL} samples per gesture (8–15 for best accuracy).
              </p>
              <button
                type="button"
                onClick={resetAll}
                className="mt-3 w-full rounded-xl border border-border bg-background px-4 py-2 text-xs text-muted-foreground transition hover:bg-accent hover:text-foreground"
              >
                Reset everything
              </button>
            </section>
          </aside>
        </div>
      </section>

      {trainHistory.length > 0 && (
        <section className="mx-auto max-w-7xl px-5 pb-12 sm:px-8">
          <div className="rounded-3xl border border-border bg-card p-5 shadow-sm">
            <h2 className="text-xl font-semibold">Training accuracy</h2>
            <div className="mt-4 grid grid-cols-10 gap-1 sm:grid-cols-25 md:grid-cols-50">
              {trainHistory.map((h) => (
                <div
                  key={h.epoch}
                  title={`epoch ${h.epoch} · acc ${(h.acc * 100).toFixed(1)}%`}
                  className="h-16 rounded bg-secondary"
                >
                  <div
                    className="mt-auto h-full rounded bg-primary"
                    style={{ height: `${Math.max(4, h.acc * 100)}%` }}
                  />
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
