import { createFileRoute, Link } from "@tanstack/react-router";
import { Mic, Mic2, Play, RotateCcw, Sparkles, Volume2, Zap, Brain } from "lucide-react";
import { useCallback, useState } from "react";
import { useHandTracking } from "@/hooks/useHandTracking";
import { useSentenceBuilder } from "@/hooks/useSentenceBuilder";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "SignSpeak — Live Sign Translator" },
      {
        name: "description",
        content:
          "Train gestures in the browser, recognize signs from your webcam, build sentences, and speak them with browser text-to-speech.",
      },
    ],
  }),
});

function Index() {
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [mode, setMode] = useState<"heuristic" | "model">("heuristic");
  const sentenceBuilder = useSentenceBuilder({ ttsEnabled });
  const { addPrediction, unlockTts } = sentenceBuilder;

  const onPrediction = useCallback(
    (p: { word: string; confidence: number }) => addPrediction(p.word, p.confidence),
    [addPrediction],
  );

  const {
    videoRef,
    canvasRef,
    prediction,
    isReady,
    demoMode,
    status,
    handVisible,
    modelSource,
    cameraStarted,
    cameraError,
    startCamera,
    vocabulary,
  } = useHandTracking({ onPrediction, mode });

  const enableVoice = () => {
    unlockTts();
    setTtsEnabled(true);
  };

  const speakCurrent = () => {
    unlockTts();
    sentenceBuilder.speakNow();
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-card/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
          <Link to="/" className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
              <Mic2 className="h-5 w-5" />
            </span>
            <span>
              <span className="block text-base font-semibold">SignSpeak</span>
              <span className="block text-xs text-muted-foreground">train · sign · speak</span>
            </span>
          </Link>
          <nav className="flex items-center gap-2 text-sm">
            <Link
              to="/train"
              className="rounded-lg bg-primary px-4 py-2 font-medium text-primary-foreground transition hover:opacity-90"
            >
              Train
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
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
          <div className="overflow-hidden rounded-3xl border border-border bg-stage text-stage-foreground shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stage-foreground/10 px-5 py-4">
              <div>
                <h1 className="text-2xl font-semibold sm:text-3xl">Live sign recognition</h1>
                <p className="mt-1 text-sm text-stage-foreground/70">
                  {modelSource === "heuristic"
                    ? "Pre-trained signs active — no training required."
                    : demoMode
                      ? "No trained model yet — switch to pre-trained or train your own."
                      : modelSource === "indexeddb"
                        ? "Using your browser-trained gesture model."
                        : "Using the project gesture model."}
                </p>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-stage-foreground/10 bg-stage-foreground/10 px-3 py-1.5 text-xs">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    cameraStarted && handVisible
                      ? "bg-live"
                      : cameraStarted
                        ? "bg-caution"
                        : "bg-muted-foreground"
                  }`}
                />
                {status}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 border-b border-stage-foreground/10 px-5 py-3">
              <button
                type="button"
                onClick={() => setMode("heuristic")}
                className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition ${
                  mode === "heuristic"
                    ? "bg-primary text-primary-foreground"
                    : "bg-stage-foreground/10 text-stage-foreground/80 hover:bg-stage-foreground/20"
                }`}
              >
                <Zap className="h-3.5 w-3.5" />
                Pre-trained signs
              </button>
              <button
                type="button"
                onClick={() => setMode("model")}
                className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition ${
                  mode === "model"
                    ? "bg-primary text-primary-foreground"
                    : "bg-stage-foreground/10 text-stage-foreground/80 hover:bg-stage-foreground/20"
                }`}
              >
                <Brain className="h-3.5 w-3.5" />
                Trained model
              </button>
              {mode === "heuristic" && vocabulary.length > 0 && (
                <span className="ml-auto text-xs text-stage-foreground/60">
                  {vocabulary.length} built-in signs
                </span>
              )}
            </div>

            <div className="relative aspect-[4/3] bg-stage sm:aspect-video lg:aspect-[4/3]">
              <video
                ref={videoRef}
                className="absolute inset-0 h-full w-full -scale-x-100 object-cover opacity-95"
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
                    <h2 className="mt-5 text-2xl font-semibold">Start the camera first</h2>
                    <p className="mt-2 text-sm text-stage-foreground/70">
                      Browser camera access must be started by a click. This prevents the
                      blocked-camera error you were seeing.
                    </p>
                    {cameraError && (
                      <p className="mt-3 rounded-xl border border-stage-foreground/10 bg-stage-foreground/10 p-3 text-sm">
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

              {cameraStarted && !demoMode && prediction.word && (
                <div className="absolute bottom-5 left-1/2 w-[min(90%,420px)] -translate-x-1/2 rounded-2xl border border-stage-foreground/10 bg-stage-foreground px-5 py-4 text-center text-stage shadow-xl">
                  <p className="text-xs uppercase text-stage/60">stable prediction</p>
                  <p className="mt-1 text-3xl font-semibold">{prediction.word}</p>
                  <p className="mt-1 text-sm text-stage/70">
                    {Math.round(prediction.confidence * 100)}% confidence
                  </p>
                </div>
              )}

              {cameraStarted && demoMode && (
                <div className="absolute bottom-5 left-1/2 w-[min(90%,460px)] -translate-x-1/2 rounded-2xl border border-stage-foreground/10 bg-stage-foreground/95 p-4 text-center text-stage shadow-xl">
                  <p className="text-sm font-medium">No trained model is active</p>
                  <p className="mt-1 text-xs text-stage/70">
                    Go to Train, record at least 5 samples for 2+ gestures, then come back here.
                  </p>
                  <Link
                    to="/train"
                    className="mt-3 inline-flex rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"
                  >
                    Open trainer
                  </Link>
                </div>
              )}
            </div>
          </div>

          <aside className="grid gap-4">
            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs uppercase text-muted-foreground">Current sentence</p>
                  <p className="mt-2 min-h-[5rem] text-2xl font-semibold leading-snug">
                    {sentenceBuilder.words.length > 0 ? (
                      sentenceBuilder.words.join(" ")
                    ) : (
                      <span className="text-muted-foreground">Waiting for signs</span>
                    )}
                  </p>
                </div>
                <span className="rounded-full bg-secondary px-3 py-1 text-xs text-secondary-foreground">
                  {sentenceBuilder.words.length} words
                </span>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onPointerDown={unlockTts}
                  onClick={speakCurrent}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
                >
                  <Volume2 className="h-4 w-4" />
                  Speak
                </button>
                <button
                  type="button"
                  onPointerDown={unlockTts}
                  onClick={enableVoice}
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm font-medium transition hover:bg-accent"
                >
                  <Mic className="h-4 w-4" />
                  Enable voice
                </button>
                <button
                  type="button"
                  onPointerDown={unlockTts}
                  onClick={() => setTtsEnabled((v) => !v)}
                  className="rounded-xl border border-border bg-background px-4 py-3 text-sm font-medium transition hover:bg-accent"
                >
                  Auto-TTS {ttsEnabled ? "ON" : "OFF"}
                </button>
                <button
                  type="button"
                  onClick={sentenceBuilder.clear}
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm font-medium transition hover:bg-accent"
                >
                  <RotateCcw className="h-4 w-4" />
                  Clear
                </button>
              </div>

              {!sentenceBuilder.ttsReady && (
                <p className="mt-3 rounded-xl bg-secondary p-3 text-xs text-secondary-foreground">
                  Tap Enable voice once before using Auto-TTS. Browsers block speech until the user
                  interacts.
                </p>
              )}
            </section>

            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <p className="text-xs uppercase text-muted-foreground">Recognition status</p>
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-muted-foreground">Camera</span>
                  <span className="font-medium">{cameraStarted ? "Running" : "Not started"}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-muted-foreground">Hands</span>
                  <span className="font-medium">{handVisible ? "Detected" : "Not visible"}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-muted-foreground">Model</span>
                  <span className="font-medium">
                    {demoMode
                      ? "Needs training"
                      : modelSource === "indexeddb"
                        ? "Browser trained"
                        : "Project file"}
                  </span>
                </div>
              </div>
              <Link
                to="/train"
                className="mt-5 flex items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm font-semibold transition hover:bg-accent"
              >
                <Sparkles className="h-4 w-4" />
                Train or improve model
              </Link>
            </section>

            <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
              <p className="text-xs uppercase text-muted-foreground">Session history</p>
              {sentenceBuilder.history.length === 0 ? (
                <p className="mt-3 text-sm text-muted-foreground">Spoken sentences appear here.</p>
              ) : (
                <ul className="mt-3 divide-y divide-border">
                  {sentenceBuilder.history.map((h, i) => (
                    <li key={i} className="flex items-center justify-between gap-3 py-2 text-sm">
                      <span className="truncate">{h.text}</span>
                      <span className="shrink-0 text-xs text-muted-foreground">{h.time}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>
        </div>
      </section>
    </main>
  );
}
