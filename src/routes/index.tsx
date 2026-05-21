import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useHandTracking } from "@/hooks/useHandTracking";
import { useSentenceBuilder } from "@/hooks/useSentenceBuilder";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "SignSpeak — Real-time Sign Language Translator" },
      {
        name: "description",
        content:
          "Browser-based sign language to speech translator using MediaPipe hand tracking and an LSTM model running on TensorFlow.js.",
      },
    ],
  }),
});

function Index() {
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const sentenceBuilder = useSentenceBuilder({ ttsEnabled });
  const { addPrediction } = sentenceBuilder;

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
    vocabulary,
  } = useHandTracking({ onPrediction });

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <div
              className="h-8 w-8 rounded-lg"
              style={{ background: "var(--gradient-primary)" }}
            />
            <span className="font-semibold tracking-tight">SignSpeak</span>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <Link to="/train" className="text-muted-foreground hover:text-foreground">
              Train
            </Link>
            <Link to="/about" className="text-muted-foreground hover:text-foreground">
              About
            </Link>
            <a
              href="https://github.com"
              className="text-muted-foreground hover:text-foreground"
            >
              Docs
            </a>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 pt-10 pb-6">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Sign language → spoken English, live in your browser.
        </h1>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          MediaPipe extracts 21 hand landmarks per frame. A 30-frame sliding
          buffer feeds an LSTM that recognizes signs, then a pause-detection
          sentence builder triggers the Web Speech API.
        </p>

        {demoMode && (
          <div className="mt-5 rounded-lg border border-border bg-card p-4 text-sm">
            <strong>Demo mode active.</strong> No trained model found yet.{" "}
            <Link to="/train" className="text-primary underline underline-offset-2">
              Train your own gestures in the browser →
            </Link>{" "}
            (no Python, no Colab). Or drop an exported model into{" "}
            <code>public/model/</code>.
          </div>
        )}
      </section>

      <section className="mx-auto grid max-w-6xl gap-6 px-6 pb-10 lg:grid-cols-[1fr_360px]">
        {/* Webcam stage */}
        <div
          className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-border bg-black"
          style={{ boxShadow: "var(--shadow-elegant)" }}
        >
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

          {/* Status pill */}
          <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs text-white backdrop-blur">
            <span
              className={`h-2 w-2 rounded-full ${
                isReady ? (handVisible ? "bg-emerald-400" : "bg-amber-400") : "bg-zinc-400"
              }`}
            />
            {status}
          </div>

          {/* Live prediction */}
          {prediction.word && (
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-white/20 bg-white/95 px-5 py-2 text-lg font-medium text-foreground shadow-lg">
              {prediction.word}
              <span className="ml-2 text-sm text-muted-foreground">
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
          )}
        </div>

        {/* Side panel */}
        <aside className="flex flex-col gap-4">
          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">
              Current sentence
            </p>
            <p className="mt-2 min-h-[3rem] text-lg leading-relaxed">
              {sentenceBuilder.words.length > 0 ? (
                sentenceBuilder.words.join(" ")
              ) : (
                <span className="text-muted-foreground">Start signing…</span>
              )}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={sentenceBuilder.speakNow}
                className="rounded-lg px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
                style={{ background: "var(--gradient-primary)" }}
              >
                Speak
              </button>
              <button
                onClick={() => setTtsEnabled((v) => !v)}
                className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-accent"
              >
                Auto-TTS: {ttsEnabled ? "ON" : "OFF"}
              </button>
              <button
                onClick={sentenceBuilder.clear}
                className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-accent"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">
              Recognized vocabulary ({vocabulary.length})
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {vocabulary.map((w) => (
                <span
                  key={w}
                  className={`rounded-full border px-2.5 py-1 text-xs ${
                    prediction.word === w
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-muted-foreground"
                  }`}
                >
                  {w}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-card p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">
              Session history
            </p>
            {sentenceBuilder.history.length === 0 ? (
              <p className="mt-2 text-sm text-muted-foreground">
                Completed sentences appear here.
              </p>
            ) : (
              <ul className="mt-2 divide-y divide-border">
                {sentenceBuilder.history.map((h, i) => (
                  <li
                    key={i}
                    className="flex items-center justify-between gap-3 py-2 text-sm"
                  >
                    <span className="truncate">{h.text}</span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {h.time}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <h2 className="text-xl font-semibold">Pipeline</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {[
            ["Webcam", "getUserMedia"],
            ["MediaPipe", "21 landmarks/hand"],
            ["Buffer", "30-frame window"],
            ["LSTM", "TF.js inference"],
            ["Sentence", "1.5s pause = end"],
            ["TTS", "Web Speech API"],
          ].map(([title, sub]) => (
            <div key={title} className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-medium">{title}</p>
              <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
