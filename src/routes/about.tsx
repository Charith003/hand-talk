import { createFileRoute, Link } from "@tanstack/react-router";

export const Route = createFileRoute("/about")({
  component: About,
  head: () => ({
    meta: [
      { title: "About SignSpeak — Architecture & Training" },
      {
        name: "description",
        content:
          "How SignSpeak works: MediaPipe hand tracking, LSTM gesture recognition, sentence builder with pause detection, and Web Speech API output.",
      },
    ],
  }),
});

function About() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">
        ← Back
      </Link>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">
        How SignSpeak works
      </h1>

      <h2 className="mt-8 text-lg font-semibold">Stages</h2>
      <ol className="mt-2 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-muted-foreground">
        <li>
          <strong>Data collection</strong> — record 30 samples × 30 frames per
          class with <code>python/collect_data.py</code>.
        </li>
        <li>
          <strong>Training</strong> — train a 3-layer stacked LSTM on Colab GPU
          using <code>python/train_lstm.py</code>.
        </li>
        <li>
          <strong>Export</strong> — convert Keras model to TF.js with{" "}
          <code>python/export_tfjs.py</code> (float16 quantized).
        </li>
        <li>
          <strong>Deploy</strong> — drop <code>model.json</code> + shards into{" "}
          <code>public/model/</code> and <code>labels.json</code> into{" "}
          <code>public/</code>.
        </li>
        <li>
          <strong>Inference</strong> — the React app loads the model once and
          runs it at ~10 Hz on a 30-frame rolling keypoint buffer.
        </li>
      </ol>

      <h2 className="mt-8 text-lg font-semibold">Tunables</h2>
      <ul className="mt-2 list-disc space-y-2 pl-5 text-sm leading-relaxed text-muted-foreground">
        <li>Confidence threshold (default 0.85) in <code>useHandTracking</code>.</li>
        <li>Sliding-window size (30 frames) — must match training shape.</li>
        <li>Pause timeout (1500 ms) in <code>useSentenceBuilder</code>.</li>
      </ul>

      <h2 className="mt-8 text-lg font-semibold">Privacy</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        Everything runs in the browser. No video, no keypoints, and no
        predictions ever leave your device.
      </p>
    </main>
  );
}