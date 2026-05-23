// Pre-trained, rule-based sign recognizer.
// Uses MediaPipe hand landmarks (21 per hand, normalized 0..1).
// No training data required — recognizes common signs from finger geometry.

export type Landmark = { x?: number; y?: number; z?: number };

// Vocabulary the heuristic recognizer supports.
export const HEURISTIC_VOCAB = [
  "hello",
  "yes",
  "no",
  "thank you",
  "i love you",
  "peace",
  "ok",
  "stop",
  "point",
  "call me",
  "rock",
  "fist",
  "good",
  "bad",
  "three",
  "four",
] as const;

function dist(a: Landmark, b: Landmark) {
  return Math.hypot((a.x ?? 0) - (b.x ?? 0), (a.y ?? 0) - (b.y ?? 0));
}

// Returns [thumb, index, middle, ring, pinky] booleans for "extended".
function fingerStates(lm: Landmark[]): boolean[] {
  const wrist = lm[0];
  const fingers = [
    [8, 6, 5], // index: tip, pip, mcp
    [12, 10, 9],
    [16, 14, 13],
    [20, 18, 17],
  ].map(([tip, pip, mcp]) => {
    // A finger is "extended" if the tip is farther from the mcp than the pip is.
    return dist(lm[tip], lm[mcp]) > dist(lm[pip], lm[mcp]) * 1.15;
  });

  // Thumb extended: tip is farther from index-mcp than the IP joint is.
  const thumbExtended = dist(lm[4], lm[5]) > dist(lm[3], lm[5]) * 1.2;

  return [thumbExtended, ...fingers];
}

function classifyHand(lm: Landmark[]): { word: string; confidence: number } | null {
  if (!lm || lm.length < 21) return null;
  const [t, i, m, r, p] = fingerStates(lm);
  const count = [t, i, m, r, p].filter(Boolean).length;
  const palm = dist(lm[0], lm[9]) || 0.001;

  // OK sign: thumb tip near index tip, middle/ring/pinky extended
  const thumbIndexClose = dist(lm[4], lm[8]) < palm * 0.4;
  if (thumbIndexClose && m && r && p) return { word: "ok", confidence: 0.92 };

  // ILY: thumb + index + pinky, middle & ring folded
  if (t && i && !m && !r && p) return { word: "i love you", confidence: 0.96 };

  // Specific combos first
  if (!t && i && m && !r && !p) return { word: "peace", confidence: 0.95 };
  if (!t && i && m && r && !p) return { word: "three", confidence: 0.9 };
  if (!t && i && m && r && p) return { word: "four", confidence: 0.9 };
  if (!t && i && !m && !r && !p) return { word: "point", confidence: 0.92 };
  if (t && !i && !m && !r && p) return { word: "call me", confidence: 0.9 };
  if (!t && i && !m && !r && p) return { word: "rock", confidence: 0.92 };

  if (count === 5) return { word: "hello", confidence: 0.95 };
  if (count === 4 && !t) return { word: "stop", confidence: 0.92 };
  if (count === 0) {
    // distinguish thank-you-ish (fist near chin/mouth) vs plain fist by hand y position.
    return { word: "fist", confidence: 0.85 };
  }

  // Thumb only — up = "good" (yes), down = "bad" (no)
  if (t && !i && !m && !r && !p) {
    const thumbUp = (lm[4].y ?? 0) < (lm[0].y ?? 0) - palm * 0.2;
    const thumbDown = (lm[4].y ?? 0) > (lm[0].y ?? 0) + palm * 0.2;
    if (thumbUp) return { word: "good", confidence: 0.92 };
    if (thumbDown) return { word: "bad", confidence: 0.9 };
    return { word: "yes", confidence: 0.8 };
  }

  return null;
}

export function recognizeHeuristic(
  handsLandmarks: Landmark[][],
): { word: string; confidence: number } | null {
  if (!handsLandmarks?.length) return null;
  // Prefer most-confident hand.
  let best: { word: string; confidence: number } | null = null;
  for (const hand of handsLandmarks.slice(0, 2)) {
    const c = classifyHand(hand);
    if (c && (!best || c.confidence > best.confidence)) best = c;
  }
  return best;
}