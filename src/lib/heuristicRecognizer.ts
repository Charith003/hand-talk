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
] as const;

function dist(a: Landmark, b: Landmark) {
  return Math.hypot((a.x ?? 0) - (b.x ?? 0), (a.y ?? 0) - (b.y ?? 0));
}

// Returns [thumb, index, middle, ring, pinky] booleans for "extended".
function fingerStates(lm: Landmark[]): boolean[] {
  const wrist = lm[0];
  // For 4 fingers: tip is more "extended" if tip is farther from wrist than PIP joint.
  const fingers = [
    [8, 6], // index
    [12, 10], // middle
    [16, 14], // ring
    [20, 18], // pinky
  ].map(([tip, pip]) => dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.08);

  // Thumb: compare tip (4) horizontal distance from index MCP (5) vs IP (3).
  const thumbTip = lm[4];
  const thumbIp = lm[3];
  const indexMcp = lm[5];
  const thumbExtended =
    Math.abs((thumbTip.x ?? 0) - (indexMcp.x ?? 0)) >
    Math.abs((thumbIp.x ?? 0) - (indexMcp.x ?? 0)) * 1.1;

  return [thumbExtended, ...fingers];
}

function classifyHand(lm: Landmark[]): { word: string; confidence: number } | null {
  if (!lm || lm.length < 21) return null;
  const [t, i, m, r, p] = fingerStates(lm);
  const count = [t, i, m, r, p].filter(Boolean).length;

  // OK sign: thumb tip near index tip, middle/ring/pinky extended
  const thumbIndexClose = dist(lm[4], lm[8]) < dist(lm[0], lm[5]) * 0.45;
  if (thumbIndexClose && m && r && p) return { word: "ok", confidence: 0.9 };

  // ILY: thumb + index + pinky, middle & ring folded
  if (t && i && !m && !r && p) return { word: "i love you", confidence: 0.95 };

  // Specific combos first
  if (!t && i && m && !r && !p) return { word: "peace", confidence: 0.95 };
  if (!t && i && !m && !r && !p) return { word: "point", confidence: 0.9 };
  if (t && !i && !m && !r && p) return { word: "call me", confidence: 0.9 };
  if (!t && i && !m && !r && p) return { word: "rock", confidence: 0.9 };

  if (count === 5) return { word: "hello", confidence: 0.95 };
  if (count === 4 && !t) return { word: "stop", confidence: 0.9 };
  if (count === 0) return { word: "fist", confidence: 0.85 };

  // Thumb only (up) → yes; thumb down detection via thumb tip vs wrist Y
  if (t && !i && !m && !r && !p) {
    const thumbUp = (lm[4].y ?? 0) < (lm[0].y ?? 0);
    return thumbUp
      ? { word: "yes", confidence: 0.9 }
      : { word: "no", confidence: 0.85 };
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