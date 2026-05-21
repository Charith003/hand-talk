import { useCallback, useEffect, useRef, useState } from "react";

const PAUSE_TIMEOUT_MS = 1500;

export interface SentenceEntry {
  text: string;
  time: string;
}

export function useSentenceBuilder(opts: {
  pauseMs?: number;
  ttsEnabled?: boolean;
}) {
  const { pauseMs = PAUSE_TIMEOUT_MS, ttsEnabled = true } = opts;
  const [words, setWords] = useState<string[]>([]);
  const [history, setHistory] = useState<SentenceEntry[]>([]);
  const lastWordRef = useRef<string>("");
  const lastAddRef = useRef<number>(0);
  const pauseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const speak = useCallback((text: string) => {
    if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.95;
    utt.pitch = 1;
    window.speechSynthesis.speak(utt);
  }, []);

  const finalizeSentence = useCallback(() => {
    setWords((prev) => {
      if (prev.length === 0) return prev;
      const text = prev.join(" ");
      if (ttsEnabled) speak(text);
      setHistory((h) => [
        { text, time: new Date().toLocaleTimeString() },
        ...h,
      ].slice(0, 20));
      lastWordRef.current = "";
      return [];
    });
  }, [speak, ttsEnabled]);

  const addPrediction = useCallback(
    (word: string, _confidence: number) => {
      if (!word) return;
      const now = performance.now();
      // Debounce: skip same word repeated within 700ms
      if (word === lastWordRef.current && now - lastAddRef.current < 700) {
        // still keep timer alive
        if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
        pauseTimerRef.current = setTimeout(finalizeSentence, pauseMs);
        return;
      }
      lastWordRef.current = word;
      lastAddRef.current = now;
      setWords((prev) => [...prev, word].slice(-30));
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = setTimeout(finalizeSentence, pauseMs);
    },
    [finalizeSentence, pauseMs],
  );

  const clear = useCallback(() => {
    setWords([]);
    lastWordRef.current = "";
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
  }, []);

  const speakNow = useCallback(() => {
    const text = words.join(" ");
    if (text) {
      speak(text);
      setHistory((h) => [
        { text, time: new Date().toLocaleTimeString() },
        ...h,
      ].slice(0, 20));
    }
  }, [words, speak]);

  useEffect(() => () => {
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
  }, []);

  return { words, history, addPrediction, clear, speakNow, speak };
}
