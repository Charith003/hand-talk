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
  const [ttsReady, setTtsReady] = useState(false);
  const lastWordRef = useRef<string>("");
  const pauseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const unlockTts = useCallback(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;
    const synth = window.speechSynthesis;
    try {
      synth.resume();
      synth.getVoices();
      setTtsReady(true);
      return true;
    } catch {
      return false;
    }
  }, []);

  const speak = useCallback((text: string) => {
    if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (!unlockTts()) return;

    const synth = window.speechSynthesis;
    let spoken = false;
    const run = () => {
      if (spoken) return;
      spoken = true;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      const voices = synth.getVoices();
      const englishVoice = voices.find((voice) => voice.lang?.toLowerCase().startsWith("en"));
      if (englishVoice) utterance.voice = englishVoice;
      utterance.rate = 0.95;
      utterance.pitch = 1;
      utterance.volume = 1;
      synth.speak(utterance);
    };

    if (synth.getVoices().length === 0) {
      synth.onvoiceschanged = run;
      window.setTimeout(run, 200);
    } else {
      run();
    }
  }, [unlockTts]);

  const finalizeSentence = useCallback(() => {
    setWords((prev) => {
      if (prev.length === 0) return prev;
      const text = prev.join(" ");
      if (ttsEnabled && ttsReady) speak(text);
      setHistory((h) => [
        { text, time: new Date().toLocaleTimeString() },
        ...h,
      ].slice(0, 20));
      lastWordRef.current = "";
      return [];
    });
  }, [speak, ttsEnabled, ttsReady]);

  const addPrediction = useCallback(
    (word: string, _confidence: number) => {
      if (!word) return;
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = setTimeout(finalizeSentence, pauseMs);

      if (word === lastWordRef.current) return;
      lastWordRef.current = word;
      setWords((prev) => [...prev, word].slice(-30));
    },
    [finalizeSentence, pauseMs],
  );

  const clear = useCallback(() => {
    setWords([]);
    lastWordRef.current = "";
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
  }, []);

  const speakNow = useCallback(() => {
    unlockTts();
    const text = words.join(" ");
    if (text) {
      speak(text);
      setHistory((h) => [
        { text, time: new Date().toLocaleTimeString() },
        ...h,
      ].slice(0, 20));
    }
  }, [words, speak, unlockTts]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.getVoices();
  }, []);

  useEffect(() => () => {
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
  }, []);

  return { words, history, addPrediction, clear, speakNow, speak, unlockTts, ttsReady };
}
