"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Alternative = { label: string; confidence: number };

type Prediction = {
  status: "ok" | "model_unavailable" | "data_unavailable" | "error";
  state: string | null;
  state_confidence: number | null;
  prediction: string | null;
  prediction_confidence: number | null;
  alternatives: Alternative[];
  source_dataset: string | null;
  source_recording: string | null;
  model_version: string | null;
  message: string | null;
};

type ApiStatus = {
  data_ready: boolean;
  model_ready: boolean;
  state_model_ready: boolean;
  dataset_id: string;
  prepared_path: string;
  model_path: string;
  state_model_path: string;
  rule: string;
};

const API = process.env.NEXT_PUBLIC_LUCID_API ?? "http://localhost:8000";
const WS = process.env.NEXT_PUBLIC_LUCID_WS ?? "ws://localhost:8000/ws/live";
const MAX_POINTS = 180;

function pct(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function titleCase(value: string | null | undefined) {
  if (!value) return "—";
  return value.replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function Signal({
  points,
  channels,
}: {
  points: number[][];
  channels: number;
}) {
  const width = 1000;
  const height = 280;

  const paths = useMemo(() => {
    if (!points.length || !channels) return [];
    const n = points.length;

    return Array.from({ length: channels }, (_, channel) => {
      const values = points.map((point) => point[channel] ?? 0);
      const maxAbs = Math.max(...values.map(Math.abs), 1e-4);
      const lane = height / channels;
      const mid = lane * (channel + 0.5);

      return values
        .map((value, index) => {
          const x = n === 1 ? 0 : (index / (n - 1)) * width;
          const y = mid - (value / maxAbs) * lane * 0.32;
          return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(" ");
    });
  }, [points, channels]);

  return (
    <div className="signalFrame">
      <div className="signalGrid" />
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {paths.map((path, index) => (
          <path
            key={index}
            d={path}
            className="trace"
            opacity={0.45 + index * 0.06}
          />
        ))}
      </svg>
      {!points.length && (
        <div className="emptyOverlay">Waiting for recorded EEG samples…</div>
      )}
    </div>
  );
}

export default function Home() {
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [connection, setConnection] = useState<
    "connecting" | "live" | "offline"
  >("connecting");
  const [points, setPoints] = useState<number[][]>([]);
  const [visibleChannels, setVisibleChannels] = useState(0);
  const [dataset, setDataset] = useState("nm000113");
  const [recording, setRecording] = useState<string | null>(null);
  const [groundTruth, setGroundTruth] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const reconnectRef = useRef<number | null>(null);

  useEffect(() => {
    fetch(`${API}/api/status`)
      .then((response) => response.json())
      .then(setStatus)
      .catch(() => setStatus(null));

    let closed = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (closed) return;

      setConnection("connecting");
      socket = new WebSocket(WS);

      socket.onopen = () => {
        setConnection("live");
        setStreamError(null);
      };

      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === "error") {
          setStreamError(message.message);
          return;
        }

        if (message.type === "trial_start") {
          setPoints([]);
          setPrediction(null);
          setDataset(message.dataset);
          setRecording(message.recording);
          setGroundTruth(message.ground_truth);
          setVisibleChannels(Math.min(8, message.channels));
          return;
        }

        if (message.type === "sample") {
          setPoints((previous) => {
            const next = [...previous, message.values];
            return next.length > MAX_POINTS
              ? next.slice(-MAX_POINTS)
              : next;
          });
          return;
        }

        if (message.type === "prediction") {
          setPrediction(message.prediction);
          setGroundTruth(message.ground_truth);
        }
      };

      socket.onclose = () => {
        setConnection("offline");
        if (!closed) {
          reconnectRef.current = window.setTimeout(connect, 2500);
        }
      };

      socket.onerror = () => setConnection("offline");
    };

    connect();

    return () => {
      closed = true;
      if (reconnectRef.current) {
        window.clearTimeout(reconnectRef.current);
      }
      socket?.close();
    };
  }, []);

  const wordModelReady = status?.model_ready ?? false;
  const stateModelReady = status?.state_model_ready ?? false;
  const dataReady = status?.data_ready ?? false;

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <div className="mark">L</div>
          <div>
            <h1>LUCID</h1>
            <p>PUBLIC EEG ANALYZER</p>
          </div>
        </div>
        <div className="headerMeta">
          <span className={`dot ${connection}`} />
          <span>{connection.toUpperCase()}</span>
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">ZERO-COST NEURAL DECODING RESEARCH</p>
          <h2>
            Recorded brain signals.
            <br />
            No fabricated output.
          </h2>
        </div>

        <div className="statusRail">
          <div className="statusItem">
            <span>PUBLIC DATA</span>
            <strong>{dataReady ? "READY" : "MISSING"}</strong>
          </div>
          <div className="statusItem">
            <span>WORD MODEL</span>
            <strong>{wordModelReady ? "TRAINED" : "NOT TRAINED"}</strong>
          </div>
          <div className="statusItem">
            <span>STATE MODEL</span>
            <strong>{stateModelReady ? "TRAINED" : "NOT TRAINED"}</strong>
          </div>
          <div className="statusItem">
            <span>DATASET</span>
            <strong>{dataset}</strong>
          </div>
        </div>
      </section>

      {streamError && (
        <section className="notice">
          <strong>DATA REQUIRED</strong>
          <span>{streamError}</span>
        </section>
      )}

      <section className="panel signalPanel">
        <div className="panelHead">
          <div>
            <span className="label">SIGNAL</span>
            <h3>Live replay</h3>
          </div>
          <div className="sourceTag">
            REAL RECORDED EEG · {visibleChannels || "—"} CH SHOWN
          </div>
        </div>

        <Signal points={points} channels={visibleChannels} />

        <div className="recordingPath" title={recording ?? ""}>
          {recording
            ? recording.split(/[\\/]/).slice(-3).join(" / ")
            : "No recording loaded"}
        </div>
      </section>

      <section className="predictionGrid">
        <article className="panel primaryPrediction">
          <span className="label">NEURAL PREDICTION</span>
          <div className="bigPrediction">
            {prediction?.status === "ok"
              ? titleCase(prediction.prediction)
              : wordModelReady
                ? "WAITING"
                : "NO MODEL"}
          </div>
          <div className="confidence">
            <span>Confidence</span>
            <strong>
              {prediction?.status === "ok"
                ? pct(prediction.prediction_confidence)
                : "—"}
            </strong>
          </div>
          {prediction?.status === "model_unavailable" && (
            <p className="muted">{prediction.message}</p>
          )}
        </article>

        <article className="panel">
          <span className="label">RECORDED TASK ANNOTATION</span>
          <div className="groundTruth">{titleCase(groundTruth)}</div>
          <p className="muted">
            Ground truth from the public dataset. This is not presented as a
            model prediction.
          </p>
        </article>

        <article className="panel">
          <span className="label">DETECTED STATE</span>
          <div className="groundTruth">
            {prediction?.state
              ? titleCase(prediction.state)
              : stateModelReady
                ? "WAITING"
                : "NO MODEL"}
          </div>
          <div className="confidence compact">
            <span>Model confidence</span>
            <strong>{pct(prediction?.state_confidence)}</strong>
          </div>
          <p className="muted">
            State output appears only when a dedicated state classifier produces
            it.
          </p>
        </article>
      </section>

      <section className="panel alternatives">
        <div className="panelHead">
          <div>
            <span className="label">CLASS DISTRIBUTION</span>
            <h3>Alternative predictions</h3>
          </div>
          <span className="sourceTag">
            {prediction?.model_version ?? "MODEL UNAVAILABLE"}
          </span>
        </div>

        <div className="bars">
          {prediction?.status === "ok" ? (
            [
              {
                label: prediction.prediction ?? "",
                confidence: prediction.prediction_confidence ?? 0,
              },
              ...prediction.alternatives,
            ].map((item) => (
              <div className="barRow" key={item.label}>
                <span>{titleCase(item.label)}</span>
                <div className="barTrack">
                  <div
                    className="barFill"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, item.confidence * 100),
                      )}%`,
                    }}
                  />
                </div>
                <strong>{pct(item.confidence)}</strong>
              </div>
            ))
          ) : (
            <div className="emptyBars">
              Probabilities appear only after a locally trained model produces
              them.
            </div>
          )}
        </div>
      </section>

      <footer>
        <span>LUCID v0.2</span>
        <span>PUBLIC DATA · LOCAL MODELS · SQLITE</span>
        <a
          href="https://nemar.org/dataset/nm000113"
          target="_blank"
          rel="noreferrer"
        >
          NEMAR nm000113 ↗
        </a>
      </footer>
    </main>
  );
}
