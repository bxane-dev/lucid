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
  provenance_sha256: string | null;
  message: string | null;
};

type DatasetInfo = {
  id: string;
  title: string;
  description: string;
  version: string;
  license: string;
  doi: string;
  approx_size: string;
  tasks: string[];
  labels: string[];
  recommended_subjects: string[];
  derivatives_only: boolean;
  downloaded: boolean;
  downloaded_files: number;
  bytes_on_disk: number;
  partial_files: number;
  word_prepared: boolean;
  state_prepared: boolean;
};

type DatasetJob = {
  id: string;
  kind: "download" | "prepare";
  dataset_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: {
    phase?: string;
    files_done?: number;
    files_total?: number;
    bytes_downloaded?: number;
    bytes_expected?: number | null;
    message?: string;
  };
  error: string | null;
};

type ApiStatus = {
  data_ready: boolean;
  model_ready: boolean;
  state_model_ready: boolean;
  dataset_id: string;
  prepared_path: string;
  model_path: string;
  state_model_path: string;
  prepared_provenance: string | null;
  model_provenance: string | null;
  state_model_provenance: string | null;
  rule: string;
};

function runtimeEndpoints() {
  const configuredApi = process.env.NEXT_PUBLIC_LUCID_API;
  const configuredWs = process.env.NEXT_PUBLIC_LUCID_WS;

  if (configuredApi && configuredWs) {
    return { api: configuredApi, ws: configuredWs };
  }

  if (
    typeof window !== "undefined" &&
    window.location.hostname === "127.0.0.1" &&
    window.location.port !== "3000"
  ) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return {
      api: window.location.origin,
      ws: `${protocol}//${window.location.host}/ws/live`,
    };
  }

  return {
    api: configuredApi ?? "http://localhost:8000",
    ws: configuredWs ?? "ws://localhost:8000/ws/live",
  };
}

const MAX_POINTS = 180;

function humanBytes(value: number | null | undefined) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function pct(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function shortHash(value: string | null | undefined) {
  if (!value) return "—";
  return `${value.slice(0, 12)}…${value.slice(-8)}`;
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
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetJob, setDatasetJob] = useState<DatasetJob | null>(null);
  const [datasetMessage, setDatasetMessage] = useState<string | null>(null);
  const [connection, setConnection] = useState<
    "connecting" | "live" | "offline"
  >("connecting");
  const [points, setPoints] = useState<number[][]>([]);
  const [visibleChannels, setVisibleChannels] = useState(0);
  const [dataset, setDataset] = useState("nm000113");
  const [recording, setRecording] = useState<string | null>(null);
  const [provenance, setProvenance] = useState<string | null>(null);
  const [groundTruth, setGroundTruth] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const reconnectRef = useRef<number | null>(null);

  useEffect(() => {
    const endpoints = runtimeEndpoints();

    const refreshStatus = () => {
      fetch(`${endpoints.api}/api/status`)
        .then((response) => response.json())
        .then(setStatus)
        .catch(() => setStatus(null));
    };

    const refreshDatasets = () => {
      fetch(`${endpoints.api}/api/datasets`)
        .then((response) => response.json())
        .then((payload) => setDatasets(payload.datasets ?? []))
        .catch(() => setDatasets([]));
    };

    refreshStatus();
    refreshDatasets();

    let closed = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (closed) return;

      setConnection("connecting");
      socket = new WebSocket(endpoints.ws);

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
          setProvenance(message.provenance_sha256 ?? null);
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

  useEffect(() => {
    if (!datasetJob || !["queued", "running"].includes(datasetJob.status)) {
      return;
    }

    const endpoints = runtimeEndpoints();
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(
          `${endpoints.api}/api/dataset-jobs/${datasetJob.id}`,
        );
        const next = (await response.json()) as DatasetJob;
        setDatasetJob(next);

        if (next.status === "completed" || next.status === "failed") {
          window.clearInterval(timer);
          setDatasetMessage(
            next.status === "completed"
              ? `${next.kind === "download" ? "Download" : "Preparation"} completed for ${next.dataset_id}.`
              : next.error ?? "Dataset operation failed.",
          );

          const [datasetsResponse, statusResponse] = await Promise.all([
            fetch(`${endpoints.api}/api/datasets`),
            fetch(`${endpoints.api}/api/status`),
          ]);
          const datasetPayload = await datasetsResponse.json();
          setDatasets(datasetPayload.datasets ?? []);
          setStatus(await statusResponse.json());
        }
      } catch {
        setDatasetMessage("Could not read dataset job status.");
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [datasetJob?.id, datasetJob?.status]);

  async function startDatasetJob(
    datasetId: string,
    kind: "download" | "prepare",
    mode: "recommended" | "all" = "recommended",
  ) {
    const endpoints = runtimeEndpoints();
    setDatasetMessage(null);

    try {
      const response = await fetch(
        `${endpoints.api}/api/datasets/${datasetId}/${kind}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: kind === "download" ? JSON.stringify({ mode }) : undefined,
        },
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Dataset operation could not start.");
      }
      setDatasetJob(payload);
    } catch (error) {
      setDatasetMessage(
        error instanceof Error ? error.message : "Dataset operation failed.",
      );
    }
  }

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


      <section className="panel datasetManager">
        <div className="panelHead">
          <div>
            <span className="label">PUBLIC DATASETS</span>
            <h3>Dataset manager</h3>
          </div>
          <div className="sourceTag">VERIFIED NEMAR SOURCES ONLY</div>
        </div>

        <p className="datasetIntro">
          Download genuine public EEG directly into Lucid. Starter downloads use
          three participants so subject-held-out train / validation / test splits
          remain possible.
        </p>

        {datasetMessage && <div className="datasetMessage">{datasetMessage}</div>}

        {datasetJob && ["queued", "running"].includes(datasetJob.status) && (
          <div className="jobStrip">
            <div>
              <strong>
                {datasetJob.kind === "download" ? "Downloading" : "Preparing"}{" "}
                {datasetJob.dataset_id}
              </strong>
              <span>
                {datasetJob.progress.message ??
                  datasetJob.progress.phase ??
                  datasetJob.status}
              </span>
            </div>
            <div className="jobProgress">
              {datasetJob.progress.files_total
                ? `${datasetJob.progress.files_done ?? 0} / ${datasetJob.progress.files_total} files`
                : datasetJob.status.toUpperCase()}
            </div>
          </div>
        )}

        <div className="datasetGrid">
          {datasets.map((item) => {
            const busy =
              datasetJob?.dataset_id === item.id &&
              ["queued", "running"].includes(datasetJob.status);
            return (
              <article className="datasetCard" key={item.id}>
                <div className="datasetCardTop">
                  <div>
                    <span className="datasetId">{item.id}</span>
                    <h4>{item.title}</h4>
                  </div>
                  <span className={item.word_prepared ? "readyBadge" : "idleBadge"}>
                    {item.word_prepared ? "PREPARED" : item.downloaded ? "DOWNLOADED" : "AVAILABLE"}
                  </span>
                </div>

                <p>{item.description}</p>

                <div className="datasetMeta">
                  <span>{item.license}</span>
                  <span>{item.approx_size}</span>
                  <span>{humanBytes(item.bytes_on_disk)} local</span>
                </div>

                <div className="datasetLabels">
                  {item.labels.map((label) => (
                    <span key={label}>{label}</span>
                  ))}
                </div>

                <div className="datasetActions">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => startDatasetJob(item.id, "download", "recommended")}
                  >
                    {busy && datasetJob?.kind === "download"
                      ? "WORKING…"
                      : "DOWNLOAD 3 SUBJECTS"}
                  </button>
                  <button
                    type="button"
                    className="secondaryButton"
                    disabled={busy}
                    onClick={() => startDatasetJob(item.id, "download", "all")}
                  >
                    DOWNLOAD ALL
                  </button>
                  <button
                    type="button"
                    className="secondaryButton"
                    disabled={busy || !item.downloaded}
                    onClick={() => startDatasetJob(item.id, "prepare")}
                  >
                    PREPARE EEG
                  </button>
                </div>

                <div className="datasetFooter">
                  <span>{item.downloaded_files} files</span>
                  <span>
                    {item.tasks.join(" + ")}
                    {item.state_prepared ? " · state ready" : ""}
                  </span>
                  <a
                    href={`https://nemar.org/dataset/${item.id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    DOI {item.doi} ↗
                  </a>
                </div>
              </article>
            );
          })}
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
        <div className="provenanceLine" title={provenance ?? ""}>
          <span>SOURCE SHA-256</span>
          <strong>{shortHash(provenance)}</strong>
          <span>
            {status?.model_provenance && provenance
              ? status.model_provenance === provenance
                ? "MODEL MATCH"
                : "MODEL MISMATCH"
              : "MODEL NOT VERIFIED"}
          </span>
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
