# Architecture - Person Anonymizer

Technical diagrams for the detection pipeline, web review interaction, and backend selection.

For the high-level overview and pipeline flow, see the [README](../README.md).

---

## Detection Stage Internals

Each frame passes through a multi-step processing pipeline inside `stage_detection.py`. CLAHE enhancement activates only for dark frames. Multi-scale inference runs at 4 scales, with an optional 3x3 sliding window for small targets. After NMS, ByteTrack assigns persistent IDs, and temporal smoothing (EMA) stabilizes bounding boxes across frames. Ghost boxes keep anonimization active for temporarily occluded persons.

```mermaid
%%{init: {'theme': 'default'}}%%
graph TD
  frame_in["Frame Input"] --> clahe_check{"Dark frame?"}
  clahe_check -->|"Yes"| clahe["CLAHE Enhancement"]
  clahe_check -->|"No"| multiscale
  clahe --> multiscale["Multi-scale Inference<br/>1.0x / 1.5x / 2.0x / 2.5x"]
  multiscale --> sliding{"Sliding window<br/>enabled?"}
  sliding -->|"Yes"| sw["Sliding Window 3x3<br/>30% overlap"]
  sliding -->|"No"| nms
  sw --> nms["NMS<br/>Non-Maximum Suppression"]
  multiscale --> nms
  nms --> bytetrack["ByteTrack Update"]
  bytetrack --> ema["Temporal Smoothing<br/>EMA alpha=0.35"]
  ema --> ghost["Ghost Boxes<br/>occluded tracks"]
  ghost --> convert["Box to Polygon<br/>+ Resolve Intensity"]
  convert --> frame_out["Frame Annotations"]

  classDef core fill:#2563eb,stroke:#1d4ed8,color:#fff
  classDef data fill:#d97706,stroke:#b45309,color:#fff
  classDef engine fill:#059669,stroke:#047857,color:#fff

  class frame_in,frame_out data
  class multiscale,sw,nms,bytetrack,ema,ghost,convert core
  class clahe_check,sliding engine
  class clahe engine
```

**Key modules:** `detection.py` (multi-scale + NMS), `tracking.py` (ByteTrack + TemporalSmoother), `preprocessing.py` (CLAHE + motion detection).

---

## Web Review Sequence

When `mode=manual`, the pipeline thread blocks until the user confirms annotations via the web browser. `ReviewState` bridges the pipeline thread and Flask request threads using `threading.Event`. SSE pushes real-time status updates to the browser.

```mermaid
sequenceDiagram
  participant pt as Pipeline Thread
  participant rs as Review State
  participant sse as SSE Manager
  participant flask as Flask App
  participant browser as Browser

  pt->>rs: setup(video, annotations)
  pt->>sse: emit("review_ready")
  sse-->>browser: SSE event review_ready
  pt->>rs: wait_for_completion()
  Note over pt: BLOCKED

  loop Frame navigation
    browser->>flask: GET /api/review/frame/{idx}
    flask->>rs: get_frame_jpeg(idx)
    rs-->>flask: JPEG bytes
    flask-->>browser: Frame image
  end

  browser->>flask: PUT /api/review/annotations/{idx}
  flask->>rs: update_annotations(idx, data)
  rs-->>flask: OK

  browser->>flask: POST /api/review/confirm
  flask->>rs: complete(annotations)
  rs-->>pt: Event.set()
  Note over pt: UNBLOCKED
  pt->>pt: Continue to Rendering
```

**Key modules:** `web/review_state.py` (thread-safe state bridge), `web/sse_manager.py` (event distribution), `web/routes_review.py` (REST endpoints), `stage_review.py` (pipeline integration).

---

## Backend Factory - Detection Mode Selection

`backend_factory.py` implements the factory pattern for detection backends. YOLO is always loaded as the base model. SAM3 components are conditionally loaded based on the `detection_backend` config parameter.

```mermaid
%%{init: {'theme': 'default'}}%%
graph TD
  config["PipelineConfig"] --> load_yolo["Load YOLO Model<br/>(always)"]
  load_yolo --> check_backend{"detection_backend?"}

  check_backend -->|"yolo"| ret_yolo["DetectionBackend<br/>yolo_model only"]
  check_backend -->|"yolo+sam3"| load_refiner["Load Sam3ImageRefiner"]
  check_backend -->|"sam3"| load_detector["Load Sam3VideoDetector"]

  load_refiner --> ret_hybrid["DetectionBackend<br/>yolo + refiner"]
  load_detector --> ret_sam3["DetectionBackend<br/>yolo + video_detector"]

  ret_yolo --> use_yolo["run_detection_loop<br/>(model)"]
  ret_hybrid --> use_hybrid["run_detection_loop<br/>(model, sam3_refiner)"]
  ret_sam3 --> use_sam3["sam3_video_detector<br/>.detect_video()"]

  classDef core fill:#2563eb,stroke:#1d4ed8,color:#fff
  classDef data fill:#d97706,stroke:#b45309,color:#fff
  classDef engine fill:#059669,stroke:#047857,color:#fff

  class config data
  class load_yolo,load_refiner,load_detector core
  class check_backend engine
  class ret_yolo,ret_hybrid,ret_sam3 core
  class use_yolo,use_hybrid,use_sam3 engine
```

| Mode | Detection | Segmentation | GPU Requirement |
|------|-----------|--------------|-----------------|
| `yolo` | YOLO v8 multi-scale | Bounding box to polygon | CUDA recommended, CPU supported |
| `yolo+sam3` | YOLO v8 | SAM3 refines masks per-frame | CUDA required |
| `sam3` | SAM3 end-to-end | SAM3 pixel-precise masks | CUDA required, 8+ GB VRAM |

**Key modules:** `backend_factory.py` (factory + `DetectionBackend` dataclass), `sam3_backend.py` (`Sam3ImageRefiner`, `Sam3VideoDetector`).
