# Road Perception AR (Android)

V0.1 Android client for Road Perception AR. Package `com.ebike.rpar`. Camera2 preview + YUV analysis, heuristic dual-scale perception, tracking, geometry, optional TTS alerts, OpenGL overlay, and local session recording.

## Requirements

- JDK 17
- Android SDK 35 (`compileSdk` / `targetSdk` 35, `minSdk` 26)
- `local.properties` already points at `C:\Users\root\AppData\Local\Android\Sdk`

## Build debug APK

From this `android/` directory:

```bat
set JAVA_HOME=C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot
.\gradlew.bat assembleDebug
```

On PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
.\gradlew.bat assembleDebug
```

Output APK:

`app/build/outputs/apk/debug/app-debug.apk`

Install on a device or emulator:

```bat
.\gradlew.bat installDebug
```

## Signing (chapter 12)

Debug APK uses the Android default debug keystore (`~/.android/debug.keystore`). Do not commit keystores or passwords.

Field / release APK:

1. Create an upload keystore **outside this repo**.
2. Put passwords in `android/keystore.properties` (gitignored) or environment variables — never in committed Gradle sources.
3. Wire `signingConfigs.release` in `app/build.gradle.kts` when the team has a real key.
4. Record the SHA-256 fingerprint in `docs/RELEASE_NOTES.md` after the first signed build.

Until that key exists, `assembleDebug` is the installable deliverable for development.

## Emulator / safe mode

The app starts without opening a camera. Replay and SAFE_MODE feed a generated road test pattern so perception, AR, and HUD work on an emulator.

## Run modes

- `CAPTURE_ONLY` — camera/test pattern + recording, no engine
- `REALTIME_PERCEPTION` — live perception + AR
- `REALTIME_PERCEPTION_FULL_LOG` — perception plus full jsonl session logs
- `SAFE_MODE` — engine disabled, test pattern allowed
- `REPLAY` — test pattern, no camera

## Bump LiteRT sidecar

`rpar export-bump-tflite` writes `models/bump-world-0.1.0/model.onnx` on Windows (Ultralytics LiteRT export is Linux/macOS only) or `model.tflite` when that export works. Default letterbox size is **640** to match desktop YOLO-World. If the graph exists at build time, Gradle copies it into APK assets. At runtime `ModelManager` keeps the heuristic engine for road/info-layer and replaces pothole / speed-bump / manhole observations when ONNX Runtime (NNAPI, then CPU) or LiteRT loads the graph. Camera2 analysis now copies U/V so letterbox RGB is not luma-only.

V0.1 only ships `arm64-v8a` (PKC110). Emulator x86 falls back to heuristic if ONNX native libs are absent.
