# heuristic-cv-0.1.0

Heuristic dual-scale CV engine is the source of `RoadObservation` until a calibrated segmentation pack is sideloaded.

`model.tflite` is a real LiteRT graph (12-D far/near linear scorer) used as an Interpreter sidecar and capability probe. It is not a road-segmentation network; a failed or unmapped graph never blanks the HUD.

Android ModelManager refuses load if SHA-256 / app compatibility fail, then rolls back.
