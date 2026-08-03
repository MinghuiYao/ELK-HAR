# Reproducibility status

This repository verifies model forward/backward behavior and numerical
equivalence before and after structural reparameterization. The original
release did not include dataset preprocessing, subject split files, training
configurations, checkpoints, or generated result tables.

A complete paper artifact should record dataset source and checksum, sampling
rate, windowing and normalization, exact subject identifiers, all seeds,
optimizer and schedule, checkpoint selection, accuracy and macro-F1 per run,
latency before and after reparameterization, hardware, environment, and git
commit. Missing protocols must not be inferred from another repository.
