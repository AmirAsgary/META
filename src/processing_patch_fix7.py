"""
╔══════════════════════════════════════════════════════════════════════╗
║  PATCH for src/processing.py — Fix #7: direction vector uses CB     ║
║  Paper §2.2.2: edges connect CB pairs, direction vector should be   ║
║  computed from CB positions (not CA) for consistency.               ║
╚══════════════════════════════════════════════════════════════════════╝

In build_cochain_complex(), find these two lines in the edge feature block:

    dir_raw = CA[edst] - CA[esrc]
    dir_n = dir_raw / (np.linalg.norm(dir_raw, axis=-1, keepdims=True) + 1e-8)

Replace with:

    dir_raw = CB[edst] - CB[esrc]
    dir_n = dir_raw / (np.linalg.norm(dir_raw, axis=-1, keepdims=True) + 1e-8)

Rationale: The paper defines edges between CB pairs and says
  d_hat_ij = (r_j - r_i) / ||r_j - r_i||
where r_i, r_j are the positions used for edge construction (CB).
Using CB is consistent: the distance RBF uses CB distances, so the
direction vector should use the same coordinate system.
The local frame projection (N-CA-C) stays unchanged — it defines
the reference frame, not the direction source.
"""
