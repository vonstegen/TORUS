# INVALID — autograd STE path dead (zero gradients for 500 steps)

This run trained 500 steps and produced an adapter whose eval is
bit-identical to the untrained EXP-A-011 PTQ arm (wikitext ppl
427.7117, arc_easy 0.5396). Root cause: hf_adapter._make_forward_stub
quantized via numpy, silently breaking the autograd graph; every STE
gradient was None (hidden by allow_unused=True), zero-filled, and SGD
stepped zeros. The logged KL "improvement" 1.5756 -> 1.1403 was batch
noise (trajectory bounced 0.38..1.63, no trend).

Secondary bugs found in the same audit: _autograd_grads grad-slot
scramble (flatten order != consumption order) and a calibrate_norm
train/eval regime mismatch (distill_run defaulted True; eval used
--no-calibrate).

Fixed in commit eac2c04 (torch-native straight-through STE,
interleaved grad order, --no-calibrate flag). Superseded by a rerun
under a fresh namespace with the same preregistered manifest.
