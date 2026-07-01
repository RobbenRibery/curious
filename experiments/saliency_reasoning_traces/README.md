# AD-CISPO Saliency Reasoning Trace Comparison

Generated on Modal H100 with:

```bash
scripts/run_saliency_comparison_modal.sh \
  --model Qwen/Qwen3-1.7B \
  --trace-source curated \
  --num-traces 6 \
  --max-new-tokens 384 \
  --temperature 0.6 \
  --seed 17 \
  --top-layers 4 \
  --attention-block-size 128 \
  --local-output-dir experiments/saliency_reasoning_traces
```

Artifacts:

- `attention_heatmap.html`: future-attention in-degree saliency over the six traces.
- `causal_tangent_heatmap.html`: causal-tangent saliency over the same six traces.
- `head_to_head_heatmap.html`: both methods stacked trace-by-trace over identical token rows.
- `summary.json`: compact run metadata and top-token summaries.
- `saliency_data.json`: full token rows, saliency scores, and derived multipliers.

The default run intentionally keeps sink guarding disabled so separator and punctuation tokens remain visible. This makes the old attention-centrality failure mode easier to inspect directly.

Observed pattern in this focused sample: future-attention in-degree often emphasizes early setup tokens, punctuation, whitespace-like tokens, and paragraph separators. Causal-tangent saliency is much lower scale and tends to move mass toward objective-connected terms in the algebra or self-check regions, though punctuation can still appear when it is locally tied to the model's likelihood objective.
