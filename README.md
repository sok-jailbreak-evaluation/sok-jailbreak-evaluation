# Jailbreak evaluator experiments

This source tree accompanies Chapters 4 and 5. It contains the evaluation logic, transformations, prompts, model settings and reporting code. Supply datasets, selected attack attempts, model snapshots and API access separately. Run experiments in a private working copy. Generated files can contain the full input and model response.

## Chapter 4

| Question | Source and readout |
|---|---|
| Evaluation criteria and success | `cases/cpu_rules.py` runs the refusal-substring and target-match checks. `cases/component_statistics_v2.py` separates BOOST word coverage from the full gated decision. |
| Task-specific assessment | `goal_swap/` constructs alternative goals for an unchanged response, scores both pairings and reports the results. |
| Instruction and data separation | `batch/jobs.py --job api` applies the configured scoring note to the original response and evaluates both with the same native rubric. |
| Decision robustness | `batch/worker.py` runs single-evaluator suffix optimization with native early stopping. `thresholds/threshold_analysis.py` re-thresholds saved continuous scores. Independent StrongREJECT optimization at another threshold uses the worker with a different stopping target. |
| Predictability and quality | `batch/jobs.py --job ppl_gpt2` and `--job ppl_llama2` append 1, 4 and 8 copies of the last sentence and score full-vocabulary PPL. The report includes the proportion decreasing and the median relative drop. |
| Risk attribution | `batch/jobs.py --job parden` records the rendered input, actual repetition and character comparison. `batch/report.py` counts new flags among valid, initially negative responses for each dataset. |

Paths in the table are relative to `chapter4/`. The `sampling/` scripts implement dataset selection and extension from WildGuardTest, the HarmBench validation responses and the JailbreakBench judge comparison. They expect the upstream files and their source manifests in the layout in `sampling/build.py` and `sampling/freeze_619.py`. The diagnostic studies also expect their declared external case or calibration inputs. Code for the supplementary reverse-direction tests is in `down/`. Those tests require the separate harmful-response input and the fixed ten-row screen.

For the upward-direction batch, set `EVAL_INPUT` to the directory containing `unharmful_up.jsonl` and, for optimization, `mapping.jsonl`. A response row needs `sample_id`, `dataset_id`, `prompt`, `response` and optional `context`. Preserve the original text and input order. The mapping associates `sample_id` with its frozen integer `search_seed_index`. Use an empty `EVAL_OUTPUT` for each distinct configuration. Source labels, sampling strata and incompatible cases retain their own meaning throughout. For the reverse direction supply `harmful_down.jsonl`, `mapping.jsonl` and `screen_10.jsonl` to its worker.

Commands from the repository root:

```sh
python chapter4/batch/jobs.py --job rules --phase full
CUDA_VISIBLE_DEVICES=0 python chapter4/batch/jobs.py --job ppl_gpt2 --phase preflight
CUDA_VISIBLE_DEVICES=0 python chapter4/batch/jobs.py --job ppl_gpt2 --phase full
export INC_SUFFIX_MODEL=strongreject
CUDA_VISIBLE_DEVICES=0 python chapter4/batch/worker.py --model strongreject --phase preflight
CUDA_VISIBLE_DEVICES=0 python chapter4/batch/worker.py --model strongreject --phase full
python chapter4/batch/report.py
```

Use model paths from the configuration relative to the working directory. Each GPU worker sees one complete model. The suffix protocol uses 20 tokens, 128 candidates, top-k 256, 4 rechecks, at most 200 steps, and native early stopping. To repeat the independent StrongREJECT threshold experiment, supply the fixed 30-row cohort with the original mapping, choose a fresh output directory per threshold and set `STRONGREJECT_THRESHOLD` to `0.75` or `0.9` before both worker commands. For offline analysis gzip the generated `all_records.jsonl` and pass the resulting file to the threshold analyzer. `thresholds/api_repeat.py` is the separate fixed-input rubric rescorer. The two score-analysis scripts expose their flags through `--help`.

## Chapter 5

Start from one selected attack attempt per target model, attack method and goal. Reuse the actual submitted prompt and response of that attempt. The evaluation procedure does not require regenerating third-party attacks. `generation/` contains prompt-generation components used for Controlled Release and Response Attack. Their public prompt resources, archived target responses and model weights are external inputs.

From `chapter5/`, run `python prepare_inputs.py /path/to/selected.jsonl`. Each JSONL object supplies `target`, `method`, `goal_id`, `goal`, `attack_prompt`, `attack_prompt_scope` and `response`. Supply null for missing text. For a single user message use `single_submitted_user_turn`; for recorded multi-turn inputs preserve the JSON string with the named submitted user turns and separate context. The builder checks duplicate slots and conflicting goals, hashes exact text, and creates a goal-only input independently of the response records.

1. **Align aspects and mechanisms.** Prepare goal specifications and role plans. Compute goal-conditioned evidence, native decisions, static checks and response properties on the same inputs.
2. **Verify judgments against evidence.** Compose the evidence, route disagreements, check necessary conditions and apply the verification protocol. `final_conditions_v2.py` retains condition evidence for the assessment and review of response aspects.
3. **Aggregate and compare.** Score the submitted prompt and the generation-prefix evidence at their respective units. `finish_five_aspects.py` groups five separate numerators and their planned denominator by model and method. G measures goal preservation (L1.2), H harmful help (L2.2), R relevance (L2.3), A complete assistance (L2.4), and D late onset of harmful help (L3.1).

Example sequence from `chapter5/`, after preparing the inputs and installing models in `models/`:

```sh
python code/evidence_core.py --phase specs --data input/goals.jsonl --specs specifications --out baselines --ledger logs/specs.jsonl
python code/profile_api.py --phase plans
python code/evidence_core.py --phase grade --data input/attempts.jsonl --specs specifications --out baselines --modes evidence --ledger logs/evidence.jsonl
python code/static_checks.py --data input/attempts.jsonl --specs specifications --out results/static
for model in rules llamaguard harmaug harmbench strongreject; do
  CUDA_VISIBLE_DEVICES=0 python code/native.py --model "$model" --data input/attempts.jsonl --out "baselines/$model"
done
python code/profile_api.py --phase profile
python code/profile_api.py --phase compose
python final_conditions_v2.py
python input_and_process.py G
python input_and_process.py D new_final.json
python finish_five_aspects.py
```

Set the API concurrency and request budgets explicitly for the chosen cohort. Use a private `.env` in the working directory containing `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The account must support the configured model names. Use the documented evaluator-specific output scale for comparisons. The comparator implementations include prompted and fine-tuned StrongREJECT, native classifiers, keyword checks, token counts, PPL, PARDEN and HarmScore. The additional comparator folders contain the Protect AI rejection classifier (right truncation to 512 tokens including special tokens), the Llama Guard 3 8B native decoder, and the TriPlay-RL semantic judge of the submitted prompt. The model directory for the Protect AI head is supplied through `REFUSAL_MODEL`, and the TriPlay-RL directory through `SEMANTIC_MODEL`. These comparator scripts have separate resource requirements from the main evidence pipeline.

## Environment and sources

Use Linux and Python 3.10 or later. Install a CUDA-compatible PyTorch and Transformers environment for the local models. See `requirements.txt` for the remaining imports. The semantic comparator also uses vLLM and json-repair. Conditions and reference prompts are versioned in the source tree. Model snapshots, including gated Llama or Gemma weights, must be obtained under their respective terms.

The StrongREJECT native consistency check imports `cached_models` and `strongreject_finetuned` from `strong_reject.evaluate`. For runs using that check, install the [upstream StrongREJECT implementation](https://github.com/dsbowen/strong_reject) in the same Python environment.

The source includes extracts of the public evaluator implementations identified in the code for StrongREJECT, HarmBench, HarmAug, GPTFuzzer, Llama Guard, llm-attacks, BOOST, PARDEN, the n-gram constraint, TriPlay-RL, and the published attack generators. The upstream Controlled Release license is included under `chapter5/generation/LICENSE`. Public reference credits identify the originating methods.

To reproduce the native HarmScore aggregation, score the same attempt list with `comparators/harmscore/run.py branch` for each released head, then use `run.py combine` on both outputs. It takes the geometric mean for each valid response; the group statistic is the arithmetic mean of valid response scores. The two-stage classifier and the five-aspect assessment have separate outputs.

## Reported evaluation data

`aspect_results/records.jsonl` provides the 8,320 selected-response records for the 5 reported aspects. `summary.json` and `summary.csv` retain the per-method counts for both targets. `summarize.py` reproduces these files without model calls. The published readouts are G (L1.2), H (L2.2), R (L2.3), A (L2.4) and D (L3.1). Necessary-condition records are supporting evidence within the evaluation process.
