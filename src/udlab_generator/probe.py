"""Out-of-fold ToM shortcut probes (a)-(e) grouped by `scenario_id` and class-balance checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
VERB_TOKENS: frozenset[str] = frozenset(
    {
        "moved",
        "walked",
        "exited",
        "observed",
        "looked",
        "said",
        "sent",
        "plans",
        "depends",
        "adjusted",
        "starts",
    }
)


@dataclass(frozen=True)
class ProbeItemResult:
    """Result for a single shortcut probe."""

    probe_id: str
    description: str
    num_samples: int
    base_rate: float
    oof_accuracy: float
    margin_over_base: float
    max_allowed_margin: float
    passed: bool


@dataclass(frozen=True)
class ClassBalanceReport:
    """Class balance proportions across the dataset."""

    u_share_among_open_stems: float
    who_u_share: float
    why_u_share: float
    b1_true_belief_share: float
    b2_true_belief_share: float
    x_unnotified_share: float
    p_yes_share: float
    k_yes_share: float
    balanced: bool


@dataclass(frozen=True)
class ShortcutProbeReport:
    """Complete shortcut probe and balance report."""

    num_scenarios: int
    num_renderings: int
    probes: tuple[ProbeItemResult, ...]
    balance: ClassBalanceReport
    passed: bool


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _ngrams(tokens: Sequence[str], max_n: int = 2) -> list[str]:
    feats: list[str] = list(tokens)
    if max_n >= 2:
        feats.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return feats


def _assign_scenario_folds(scenario_ids: Sequence[str], n_folds: int = 5) -> dict[str, int]:
    """Deterministically assign each unique `scenario_id` to one of `n_folds` folds."""
    unique_ids = sorted(set(scenario_ids))
    return {sid: idx % n_folds for idx, sid in enumerate(unique_ids)}


def _oof_multinomial_nb(
    samples: Sequence[tuple[str, list[str], str]],
    n_folds: int = 5,
    alpha: float = 1.0,
) -> float:
    """Run out-of-fold multinomial Naive Bayes grouped by `scenario_id` and return OOF accuracy."""
    if not samples:
        return 0.0
    fold_map = _assign_scenario_folds([s[0] for s in samples], n_folds=n_folds)
    correct = 0
    total = 0

    for fold in range(n_folds):
        train = [s for s in samples if fold_map[s[0]] != fold]
        test = [s for s in samples if fold_map[s[0]] == fold]
        if not test or not train:
            continue

        class_counts: Counter[str] = Counter()
        feat_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_feats_by_class: Counter[str] = Counter()
        vocab: set[str] = set()

        for _, feats, label in train:
            class_counts[label] += 1
            for f in feats:
                feat_counts[label][f] += 1
                total_feats_by_class[label] += 1
                vocab.add(f)

        classes = sorted(class_counts.keys())
        n_train = len(train)
        v_size = max(len(vocab), 1)

        for _, feats, true_label in test:
            best_label = classes[0]
            best_score = -math.inf
            for c in classes:
                log_prob = math.log((class_counts[c] + alpha) / (n_train + alpha * len(classes)))
                denom = total_feats_by_class[c] + alpha * v_size
                c_counter = feat_counts[c]
                for f in feats:
                    log_prob += math.log((c_counter.get(f, 0) + alpha) / denom)
                if log_prob > best_score:
                    best_score = log_prob
                    best_label = c
            if best_label == true_label:
                correct += 1
            total += 1

    return correct / max(total, 1)


def _oof_numeric_classifier(
    samples: Sequence[tuple[str, tuple[float, ...], int]],
    n_folds: int = 5,
) -> float:
    """Run out-of-fold L2-regularized binary logistic + Gaussian centroid classifier grouped by `scenario_id`."""
    if not samples:
        return 0.0
    fold_map = _assign_scenario_folds([s[0] for s in samples], n_folds=n_folds)
    dim = len(samples[0][1])

    correct_gd = 0
    correct_centroid = 0
    total = 0

    for fold in range(n_folds):
        train = [s for s in samples if fold_map[s[0]] != fold]
        test = [s for s in samples if fold_map[s[0]] == fold]
        if not test or not train:
            continue

        # Standardize features on the training fold.
        means = [0.0] * dim
        stds = [1.0] * dim
        for d in range(dim):
            vals = [s[1][d] for s in train]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            means[d] = m
            stds[d] = math.sqrt(var) if var > 1e-12 else 1.0

        def _norm(x: tuple[float, ...]) -> list[float]:
            return [(x[d] - means[d]) / stds[d] for d in range(dim)]

        train_x = [_norm(s[1]) for s in train]
        train_y = [s[2] for s in train]

        # 1. Fit L2 logistic regression with batch gradient descent on training fold.
        weights = [0.0] * dim
        bias = 0.0
        lr = 0.15
        l2 = 0.05
        n_tr = len(train_x)
        for _ in range(60):
            grad_w = [0.0] * dim
            grad_b = 0.0
            for x_vec, y_val in zip(train_x, train_y):
                z = max(-20.0, min(20.0, sum(w * xv for w, xv in zip(weights, x_vec)) + bias))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - y_val
                grad_b += err
                for d in range(dim):
                    grad_w[d] += err * x_vec[d]
            bias -= lr * (grad_b / n_tr)
            for d in range(dim):
                weights[d] -= lr * ((grad_w[d] / n_tr) + l2 * weights[d])

        # 2. Fit class centroids on training fold.
        pos_x = [x for x, y in zip(train_x, train_y) if y == 1]
        neg_x = [x for x, y in zip(train_x, train_y) if y == 0]
        c_pos = [sum(x[d] for x in pos_x) / max(len(pos_x), 1) for d in range(dim)]
        c_neg = [sum(x[d] for x in neg_x) / max(len(neg_x), 1) for d in range(dim)]

        for _, raw_x, true_y in test:
            nx = _norm(raw_x)
            z = sum(w * xv for w, xv in zip(weights, nx)) + bias
            pred_gd = 1 if z >= 0.0 else 0
            if pred_gd == true_y:
                correct_gd += 1

            dist_pos = sum((nx[d] - c_pos[d]) ** 2 for d in range(dim))
            dist_neg = sum((nx[d] - c_neg[d]) ** 2 for d in range(dim))
            pred_c = 1 if dist_pos <= dist_neg else 0
            if pred_c == true_y:
                correct_centroid += 1

            total += 1

    return max(correct_gd, correct_centroid) / max(total, 1)


def _extract_candidate_locations_in_prose_order(prose: str, candidate_locs: Sequence[str]) -> list[str]:
    """Return candidate locations sorted by their first character mention offset in `prose` (and last offset)."""
    prose_low = prose.lower()
    found: list[tuple[int, str]] = []
    for loc in dict.fromkeys(candidate_locs):
        idx = prose_low.find(loc.lower())
        if idx >= 0:
            found.append((idx, loc))
    found.sort(key=lambda t: t[0])
    return [loc for _, loc in found]


def _extract_last_mentioned_location(prose: str, candidate_locs: Sequence[str]) -> str | None:
    prose_low = prose.lower()
    found: list[tuple[int, str]] = []
    for loc in dict.fromkeys(candidate_locs):
        idx = prose_low.rfind(loc.lower())
        if idx >= 0:
            found.append((idx, loc))
    if not found:
        return None
    found.sort(key=lambda t: t[0])
    return found[-1][1]


def _sentence_split(prose: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]


def _agent_event_distance_features(
    prose: str,
    agent_name: str,
    object_name: str,
) -> tuple[tuple[float, ...], list[str]]:
    """Compute name-to-event sentence/token distance and local n-grams around `agent_name`."""
    sentences = _sentence_split(prose)
    agent_low = agent_name.lower()
    obj_low = object_name.lower()

    agent_sent_indices = [i for i, s in enumerate(sentences) if agent_low in s.lower()]
    obj_sent_indices = [
        i
        for i, s in enumerate(sentences)
        if obj_low in s.lower() and ("moved" in s.lower() or "said:" in s.lower() or "stored" in s.lower())
    ]
    if not obj_sent_indices:
        obj_sent_indices = [i for i, s in enumerate(sentences) if obj_low in s.lower()]

    if agent_sent_indices and obj_sent_indices:
        min_sent_dist = float(
            min(abs(a_i - o_i) for a_i in agent_sent_indices for o_i in obj_sent_indices)
        )
        co_occur = float(sum(1 for i in agent_sent_indices if i in obj_sent_indices))
    else:
        min_sent_dist = float(len(sentences))
        co_occur = 0.0

    tokens = _tokenize(prose)
    ag_tok_idx = [i for i, t in enumerate(tokens) if t == agent_low]
    obj_phrase_first = _tokenize(object_name)[0] if _tokenize(object_name) else obj_low
    obj_tok_idx = [i for i, t in enumerate(tokens) if t == obj_phrase_first]
    if ag_tok_idx and obj_tok_idx:
        min_tok_dist = float(min(abs(a_i - o_i) for a_i in ag_tok_idx for o_i in obj_tok_idx))
    else:
        min_tok_dist = float(len(tokens))

    # Extract surface n-grams in sentences mentioning `agent_name` (excluding the specific random room name
    # or including all local context tokens).
    local_tokens: list[str] = []
    for idx in agent_sent_indices:
        for tok in _tokenize(sentences[idx]):
            if tok == agent_low:
                local_tokens.append("<AGENT>")
            else:
                local_tokens.append(tok)

    num_feats = (min_sent_dist, min_tok_dist, co_occur, float(len(agent_sent_indices)))
    return num_feats, _ngrams(local_tokens, max_n=2)


def run_shortcut_probe(
    sealed_rows: Sequence[dict[str, Any]],
    max_margin: float = 0.05,
) -> ShortcutProbeReport:
    """Run all five out-of-fold shortcut probes (a)-(e) grouped by `scenario_id` (design section 4 [P7])."""
    if not sealed_rows:
        raise ValueError("sealed_rows must not be empty")

    scenario_ids = sorted({str(r["scenario_id"]) for r in sealed_rows})

    # Collect per-type samples across all rows.
    # (a) Question-only classifier on matched `U` vs settled `S` stems (`Who ...?` / `Why ...?`).
    probe_a_samples: list[tuple[str, list[str], str]] = []
    who_u_flags: list[int] = []
    why_u_flags: list[int] = []

    # (b) First-mentioned and last-mentioned location heuristics on B1 and B2.
    b1_true_flags: list[int] = []
    b2_true_flags: list[int] = []
    b1_first_hits = 0
    b1_last_hits = 0
    b1_total = 0
    b2_first_hits = 0
    b2_last_hits = 0
    b2_total = 0
    probe_b1_num: list[tuple[str, tuple[float, ...], int]] = []
    probe_b2_num: list[tuple[str, tuple[float, ...], int]] = []

    # (c) Mention-count and verb-count heuristics on X.
    x_unnotified_flags: list[int] = []
    probe_c_num: list[tuple[str, tuple[float, ...], int]] = []

    # (d) Name-to-event sentence distance and n-grams on P and K.
    p_yes_flags: list[int] = []
    k_yes_flags: list[int] = []
    probe_dp_num: list[tuple[str, tuple[float, ...], int]] = []
    probe_dp_ngrams: list[tuple[str, list[str], str]] = []
    probe_dk_num: list[tuple[str, tuple[float, ...], int]] = []
    probe_dk_ngrams: list[tuple[str, list[str], str]] = []

    # (e) Question order shuffled per bundle: position index -> question_type.
    probe_e_samples: list[tuple[str, list[str], str]] = []

    for row in sealed_rows:
        scen_id = str(row["scenario_id"])
        prose = str(row["prose"])
        q_list = row["questions"]

        # Gather all location values appearing in B1/B2 for this scenario.
        b_locs: list[str] = []
        for q in q_list:
            if q["question_type"] in ("B1", "B2"):
                b_locs.append(str(q["key"]))
                b_locs.append(str(q["ego_lure_key"]))
                b_locs.append(str(q["pressure_lure_answer"]))

        ordered_locs = _extract_candidate_locations_in_prose_order(prose, b_locs)
        first_loc = ordered_locs[0] if ordered_locs else ""
        last_loc = _extract_last_mentioned_location(prose, b_locs) or ""

        for pos_idx, q in enumerate(q_list):
            qtype = str(q["question_type"])
            qtext = str(q["question_text"])
            key_val = q["key"]
            lure_val = q["ego_lure_key"]
            t_agent = str(q.get("target_agent", ""))
            q_var = str(q.get("queried_variable", ""))

            # Probe (e): position index feature
            probe_e_samples.append((scen_id, [f"pos_{pos_idx}"], qtype))

            if qtype in ("U", "S"):
                label_u = "U" if qtype == "U" else "S"
                q_tokens = _tokenize(qtext)
                # Normalize proper names to `<NAME>` so the classifier focuses on stem structure + vocabulary.
                norm_tokens = ["<NAME>" if tok.capitalize() in prose else tok for tok in q_tokens]
                probe_a_samples.append((scen_id, _ngrams(norm_tokens, max_n=2), label_u))
                if qtext.startswith("Who "):
                    who_u_flags.append(1 if qtype == "U" else 0)
                elif qtext.startswith("Why "):
                    why_u_flags.append(1 if qtype == "U" else 0)

            elif qtype == "B1":
                is_true_belief = 1 if str(key_val) == str(lure_val) else 0
                b1_true_flags.append(is_true_belief)
                if str(key_val).lower() == first_loc.lower():
                    b1_first_hits += 1
                if str(key_val).lower() == last_loc.lower():
                    b1_last_hits += 1
                b1_total += 1
                first_count = float(prose.lower().count(first_loc.lower())) if first_loc else 0.0
                last_count = float(prose.lower().count(last_loc.lower())) if last_loc else 0.0
                probe_b1_num.append((scen_id, (first_count, last_count, first_count - last_count), is_true_belief))

            elif qtype == "B2":
                is_true_b2 = 1 if str(key_val) == str(lure_val) else 0
                b2_true_flags.append(is_true_b2)
                if str(key_val).lower() == first_loc.lower():
                    b2_first_hits += 1
                if str(key_val).lower() == last_loc.lower():
                    b2_last_hits += 1
                b2_total += 1
                first_count = float(prose.lower().count(first_loc.lower())) if first_loc else 0.0
                last_count = float(prose.lower().count(last_loc.lower())) if last_loc else 0.0
                probe_b2_num.append((scen_id, (first_count, last_count, first_count - last_count), is_true_b2))

            elif qtype == "X":
                is_unnotified = 1 if str(key_val) != "none" else 0
                x_unnotified_flags.append(is_unnotified)
                sentences = _sentence_split(prose)
                ag_low = t_agent.lower()
                mention_count = float(len(re.findall(rf"\b{re.escape(ag_low)}\b", prose.lower())))
                ag_sents = [s for s in sentences if re.search(rf"\b{re.escape(ag_low)}\b", s.lower())]
                verb_count = float(
                    sum(1 for s in ag_sents for tok in _tokenize(s) if tok in VERB_TOKENS)
                )
                sent_count = float(len(ag_sents))
                probe_c_num.append((scen_id, (mention_count, verb_count, sent_count), is_unnotified))

            elif qtype == "P":
                is_yes = 1 if str(key_val) == "yes" else 0
                p_yes_flags.append(is_yes)
                obj_name = q_var.split(":", 1)[-1] if ":" in q_var else q_var
                num_f, ng_f = _agent_event_distance_features(prose, t_agent, obj_name)
                probe_dp_num.append((scen_id, num_f, is_yes))
                probe_dp_ngrams.append((scen_id, ng_f, str(key_val)))

            elif qtype == "K":
                is_yes = 1 if str(key_val) == "yes" else 0
                k_yes_flags.append(is_yes)
                obj_name = q_var.split(":", 1)[-1] if ":" in q_var else q_var
                num_f, ng_f = _agent_event_distance_features(prose, t_agent, obj_name)
                probe_dk_num.append((scen_id, num_f, is_yes))
                probe_dk_ngrams.append((scen_id, ng_f, str(key_val)))

    # Compute base rates and OOF accuracies for (a)-(e).
    # (a) Question-only classifier on U vs S stems
    u_count = sum(1 for _, _, lbl in probe_a_samples if lbl == "U")
    a_base = max(u_count, len(probe_a_samples) - u_count) / max(len(probe_a_samples), 1)
    a_oof = _oof_multinomial_nb(probe_a_samples)

    # (b1) First/last location heuristic + OOF positional classifier on B1
    b1_pos_count = sum(b1_true_flags)
    b1_base = max(b1_pos_count, len(b1_true_flags) - b1_pos_count) / max(len(b1_true_flags), 1)
    b1_first_acc = b1_first_hits / max(b1_total, 1)
    b1_last_acc = b1_last_hits / max(b1_total, 1)
    b1_clf_acc = _oof_numeric_classifier(probe_b1_num)
    b1_oof = max(b1_first_acc, b1_last_acc, b1_clf_acc)

    # (b2) First/last location heuristic + OOF positional classifier on B2
    b2_pos_count = sum(b2_true_flags)
    b2_base = max(b2_pos_count, len(b2_true_flags) - b2_pos_count) / max(len(b2_true_flags), 1)
    b2_first_acc = b2_first_hits / max(b2_total, 1)
    b2_last_acc = b2_last_hits / max(b2_total, 1)
    b2_clf_acc = _oof_numeric_classifier(probe_b2_num)
    b2_oof = max(b2_first_acc, b2_last_acc, b2_clf_acc)

    # (c) Mention-count and verb-count probe on X
    x_pos_count = sum(x_unnotified_flags)
    c_base = max(x_pos_count, len(x_unnotified_flags) - x_pos_count) / max(len(x_unnotified_flags), 1)
    c_oof = _oof_numeric_classifier(probe_c_num)

    # (d_P) Sentence/token distance and surface n-grams on P
    p_pos_count = sum(p_yes_flags)
    dp_base = max(p_pos_count, len(p_yes_flags) - p_pos_count) / max(len(p_yes_flags), 1)
    dp_dist_acc = _oof_numeric_classifier(probe_dp_num)
    dp_ngram_acc = _oof_multinomial_nb(probe_dp_ngrams)
    dp_oof = max(dp_dist_acc, dp_ngram_acc)

    # (d_K) Sentence/token distance and surface n-grams on K
    k_pos_count = sum(k_yes_flags)
    dk_base = max(k_pos_count, len(k_yes_flags) - k_pos_count) / max(len(k_yes_flags), 1)
    dk_dist_acc = _oof_numeric_classifier(probe_dk_num)
    dk_ngram_acc = _oof_multinomial_nb(probe_dk_ngrams)
    dk_oof = max(dk_dist_acc, dk_ngram_acc)

    # (e) Question order / bundle position probe
    e_counts = Counter(lbl for _, _, lbl in probe_e_samples)
    e_base = max(e_counts.values()) / max(len(probe_e_samples), 1)
    e_oof = _oof_multinomial_nb(probe_e_samples)

    def _make_item(pid: str, desc: str, n_samp: int, base: float, oof: float) -> ProbeItemResult:
        margin = oof - base
        return ProbeItemResult(
            probe_id=pid,
            description=desc,
            num_samples=n_samp,
            base_rate=round(base, 4),
            oof_accuracy=round(oof, 4),
            margin_over_base=round(margin, 4),
            max_allowed_margin=max_margin,
            passed=(margin <= max_margin + 1e-9),
        )

    probe_items = (
        _make_item(
            "a_question_only_u_vs_settled",
            "(a) Question-only lexical classifier (matched settled vs U stems)",
            len(probe_a_samples),
            a_base,
            a_oof,
        ),
        _make_item(
            "b1_location_heuristics_b1",
            "(b) First/last-mentioned location heuristics & OOF classifier on B1",
            len(probe_b1_num),
            b1_base,
            b1_oof,
        ),
        _make_item(
            "b2_location_heuristics_b2",
            "(b) First/last-mentioned location heuristics & OOF classifier on B2",
            len(probe_b2_num),
            b2_base,
            b2_oof,
        ),
        _make_item(
            "c_mention_verb_count_x",
            "(c) Mention-count and verb-count heuristics & OOF classifier on X",
            len(probe_c_num),
            c_base,
            c_oof,
        ),
        _make_item(
            "d_distance_ngrams_p",
            "(d) Name-to-event sentence/token distance and n-grams on P",
            len(probe_dp_num),
            dp_base,
            dp_oof,
        ),
        _make_item(
            "d_distance_ngrams_k",
            "(d) Name-to-event sentence/token distance and n-grams on K",
            len(probe_dk_num),
            dk_base,
            dk_oof,
        ),
        _make_item(
            "e_bundle_question_order",
            "(e) Shuffled bundle question-position classifier (8 question types)",
            len(probe_e_samples),
            e_base,
            e_oof,
        ),
    )

    u_share = u_count / max(len(probe_a_samples), 1)
    who_u = sum(who_u_flags) / max(len(who_u_flags), 1)
    why_u = sum(why_u_flags) / max(len(why_u_flags), 1)
    b1_share = sum(b1_true_flags) / max(len(b1_true_flags), 1)
    b2_share = sum(b2_true_flags) / max(len(b2_true_flags), 1)
    x_share = sum(x_unnotified_flags) / max(len(x_unnotified_flags), 1)
    p_share = sum(p_yes_flags) / max(len(p_yes_flags), 1)
    k_share = sum(k_yes_flags) / max(len(k_yes_flags), 1)

    all_shares = (u_share, who_u, why_u, b1_share, b2_share, x_share, p_share, k_share)
    is_balanced = all(abs(s - 0.5) <= 0.02 for s in all_shares)

    balance = ClassBalanceReport(
        u_share_among_open_stems=round(u_share, 4),
        who_u_share=round(who_u, 4),
        why_u_share=round(why_u, 4),
        b1_true_belief_share=round(b1_share, 4),
        b2_true_belief_share=round(b2_share, 4),
        x_unnotified_share=round(x_share, 4),
        p_yes_share=round(p_share, 4),
        k_yes_share=round(k_share, 4),
        balanced=is_balanced,
    )

    overall_pass = is_balanced and all(p.passed for p in probe_items)
    return ShortcutProbeReport(
        num_scenarios=len(scenario_ids),
        num_renderings=len(sealed_rows),
        probes=probe_items,
        balance=balance,
        passed=overall_pass,
    )
