import json
import glob
import os
from collections import Counter


def normalize_answer(ans):
    if ans is None:
        return ""
    s = str(ans).strip().lower()
    s = s.replace(",", "").replace(" ", "").rstrip(".")
    try:
        return float(s)
    except Exception:
        return s


def is_correct(pred, truth):
    p = normalize_answer(pred)
    t = normalize_answer(truth)
    if isinstance(p, float) and isinstance(t, float):
        return abs(p - t) < 0.01
    return p == t


def majority_vote(values):
    vals = [normalize_answer(v) for v in values if str(v).strip() != ""]
    if not vals:
        return ""
    c = Counter(vals)
    top_count = c.most_common(1)[0][1]
    top = [k for k, v in c.items() if v == top_count]
    return sorted(top, key=lambda x: str(x))[0]


def main():
    files = sorted(glob.glob(os.path.join("results", "debate_problem_*.json")))
    if not files:
        print("No results found in results/.")
        return

    n = 0
    system_correct = 0
    consensus = 0
    improved = 0

    disagree_cases = 0
    judge_correct_on_disagree = 0

    single_correct = 0
    vote_correct = 0
    skipped_single = 0
    skipped_vote = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)

        truth = r.get("verifiable_answer")
        solvers = r.get("roles", {}).get("solvers", [])
        refined = r.get("refined_solutions", {})
        original = r.get("original_solutions", {})
        winner = r.get("judgment", {}).get("winner", "")

        n += 1

        winner_refined = refined.get(winner, {}).get("refined_answer", "")
        if is_correct(winner_refined, truth):
            system_correct += 1

        refined_answers = [refined.get(s, {}).get("refined_answer", "") for s in solvers]
        refined_norm = [normalize_answer(a) for a in refined_answers]

        if len(refined_norm) == 3 and refined_norm[0] != "" and refined_norm[0] == refined_norm[1] == refined_norm[2]:
            consensus += 1

        unique_refined = {normalize_answer(a) for a in refined_answers if str(a).strip() != ""}
        if len(unique_refined) >= 2:
            disagree_cases += 1
            if is_correct(winner_refined, truth):
                judge_correct_on_disagree += 1

        improved_this_problem = False
        for s in solvers:
            init_ans = original.get(s, {}).get("final_answer", "")
            ref_ans = refined.get(s, {}).get("refined_answer", "")
            if (not is_correct(init_ans, truth)) and is_correct(ref_ans, truth):
                improved_this_problem = True
        if improved_this_problem:
            improved += 1

        if solvers:
            single_ans = original.get(solvers[0], {}).get("final_answer", "")
            if str(single_ans).strip() == "":
                skipped_single += 1
            else:
                if is_correct(single_ans, truth):
                    single_correct += 1
        else:
            skipped_single += 1

        orig_answers = [original.get(s, {}).get("final_answer", "") for s in solvers]
        if any(str(a).strip() != "" for a in orig_answers):
            vote_ans = majority_vote(orig_answers)
            if str(vote_ans).strip() == "":
                skipped_vote += 1
            else:
                if is_correct(vote_ans, truth):
                    vote_correct += 1
        else:
            skipped_vote += 1

    def pct(x, denom):
        return 0.0 if denom == 0 else 100.0 * x / denom

    print(f"Files evaluated: {n}\n")

    print("System metrics")
    print(f"Overall Accuracy: {system_correct}/{n} = {pct(system_correct, n):.1f}%")
    print(f"Improvement Rate: {improved}/{n} = {pct(improved, n):.1f}%")
    print(f"Consensus Rate: {consensus}/{n} = {pct(consensus, n):.1f}%")
    if disagree_cases:
        print(f"Judge Accuracy (on disagreements): {judge_correct_on_disagree}/{disagree_cases} = {pct(judge_correct_on_disagree, disagree_cases):.1f}%")
    else:
        print("Judge Accuracy (on disagreements): no disagreement cases found")

    print("\nBaselines (from original_solutions.final_answer)")
    denom_single = n - skipped_single
    denom_vote = n - skipped_vote
    print(f"Single-LLM Accuracy: {single_correct}/{denom_single} = {pct(single_correct, denom_single):.1f}% (skipped {skipped_single})")
    print(f"Voting Accuracy: {vote_correct}/{denom_vote} = {pct(vote_correct, denom_vote):.1f}% (skipped {skipped_vote})")

    if skipped_single or skipped_vote:
        print("\nSome baseline cases were skipped because original final_answer was empty.")
        print("That usually means the model output got cut off before 'Final Answer:' appeared.")


if __name__ == "__main__":
    main()
