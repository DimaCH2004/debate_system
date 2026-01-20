import json
import logging
import os
from typing import Dict, Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebateSystem:
    def __init__(self):
        from llm_client import LLMClient
        from role_assignment import RoleAssignment
        from solution_generation import SolutionGenerator
        from peer_review import PeerReviewSystem
        from refinement import RefinementSystem
        from judgment import JudgmentSystem

        self.llm_client = LLMClient()
        self.gemini_instances = ["gemini-1", "gemini-2", "gemini-3", "gemini-4"]

        self.role_assigner = RoleAssignment(self.llm_client)
        self.solution_generator = SolutionGenerator(self.llm_client)
        self.peer_review_system = PeerReviewSystem(self.llm_client)
        self.refinement_system = RefinementSystem(self.llm_client)
        self.judgment_system = JudgmentSystem(self.llm_client)

    def load_problem(self, problem_id: int) -> Dict[str, Any]:
        try:
            with open("dataset/problems.json", "r", encoding='utf-8') as f:
                problems = json.load(f)

            problem_data = problems.get(str(problem_id))
            if not problem_data:
                raise KeyError(f"Problem {problem_id} not found")

            return {
                "id": problem_id,
                "category": problem_data["category"],
                "question": problem_data["question"],
                "verifiable_answer": problem_data["verifiable_answer"]
            }
        except FileNotFoundError:
            logger.error("dataset/problems.json not found")
            return {
                "id": problem_id,
                "category": "probability",
                "question": "A fair coin is flipped 10 times. What is the probability that the number of heads is exactly twice the number of tails?",
                "verifiable_answer": 0.0
            }

    def run_debate(self, problem_id: int) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"Starting debate for Problem {problem_id}")
        logger.info("=" * 60)

        problem = self.load_problem(problem_id)
        logger.info(f"Problem: {problem['question'][:100]}...")

        logger.info("\n[Stage 0] Role Assignment")
        preferences = {}
        for instance_name in self.gemini_instances:
            logger.info(f"  Getting role preference from {instance_name}")
            preference = self.role_assigner.get_role_preference(
                instance_name, problem['question']
            )
            preferences[instance_name] = preference

        roles = self.role_assigner.assign_roles(preferences)
        logger.info(f"  Assigned Judge: {roles['judge']}")
        logger.info(f"  Assigned Solvers: {', '.join(roles['solvers'])}")

        logger.info("\n[Stage 1] Independent Solution Generation")
        solutions = self.solution_generator.generate_all_solutions(
            roles["solvers"], problem['question']
        )
        logger.info(f"  Generated {len(solutions)} solutions")

        logger.info("\n[Stage 2] Peer Review Round")
        reviews = self.peer_review_system.conduct_peer_review(
            roles["solvers"], solutions, problem['question']
        )
        total_reviews = sum(len(r) for r in reviews.values())
        logger.info(f"  Generated {total_reviews} peer reviews")

        logger.info("\n[Stage 3] Refinement Based on Feedback")
        refined_solutions = self.refinement_system.refine_all_solutions(
            solutions, reviews, problem['question']
        )
        logger.info(f"  Refined {len(refined_solutions)} solutions")

        logger.info("\n[Stage 4] Final Judgment")
        judgment = self.judgment_system.make_judgment(
            roles["judge"],
            problem['question'],
            solutions,
            reviews,
            refined_solutions
        )

        if not isinstance(judgment, dict):
            judgment = {
                "winner": roles["solvers"][0],
                "confidence": 0.8,
                "reasoning": "Default judgment",
                "evaluation_criteria": {},
                "ranking": {}
            }

        winner = judgment.get('winner', roles["solvers"][0])
        confidence = judgment.get('confidence', 0.8)

        logger.info(f"  Winner: {winner}")
        logger.info(f"  Confidence: {confidence:.2%}")

        result = {
            "problem_id": problem_id,
            "problem": problem['question'],
            "category": problem['category'],
            "roles": roles,
            "preferences": {k: v.to_dict() for k, v in preferences.items()},
            "original_solutions": {k: v.to_dict() for k, v in solutions.items()},
            "peer_reviews": {
                solver: [r.to_dict() for r in review_list]
                for solver, review_list in reviews.items()
            },
            "refined_solutions": {k: v.to_dict() for k, v in refined_solutions.items()},
            "judgment": judgment,
            "verifiable_answer": problem['verifiable_answer'],
            "timestamp": datetime.now().isoformat()
        }

        self.save_results(problem_id, result)
        return result

    def save_results(self, problem_id: int, result: Dict[str, Any]):
        os.makedirs("results", exist_ok=True)
        filename = f"results/debate_problem_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to {filename}")


def main():
    print("=" * 60)
    print("MULTI-GEMINI COLLABORATIVE DEBATE SYSTEM")
    print("4 Gemini Instances Working Together")
    print("=" * 60)

    from dotenv import load_dotenv
    load_dotenv()

    use_mock = os.getenv("USE_MOCK_MODE", "true").lower() == "true"
    if use_mock:
        print("Running in MOCK MODE - Using simulated responses")
        print("Set USE_MOCK_MODE=false in .env to use real Gemini API")
    else:
        print("Running with REAL Gemini API")
        print("Using 4 separate Gemini instances")

    print("=" * 60)

    system = DebateSystem()

    try:
        problem_id = int(input("\nEnter problem ID (1-25, default 2): ") or "2")
        if problem_id < 1 or problem_id > 25:
            print("Invalid problem ID. Using default: 2")
            problem_id = 2
    except ValueError:
        print("Invalid input. Using default problem ID: 2")
        problem_id = 2

    try:
        result = system.run_debate(problem_id)

        print("\n" + "=" * 60)
        print("DEBATE RESULTS")
        print("=" * 60)
        print(f"Problem ID: {result['problem_id']}")
        print(f"Category: {result['category']}")
        print(f"Problem: {result['problem'][:100]}...")

        print("\nRoles:")
        print(f"   Judge: {result['roles']['judge']}")
        print(f"   Solvers: {', '.join(result['roles']['solvers'])}")

        print(f"\nWinner: {result['judgment']['winner']}")
        print(f"Confidence: {result['judgment']['confidence']:.1%}")

        winner = result['judgment']['winner']
        refined_answer_raw = ""
        if winner in result['refined_solutions']:
            refined_answer_raw = result['refined_solutions'][winner]['refined_answer']
            print(f"\nWinner's Answer: {refined_answer_raw}")

        verifiable = result['verifiable_answer']
        print(f"Correct Answer: {verifiable}")

        def normalize_answer(ans):
            if ans is None:
                return ""
            ans_str = str(ans).strip().lower()
            ans_str = ans_str.replace(",", "").replace(" ", "").rstrip(".")
            try:
                return float(ans_str)
            except Exception:
                return ans_str

        refined_norm = normalize_answer(refined_answer_raw)
        verifiable_norm = normalize_answer(verifiable)

        try:
            if isinstance(refined_norm, float) and isinstance(verifiable_norm, float):
                if abs(refined_norm - verifiable_norm) < 0.01:
                    print("\nSUCCESS: Answer is correct!")
                else:
                    print(f"\nINCORRECT: Expected {verifiable_norm}, got {refined_norm}")
            elif refined_norm == verifiable_norm:
                print("\nSUCCESS: Answer is correct!")
            else:
                print(f"\nINCORRECT: Expected {verifiable_norm}, got {refined_norm}")
        except Exception:
            print("\nCould not verify answer")

        print(f"\nFull results saved to: results/debate_problem_{problem_id}_*.json")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error running debate: {e}", exc_info=True)
        print(f"\nDebate failed: {e}")


if __name__ == "__main__":
    main()
