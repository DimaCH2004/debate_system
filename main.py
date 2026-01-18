import json
import logging
from typing import Dict, Any
from datetime import datetime

from config import LLM_PROVIDERS, Problem
from role_assignment import RoleAssignment
from solution_generation import SolutionGenerator
from peer_review import PeerReviewSystem
from refinement import RefinementSystem
from judgment import JudgmentSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DebateSystem:
    def __init__(self):
        self.llm_providers = LLM_PROVIDERS
        self.role_assigner = RoleAssignment(self.llm_providers)
        self.solution_generator = SolutionGenerator(self.llm_providers)
        self.peer_review_system = PeerReviewSystem(self.llm_providers)
        self.refinement_system = RefinementSystem(self.llm_providers)
        self.judgment_system = JudgmentSystem(self.llm_providers)

    def load_problem(self, problem_id: int) -> Problem:
        try:
            with open("dataset/problems.json", "r", encoding='utf-8') as f:
                problems = json.load(f)
            problem_data = problems[str(problem_id)]
            return Problem(
                id=problem_id,
                category=problem_data["category"],
                question=problem_data["question"],
                verifiable_answer=problem_data["verifiable_answer"]
            )
        except FileNotFoundError:
            logger.error("dataset/problems.json not found. Creating a sample problem.")
            # Create a sample problem if file doesn't exist
            return Problem(
                id=problem_id,
                category="probability",
                question="A fair coin is flipped 10 times. What is the probability that the number of heads is exactly twice the number of tails?",
                verifiable_answer=0.0
            )
        except KeyError:
            logger.error(f"Problem {problem_id} not found in dataset.")
            raise

    def run_debate(self, problem_id: int) -> Dict[str, Any]:
        logger.info(f"Starting debate for Problem {problem_id}")

        problem = self.load_problem(problem_id)
        logger.info(f"Loaded problem: {problem.category}")

        logger.info("Stage 0: Role Assignment")
        preferences = {}
        for llm_name in self.llm_providers.keys():
            logger.info(f"Getting role preference for {llm_name}")
            preference = self.role_assigner.get_role_preference(llm_name, problem.question)
            preferences[llm_name] = preference

        roles = self.role_assigner.assign_roles_deterministic(preferences)
        logger.info(f"Assigned roles: {roles}")

        logger.info("Stage 1: Independent Solution Generation")
        solutions = self.solution_generator.generate_all_solutions(roles["solvers"], problem.question)
        logger.info(f"Generated {len(solutions)} solutions")

        logger.info("Stage 2: Peer Review Round")
        solution_dicts = {k: v.to_dict() for k, v in solutions.items()}
        reviews = self.peer_review_system.conduct_peer_review(roles["solvers"], solution_dicts, problem.question)
        logger.info(f"Generated {sum(len(r) for r in reviews.values())} reviews")

        logger.info("Stage 3: Refinement Based on Feedback")
        refined_solutions = self.refinement_system.refine_all_solutions(
            solutions, reviews, problem.question
        )
        logger.info(f"Refined {len(refined_solutions)} solutions")

        logger.info("Stage 4: Final Judgment")
        judgment = self.judgment_system.make_judgment(
            roles["judge"],
            problem.question,
            solutions,
            reviews,
            refined_solutions
        )

        result = {
            "problem_id": problem_id,
            "problem": problem.question,
            "roles": roles,
            "original_solutions": {k: v.to_dict() for k, v in solutions.items()},
            "peer_reviews": {
                solver: [r.to_dict() for r in review_list]
                for solver, review_list in reviews.items()
            },
            "refined_solutions": {k: v.to_dict() for k, v in refined_solutions.items()},
            "judgment": judgment.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "verifiable_answer": problem.verifiable_answer
        }

        self.save_results(problem_id, result)

        logger.info(f"Debate complete. Winner: {judgment.winner} (confidence: {judgment.confidence:.2%})")
        return result

    def save_results(self, problem_id: int, result: Dict[str, Any]):
        import os
        os.makedirs("results", exist_ok=True)
        filename = f"results/debate_problem_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {filename}")


def main():
    system = DebateSystem()

    print("=" * 60)
    print("LLM DEBATE SYSTEM")
    print("=" * 60)
    print("Running in MOCK MODE - Using simulated responses")
    print("Set USE_MOCK_MODE=false in .env file to use real API calls")
    print("=" * 60)

    try:
        problem_id = int(input("Enter problem ID (1-25, default 2): ") or "2")
    except ValueError:
        print("Invalid input. Using default problem ID 2.")
        problem_id = 2

    try:
        result = system.run_debate(problem_id)

        print("\n" + "=" * 60)
        print("DEBATE RESULTS")
        print("=" * 60)
        print(f"Problem ID: {result['problem_id']}")
        print(f"Problem Type: {result.get('problem_category', 'N/A')}")
        print(f"Problem: {result['problem'][:80]}...")

        print(f"\nRoles:")
        print(f"  Judge: {result['roles']['judge']}")
        print(f"  Solvers: {', '.join(result['roles']['solvers'])}")

        print(f"\nWinner: {result['judgment']['winner']}")
        print(f"Judge Confidence: {result['judgment']['confidence']:.2%}")

        # Display reasoning
        reasoning = result['judgment']['reasoning']
        if reasoning:
            print(f"\nReasoning: {reasoning[:150]}...")
        else:
            print(f"\nReasoning: Not provided")

        # Safe access to refined answer
        winner = result['judgment']['winner']
        if winner and winner in result['refined_solutions']:
            refined_answer = result['refined_solutions'][winner].get('refined_answer', 'Not available')
            print(f"\nWinner's Refined Answer: {refined_answer}")
        else:
            print(f"\nWinner's Refined Answer: Not available")

        print(f"Verifiable Answer: {result['verifiable_answer']}")

        # Compare refined answer with verifiable answer
        if winner and winner in result['refined_solutions']:
            refined = result['refined_solutions'][winner].get('refined_answer', '')
            verifiable = str(result['verifiable_answer'])

            try:
                # Try to convert both to float for comparison
                refined_clean = refined.strip()
                verifiable_clean = verifiable.strip()

                # Handle different number formats
                def parse_number(num_str):
                    try:
                        return float(num_str)
                    except:
                        return None

                refined_num = parse_number(refined_clean)
                verifiable_num = parse_number(verifiable_clean)

                if refined_num is not None and verifiable_num is not None:
                    tolerance = 0.001
                    if abs(refined_num - verifiable_num) < tolerance:
                        print("\n✅ SUCCESS: Refined answer matches verifiable answer!")
                    else:
                        print(
                            f"\n❌ MISMATCH: Refined answer ({refined_num}) differs from verifiable answer ({verifiable_num})")
                else:
                    # String comparison
                    if refined_clean == verifiable_clean:
                        print("\n✅ SUCCESS: Refined answer matches verifiable answer!")
                    else:
                        print(
                            f"\n❌ MISMATCH: Refined answer '{refined_clean}' differs from verifiable answer '{verifiable_clean}'")
            except Exception as e:
                print(f"\n⚠️  Could not compare answers: {e}")

        print(f"\nFull results saved to: results/debate_problem_{problem_id}_*.json")

    except Exception as e:
        logger.error(f"Error running debate: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ Debate failed with error: {e}")


if __name__ == "__main__":
    main()