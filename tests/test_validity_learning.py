import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.controller.controller import ControllerPhase
from src.data.graph_loader import GraphDataset
from src.scb.problem import SCBProblem
from src.environment.environment import SCBEnvironment
from src.models.scb_encoder import SCBEncoder
from src.models.policies import ValidityPolicy
from src.models.value_head import SCBValueHead
from src.training.rollout_buffer import RolloutBuffer
from src.training.advantages import compute_gae
from src.training.ppo_updater import PPOUpdater


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "data/raw/labelled_dataset.pkl"

SEED = 42

MAX_NODES = 12
MAX_SESSIONS = 6

NUM_TRAIN_GRAPHS = 20
NUM_TEST_GRAPHS = 10

TRAIN_EPISODES = 2000
MAX_STEPS = 30

EVAL_EPISODES = 100
PRINT_EVERY = 100

LEARNING_RATE = 3e-4
PPO_EPOCHS = 4

GAMMA = 0.99
GAE_LAMBDA = 0.95

HIDDEN_DIM = 32
ACTION_EMBEDDING_DIM = 16
SCORER_HIDDEN_DIM = 32


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# OUTPUT LOGGING
# ============================================================

OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "validity_learning.txt"


class ValidityTestEnvironment(SCBEnvironment):

    @staticmethod
    def _normalize_state(state):
        state.separated_sessions = len(
            state.separated_sessions
        )
        return state

    def reset(self, selected_sessions=None):
        state = super().reset(
            selected_sessions
        )

        state = self._normalize_state(state)

        self.state = state

        return state.copy()

    def step(self, action):
        state, reward, done, info = super().step(
            action
        )

        state = self._normalize_state(state)

        self.state = state

        return state, reward, done, info


class Tee:
    """Print to terminal and save the same output to a file."""

    def __init__(self, path):
        self.file = open(
            path,
            "w",
            encoding="utf-8",
        )

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# PPO MODEL ADAPTER
# ============================================================

class ValidityPPOModel(nn.Module):
    """
    Test model combining:

        SCBEncoder
            +
        ValidityPolicy
            +
        SCBValueHead

    PPOUpdater interacts with this through evaluate_action().
    """

    def __init__(
        self,
        encoder,
        policy,
        value_head,
    ):
        super().__init__()

        self.encoder = encoder
        self.policy = policy
        self.value_head = value_head

        self._problem = None

    def set_problem(self, problem):
        self._problem = problem

    def evaluate_action(self, transition):

        if self._problem is None:
            raise RuntimeError(
                "ValidityPPOModel has no active SCBProblem."
            )

        state = transition.state

        encoder_output = self.encoder(
            self._problem,
            state,
        )

        action_logits = self.policy.scorer(
            encoder_output
        )

        flat_logits = self.policy._build_flat_logits(
            action_logits,
            self._problem.num_edges,
            self._problem.num_sessions,
        )

        mask = self.policy._build_action_mask(
            self._problem,
            state,
        )

        masked_logits = self.policy._apply_mask(
            flat_logits,
            mask,
        )

        distribution = torch.distributions.Categorical(
            logits=masked_logits
        )

        action_index = torch.tensor(
            transition.action_index,
            device=masked_logits.device,
            dtype=torch.long,
        )

        new_log_prob = distribution.log_prob(
            action_index
        )

        entropy = distribution.entropy()

        value = self.value_head(
            encoder_output.global_embedding
        )

        return (
            new_log_prob,
            entropy,
            value.squeeze(),
        )


# ============================================================
# SMALL GRAPH SELECTION
# ============================================================

def select_small_graphs(dataset):

    candidates = []

    for index in range(len(dataset)):

        graph = dataset.graph(index)

        if graph.num_nodes > MAX_NODES:
            continue

        if graph.num_sessions > MAX_SESSIONS:
            continue

        candidates.append(
            (
                index,
                graph.num_nodes,
                graph.num_edges,
                graph.num_sessions,
            )
        )

    required = (
        NUM_TRAIN_GRAPHS
        + NUM_TEST_GRAPHS
    )

    if len(candidates) < required:
        raise RuntimeError(
            "Not enough small graphs in dataset.\n"
            f"Found : {len(candidates)}\n"
            f"Need  : {required}\n"
            f"MAX_NODES={MAX_NODES}\n"
            f"MAX_SESSIONS={MAX_SESSIONS}"
        )

    random.shuffle(candidates)

    train = candidates[:NUM_TRAIN_GRAPHS]

    test = candidates[
        NUM_TRAIN_GRAPHS:
        NUM_TRAIN_GRAPHS + NUM_TEST_GRAPHS
    ]

    return train, test


# ============================================================
# ENVIRONMENT
# ============================================================

def make_environment(graph):

    problem = SCBProblem.from_graph(graph)

    environment = ValidityTestEnvironment(
        problem=problem,
        mode=SCBEnvironment.VALIDITY_MODE,
    )
    return environment, problem


# ============================================================
# ROLLOUT
# ============================================================

def collect_rollout(
    model,
    graph,
    max_steps=MAX_STEPS,
):

    environment, problem = make_environment(graph)

    model.set_problem(problem)

    state = environment.reset()

    buffer = RolloutBuffer()

    reached_valid = bool(state.valid)

    steps_to_valid = (
        0
        if reached_valid
        else None
    )

    total_reward = 0.0

    for step in range(max_steps):

        encoder_output = model.encoder(
            problem,
            state,
        )

        policy_output = model.policy(
            encoder_output=encoder_output,
            problem=problem,
            state=state,
            deterministic=False,
        )

        action = policy_output.action

        next_state, reward, done, info = (
            environment.step(action)
        )

        total_reward += float(reward)

        if (
            not reached_valid
            and next_state.valid
        ):
            reached_valid = True
            steps_to_valid = step + 1

        buffer.add(
            state=state,
            action=action,
            action_index=policy_output.action_index,
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            log_prob=policy_output.log_prob,
            entropy=policy_output.entropy,
            phase=ControllerPhase.VALIDITY,
            encoder_output=encoder_output,
        )

        state = next_state

        if done:
            break

    return (
        buffer,
        reached_valid,
        steps_to_valid,
        total_reward,
        state,
        problem,
    )


# ============================================================
# GAE TARGETS
# ============================================================

def compute_rollout_targets(
    model,
    rollout,
    problem,
):

    transitions = rollout.transitions

    if not transitions:
        raise RuntimeError(
            "Cannot compute GAE for empty rollout."
        )

    values = []
    next_values = []
    rewards = []
    dones = []

    for transition in transitions:

        state_output = model.encoder(
            problem,
            transition.state,
        )

        next_output = model.encoder(
            problem,
            transition.next_state,
        )

        value = model.value_head(
            state_output.global_embedding
        )

        next_value = model.value_head(
            next_output.global_embedding
        )

        values.append(value.squeeze())
        next_values.append(next_value.squeeze())

        rewards.append(
            float(transition.reward)
        )

        dones.append(
            float(transition.done)
        )

    values = torch.stack(values)
    next_values = torch.stack(next_values)

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=values.device,
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=values.device,
    )

    gae = compute_gae(
        rewards=rewards,
        values=values.detach(),
        next_values=next_values.detach(),
        dones=dones,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        normalize=True,
    )

    return gae


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    graphs,
):

    model.eval()

    successes = 0
    successful_steps = []

    for _ in range(EVAL_EPISODES):

        graph = random.choice(graphs)

        environment, problem = (
            make_environment(graph)
        )

        model.set_problem(problem)

        state = environment.reset()

        if state.valid:
            successes += 1
            successful_steps.append(0)
            continue

        for step in range(MAX_STEPS):

            encoder_output = model.encoder(
                problem,
                state,
            )

            policy_output = model.policy(
                encoder_output=encoder_output,
                problem=problem,
                state=state,
                deterministic=True,
            )

            state, reward, done, info = (
                environment.step(
                    policy_output.action
                )
            )

            if state.valid:

                successes += 1

                successful_steps.append(
                    step + 1
                )

                break

            if done:
                break

    model.train()

    validity_rate = (
        successes / EVAL_EPISODES
    )

    mean_steps = (
        sum(successful_steps)
        / len(successful_steps)
        if successful_steps
        else float("inf")
    )

    return (
        validity_rate,
        mean_steps,
    )


# ============================================================
# TRAINING
# ============================================================

def run_experiment():

    tee = Tee(OUTPUT_FILE)

    original_stdout = sys.stdout
    sys.stdout = tee

    try:

        print()
        print("=" * 78)
        print(
            "SCB VALIDITY POLICY PPO LEARNING EXPERIMENT"
        )
        print("=" * 78)

        print()
        print("CONFIGURATION")
        print("-" * 78)

        print(
            f"Dataset              : "
            f"{DATASET_PATH}"
        )

        print(
            f"Maximum nodes        : "
            f"{MAX_NODES}"
        )

        print(
            f"Maximum sessions     : "
            f"{MAX_SESSIONS}"
        )

        print(
            f"Training graphs      : "
            f"{NUM_TRAIN_GRAPHS}"
        )

        print(
            f"Held-out test graphs : "
            f"{NUM_TEST_GRAPHS}"
        )

        print(
            f"Training episodes    : "
            f"{TRAIN_EPISODES}"
        )

        print(
            f"Max steps / episode  : "
            f"{MAX_STEPS}"
        )

        print(
            f"Learning rate        : "
            f"{LEARNING_RATE}"
        )

        print(
            f"PPO epochs           : "
            f"{PPO_EPOCHS}"
        )

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        dataset = GraphDataset(
            DATASET_PATH
        )

        train_graphs, test_graphs = (
            select_small_graphs(dataset)
        )

        print()
        print("TRAIN GRAPHS")
        print("-" * 78)

        for (
            index,
            nodes,
            edges,
            sessions,
        ) in train_graphs:

            print(
                f"index={index:4d} "
                f"nodes={nodes:2d} "
                f"edges={edges:3d} "
                f"sessions={sessions:2d}"
            )

        print()
        print("HELD-OUT TEST GRAPHS")
        print("-" * 78)

        for (
            index,
            nodes,
            edges,
            sessions,
        ) in test_graphs:

            print(
                f"index={index:4d} "
                f"nodes={nodes:2d} "
                f"edges={edges:3d} "
                f"sessions={sessions:2d}"
            )

        train_graph_objects = [
            dataset.graph(item[0])
            for item in train_graphs
        ]

        test_graph_objects = [
            dataset.graph(item[0])
            for item in test_graphs
        ]

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        encoder = SCBEncoder(
            hidden_dim=HIDDEN_DIM,
            topology_layers=2,
            session_layers=1,
        )

        policy = ValidityPolicy(
            hidden_dim=HIDDEN_DIM,
            action_embedding_dim=(
                ACTION_EMBEDDING_DIM
            ),
            scorer_hidden_dim=(
                SCORER_HIDDEN_DIM
            ),
        )

        value_head = SCBValueHead(
            hidden_dim=HIDDEN_DIM,
            value_hidden_dim=HIDDEN_DIM,
        )

        model = ValidityPPOModel(
            encoder=encoder,
            policy=policy,
            value_head=value_head,
        )

        updater = PPOUpdater(
            model=model,
            learning_rate=LEARNING_RATE,
            ppo_epochs=PPO_EPOCHS,
            minibatch_size=None,
        )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        print()
        print("=" * 78)
        print("BASELINE EVALUATION")
        print("=" * 78)

        (
            train_rate_before,
            train_steps_before,
        ) = evaluate(
            model,
            train_graph_objects,
        )

        (
            test_rate_before,
            test_steps_before,
        ) = evaluate(
            model,
            test_graph_objects,
        )

        print(
            f"Train validity rate : "
            f"{train_rate_before:.3f}"
        )

        print(
            f"Train mean steps    : "
            f"{train_steps_before:.3f}"
        )

        print(
            f"Test validity rate  : "
            f"{test_rate_before:.3f}"
        )

        print(
            f"Test mean steps     : "
            f"{test_steps_before:.3f}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print()
        print("=" * 78)
        print("PPO TRAINING")
        print("=" * 78)

        window_success = []
        window_steps = []
        window_rewards = []

        for episode in range(
            1,
            TRAIN_EPISODES + 1,
        ):

            graph_info = random.choice(
                train_graphs
            )

            graph = dataset.graph(
                graph_info[0]
            )

            (
                rollout,
                reached_valid,
                steps_to_valid,
                total_reward,
                final_state,
                problem,
            ) = collect_rollout(
                model,
                graph,
                MAX_STEPS,
            )

            if len(rollout) == 0:
                continue

            targets = compute_rollout_targets(
                model,
                rollout,
                problem,
            )

            stats = updater.update(
                rollout,
                targets.advantages,
                targets.returns,
            )

            window_success.append(
                float(reached_valid)
            )

            window_rewards.append(
                total_reward
            )

            if steps_to_valid is not None:
                window_steps.append(
                    steps_to_valid
                )

            if episode % PRINT_EVERY == 0:

                validity_rate = (
                    sum(window_success)
                    / len(window_success)
                )

                mean_reward = (
                    sum(window_rewards)
                    / len(window_rewards)
                )

                mean_steps = (
                    sum(window_steps)
                    / len(window_steps)
                    if window_steps
                    else float("inf")
                )

                print()
                print(
                    f"[EPISODES "
                    f"{episode - PRINT_EVERY + 1}-"
                    f"{episode}]"
                )

                print(
                    f"  Rollout validity : "
                    f"{validity_rate:.3f}"
                )

                print(
                    f"  Mean reward      : "
                    f"{mean_reward:+.5f}"
                )

                print(
                    f"  Mean steps valid : "
                    f"{mean_steps:.3f}"
                )

                print()
                print("  PPO")

                print(
                    f"    policy loss    : "
                    f"{stats.policy_loss:+.6f}"
                )

                print(
                    f"    value loss     : "
                    f"{stats.value_loss:.6f}"
                )

                print(
                    f"    entropy        : "
                    f"{stats.entropy:.6f}"
                )

                print(
                    f"    total loss     : "
                    f"{stats.total_loss:+.6f}"
                )

                print(
                    f"    approx KL      : "
                    f"{stats.approx_kl:.6f}"
                )

                print(
                    f"    clip fraction  : "
                    f"{stats.clip_fraction:.6f}"
                )

                print(
                    f"    grad norm      : "
                    f"{stats.grad_norm:.6f}"
                )

                print(
                    f"    samples        : "
                    f"{stats.num_samples}"
                )

                window_success.clear()
                window_steps.clear()
                window_rewards.clear()

        # ----------------------------------------------------
        # FINAL EVALUATION
        # ----------------------------------------------------

        print()
        print("=" * 78)
        print("FINAL EVALUATION")
        print("=" * 78)

        (
            train_rate_after,
            train_steps_after,
        ) = evaluate(
            model,
            train_graph_objects,
        )

        (
            test_rate_after,
            test_steps_after,
        ) = evaluate(
            model,
            test_graph_objects,
        )

        print()
        print("TRAIN GRAPHS")
        print("-" * 78)

        print(
            f"Validity rate : "
            f"{train_rate_before:.3f}"
            f" -> "
            f"{train_rate_after:.3f}"
        )

        print(
            f"Mean steps    : "
            f"{train_steps_before:.3f}"
            f" -> "
            f"{train_steps_after:.3f}"
        )

        print()
        print("HELD-OUT GRAPHS")
        print("-" * 78)

        print(
            f"Validity rate : "
            f"{test_rate_before:.3f}"
            f" -> "
            f"{test_rate_after:.3f}"
        )

        print(
            f"Mean steps    : "
            f"{test_steps_before:.3f}"
            f" -> "
            f"{test_steps_after:.3f}"
        )

        print()
        print("=" * 78)
        print("INTERPRETATION")
        print("=" * 78)

        train_delta = (
            train_rate_after
            - train_rate_before
        )

        test_delta = (
            test_rate_after
            - test_rate_before
        )

        print(
            f"Train validity change : "
            f"{train_delta:+.3f}"
        )

        print(
            f"Test validity change  : "
            f"{test_delta:+.3f}"
        )

        print()

        if (
            train_rate_after > train_rate_before
            and test_rate_after > test_rate_before
        ):

            print(
                "RESULT: validity performance "
                "improved on both training and "
                "held-out graphs."
            )

        elif train_rate_after > train_rate_before:

            print(
                "RESULT: training performance "
                "improved, but held-out "
                "generalization did not improve."
            )

        else:

            print(
                "RESULT: no clear validity "
                "improvement was observed."
            )

        print()
        print(
            "FULL OUTPUT SAVED TO:"
        )

        print(
            OUTPUT_FILE
        )

    except Exception as exc:

        print()
        print("=" * 78)
        print("EXPERIMENT FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    finally:

        sys.stdout = original_stdout
        tee.close()


# ============================================================
# PYTEST ENTRY POINT
# ============================================================

def test_validity_policy_can_learn_from_real_graphs():

    run_experiment()

    assert OUTPUT_FILE.exists()

    assert OUTPUT_FILE.stat().st_size > 0


if __name__ == "__main__":
    run_experiment()