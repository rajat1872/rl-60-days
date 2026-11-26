# Day 06: The Gymnasium API, and a custom text environment (`ConfigFix-v0`)

## The idea
Every RL library (and every agent-training framework since) talks to environments through the same small contract:

```python
observation, info = environment.reset(seed=42)
observation, reward, terminated, truncated, info = environment.step(action)
```

| piece | meaning | the trap |
|---|---|---|
| `reset(seed=...)` | start an episode; the seed must fix **all** randomness | use `self.np_random`, never `random` or `np.random`, or the same seed gives different episodes |
| `terminated` | the episode ended **inside** the task (pole fell, answer submitted) | don't bootstrap V(s') after it |
| `truncated` | the episode was **cut off** from outside (time limit) | you *should* bootstrap; the state wasn't terminal |
| `info` | side channel for debugging | never put the answer in it if an agent can read it |
| spaces | `action_space` / `observation_space` declare what's legal | `spaces.Text` defaults to letters and digits only |

**`ConfigFix-v0`** is a tiny version of the environments the rest of this plan builds. A service config has 2
injected bugs, and the agent repairs it with text commands, a shape that plain RL algorithms *and* LLM agents can both
use:

```
 reset(seed) ──► config with 2 broken fields (chosen by self.np_random)
                         │
 agent text ──► "check"                 → lists failing fields (a tool call)
                "set <field> <value>"   → edits one field
                "submit"                → verifier checks EVERY field → reward 1.0 or 0.0, terminated
                anything else           → error message, never a crash (trust boundary)
 TimeLimit wrapper ──► truncated after 10 steps
```

## Run
    python day06_gymnasium_custom_env/test_and_agents.py

This runs Gymnasium's own `check_env`, a determinism test, and a partial-fix test, then pits an oracle agent against
a random agent over 200 seeds.

## Output (real run)
```
gymnasium check_env: PASS
determinism test: PASS
partial-fix test: PASS

=== oracle agent, one traced episode ===
[reset seed=42] hidden info: {'broken_fields': ['port', 'replicas']}
port = 99999 … replicas = 50 …
step 1: action='check'          -> 'failing: port, replicas'  reward=0.0 terminated=False
step 2: action='set port 8080'  -> 'ok: port = 8080'          reward=0.0 terminated=False
step 3: action='check'          -> 'failing: replicas'        reward=0.0 terminated=False
step 4: action='set replicas 3' -> 'ok: replicas = 3'         reward=0.0 terminated=False
step 5: action='check'          -> 'all checks pass'          reward=0.0 terminated=False
step 6: action='submit'         -> 'submitted: PASS'          reward=1.0 terminated=True
oracle agent over 200 seeds: success 100.0%, average steps 6.0
random agent over 200 seeds: success 0.0%, average steps 4.6
```
- The oracle solves every seed in exactly 6 steps (2 bugs × check + fix, plus a final check and the submit).
- The random agent never succeeds, because the reward is **sparse**: 1.0 only on a fully correct submit. That's why
  later weeks use LLM policies, which start from a good prior instead of random text.

## Dry run: `ConfigFixEnv.step` for two actions of the traced episode
**State before step 2:** `config = {port: "99999", log_level: "INFO", timeout_seconds: "30", replicas: "50", tls_enabled: "true"}`.

**`step("set port 8080")`:**
1. `words = ["set", "port", "8080"]`, and `reward, terminated = 0.0, False`.
2. `words == ["check"]`? No.
3. Three words, the first is `set`, and `port` is a key in `self.config`, so `self.config["port"] = "8080"` and
   `tool_output = "ok: port = 8080"`.
4. It returns `(render(config) + "---\nok: port = 8080", 0.0, False, False, {})`. Editing earns nothing; only the
   final result counts.

**`step("submit")`, after replicas is also fixed:**
1. `words == ["submit"]`, so it calls `run_verifier(self.config)`.
2. `run_verifier` checks **every** field, not just the ones that were broken:
   - `port`: "8080".isdigit() and 1024 ≤ 8080 ≤ 65535, so it passes;
   - `log_level`: "INFO" is in the allowed set;
   - `timeout_seconds`: 30 is in [1, 300];
   - `replicas`: 3 is in [1, 10];
   - `tls_enabled`: "true".

   The failing list is `[]`.
3. `reward = 1.0`, `terminated = True`, and `tool_output = "submitted: PASS"`.

Checking every field is the design choice that matters. Checking only the 2 injected fields would let an agent
"fix one, break another" and still score 1.0. `test_reward_cannot_be_gamed_by_partial_fix` makes sure that's
closed.

## Read
- Gymnasium docs: "Basic Usage" and "Make your own custom environment" (gymnasium.farama.org).
- The `terminated`/`truncated` split: Pardo et al. 2018, *Time Limits in Reinforcement Learning* (arXiv:1712.00378).
