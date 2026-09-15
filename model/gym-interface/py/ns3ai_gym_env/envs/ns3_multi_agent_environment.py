from typing import Any, Literal, TypeVar

import messages_pb2 as pb
import ns3ai_gym_msg_py as py_binding
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ns3ai_gym_env.typing import copy_signature_from
from torch import Use

from .ns3_environment import Ns3Env

T = TypeVar("T")


class Ns3MultiAgentEnv(Ns3Env, MultiAgentEnv):
    @copy_signature_from(Ns3Env.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract action masking flag before it reaches ns-3 cmd args
        ns3_settings = kwargs.get("ns3Settings", {})
        self.useActionMasking = ns3_settings.pop("useActionMasking", "false") == "true"

        self.action_spaces: spaces.Dict = spaces.Dict()
        self.observation_spaces: spaces.Dict = spaces.Dict()
        self.agent_selection: str | None = None
        self._last_obs: dict[str, Any] = {}
        self._inner_obs_spaces: dict[str, spaces.Dict] = {}
        super().__init__(*args, **kwargs)
        MultiAgentEnv.__init__(self)

    def initialize_env(self) -> Literal[True]:
        init_msg = pb.MultiAgentSimInitMsg()

        # Check ns-3 alive before blocking
        if self.exp.proc and not self.exp.isalive():
            raise RuntimeError(f"ns-3 process died before initialize_env PyRecvBegin")

        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        init_msg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        for agent, space in init_msg.actSpaces.items():
            self.action_spaces[agent] = self._create_space(space)

        for agent, space in init_msg.obsSpaces.items():
            self.observation_spaces[agent] = self._create_space(space)
        self._agent_ids = list(self.action_spaces.keys())

        # Conditionally restructure observation spaces for action masking
        if self.useActionMasking:
            self._inner_obs_spaces = {}
            for agent_id in self._agent_ids:
                full_space = self.observation_spaces[agent_id]
                # Skip action masking if space is not a Dict with action_mask
                # (e.g., Top-N designs use flat Box spaces)
                if not hasattr(full_space, "spaces") or "action_mask" not in full_space.spaces:
                    continue
                mask_space = full_space["action_mask"]
                inner_keys = [k for k in full_space.spaces if k != "action_mask"]
                inner_dict = spaces.Dict({k: full_space[k] for k in inner_keys})
                flat_inner = spaces.flatten_space(inner_dict)
                self._inner_obs_spaces[agent_id] = inner_dict
                self.observation_spaces[agent_id] = spaces.Dict({
                    "observations": flat_inner,
                    "action_mask": mask_space,
                })
        else:
            # Strip action_mask from space when not using action masking.
            # Only applies to Dict spaces; flat Box spaces (Top-N) are left as-is.
            for agent_id in self._agent_ids:
                full_space = self.observation_spaces[agent_id]
                if hasattr(full_space, "spaces") and "action_mask" in full_space.spaces:
                    inner_keys = [k for k in full_space.spaces if k != "action_mask"]
                    self.observation_spaces[agent_id] = spaces.Dict(
                        {k: full_space[k] for k in inner_keys}
                    )

        reply = pb.SimInitAck()
        reply.done = True
        reply.stopSimReq = False
        reply_str = reply.SerializeToString()
        assert len(reply_str) <= py_binding.msg_buffer_size

        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_str)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(reply_str)] = reply_str
        self.msgInterface.PySendEnd()
        return True

    def rx_env_state(self) -> None:
        if self.newStateRx:
            return

        # Check ns-3 is alive before blocking on semaphore
        if self.exp.proc and not self.exp.isalive():
            print(f"[MA-RX] FATAL — ns-3 process {self.exp.proc.pid} dead before PyRecvBegin", flush=True)
            raise RuntimeError(
                f"ns-3 process {self.exp.proc.pid} died before rx_env_state"
            )

        state_msg = pb.MultiAgentEnvStateMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        state_msg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.obsData = self._create_data(state_msg.obsData)
        self.reward = state_msg.reward
        self.gameOver = state_msg.isGameOver
        self.gameOverReason = state_msg.reason
        self.agent_selection = state_msg.agentID

        if self.gameOver:
            print(f"[MA-RX] gameOver=True reason={state_msg.reason} agent={self.agent_selection} — sending close cmd", flush=True)
            self.send_close_command()
        else:
            pass  # normal step

        self.extraInfo = dict(state_msg.info)

        self.newStateRx = True

    def send_actions(self, actions: dict[str, Any]) -> bool:
        assert self.agent_selection
        reply = pb.EnvActMsg()

        # Check ns-3 alive before blocking
        if self.exp.proc and not self.exp.isalive():
            raise RuntimeError(f"ns-3 process died before send_actions PySendBegin")

        action_msg = self._pack_data(actions[self.agent_selection], self.action_spaces[self.agent_selection])
        reply.actData.CopyFrom(action_msg)

        reply_msg = reply.SerializeToString()
        assert len(reply_msg) <= py_binding.msg_buffer_size
        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_msg)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(reply_msg)] = reply_msg
        self.msgInterface.PySendEnd()
        self.newStateRx = False
        return True

    def _restructure_obs(self, raw_obs, agent_id):
        """Restructure raw observation dict into action-masking format.

        Input: dict with obs keys + "action_mask" (from C++)
        Output: {"observations": flat_array, "action_mask": np.int8_array}
        """
        mask = np.round(raw_obs.pop("action_mask")).astype(np.int8)
        inner_space = self._inner_obs_spaces[agent_id]
        flat_obs = spaces.flatten(inner_space, raw_obs)
        return {"observations": flat_obs, "action_mask": mask}

    def wrap(self, data: T) -> dict[str, T]:
        assert self.agent_selection is not None
        return {self.agent_selection: data}

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        acting_agent = self.agent_selection
        if acting_agent is None:
            print(f"[MA-STEP] FATAL — agent_selection is None, cannot step", flush=True)
        assert acting_agent is not None

        # Call parent step – returns raw (obs, reward, terminated, truncated, info)
        super_step_start = __import__("time").monotonic()
        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = super().step(actions)
        super_step_dur = __import__("time").monotonic() - super_step_start

        # Restructure raw_obs for action masking (C++ sends flat dict with action_mask)
        if self.useActionMasking and isinstance(raw_obs, dict) and "action_mask" in raw_obs:
            raw_obs = self._restructure_obs(raw_obs, acting_agent)
        elif isinstance(raw_obs, dict):
            raw_obs.pop("action_mask", None)

        # If agent is done and observation is None (simulation ended), fall back to last known obs
        if raw_term or raw_trunc:
            if raw_obs is None:
                raw_obs = self._last_obs.get(acting_agent)
                print(f"[MA-STEP] episode end — raw_obs was None, using last_obs", flush=True)
            else:
                self._last_obs[acting_agent] = raw_obs
            print(f"[MA-STEP] done agent={acting_agent} term={raw_term} trunc={raw_trunc} step_dur={super_step_dur:.3f}s", flush=True)
        else:
            self._last_obs[acting_agent] = raw_obs

        # Wrap with agent key
        obs = {acting_agent: raw_obs}
        rew = {acting_agent: raw_rew}
        terminateds = {acting_agent: raw_term}
        truncateds = {acting_agent: raw_trunc}
        info = {acting_agent: raw_info}

        # RLlib sentinel for "all agents done"
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return obs, rew, terminateds, truncateds, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        import time as _time
        t0 = _time.monotonic()
        print(f"[MA-RESET] enter — agent_selection={self.agent_selection} _last_obs_keys={list(self._last_obs.keys())}", flush=True)

        # Call parent's reset – returns (raw_obs, raw_info) for the first agent
        try:
            raw_obs, raw_info = super().reset(seed=seed, options=options)
        except Exception as e:
            print(f"[MA-RESET] FATAL — super().reset() failed: {e}", flush=True)
            raise

        # After super().reset(), self.agent_selection is set to the first agent
        # from the multi-agent init message
        first_agent = self.agent_selection
        print(f"[MA-RESET] after super().reset() — agent_selection={first_agent} in {_time.monotonic()-t0:.3f}s", flush=True)
        if first_agent is None:
            print(f"[MA-RESET] FATAL — agent_selection is None after reset!", flush=True)
        assert first_agent is not None

        # Restructure raw_obs for action masking
        if self.useActionMasking and isinstance(raw_obs, dict) and "action_mask" in raw_obs:
            raw_obs = self._restructure_obs(raw_obs, first_agent)
        elif isinstance(raw_obs, dict):
            raw_obs.pop("action_mask", None)

        # Wrap the single observation with the first agent's ID
        obs_dict = {first_agent: raw_obs}
        info_dict = {first_agent: raw_info}

        # Store initial observation for step fallback
        self._last_obs = {first_agent: raw_obs}

        print(f"[MA-RESET] done — returning obs for agent={first_agent} in {_time.monotonic()-t0:.3f}s", flush=True)
        return obs_dict, info_dict

    def get_random_action(self) -> Any:
        assert self.agent_selection is not None
        return self.action_spaces[self.agent_selection].sample()
