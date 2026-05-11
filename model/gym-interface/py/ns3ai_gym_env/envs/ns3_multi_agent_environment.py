from typing import Any, Literal, TypeVar

import messages_pb2 as pb
import ns3ai_gym_msg_py as py_binding
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ns3ai_gym_env.typing import copy_signature_from
from torch import Use

from .ns3_environment import Ns3Env

T = TypeVar("T")


class Ns3MultiAgentEnv(Ns3Env, MultiAgentEnv):
    @copy_signature_from(Ns3Env.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.action_spaces: spaces.Dict = spaces.Dict()
        self.observation_spaces: spaces.Dict = spaces.Dict()
        self.agent_selection: str | None = None
        self._last_obs: dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        MultiAgentEnv.__init__(self)

    def initialize_env(self) -> Literal[True]:
        init_msg = pb.MultiAgentSimInitMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        init_msg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        for agent, space in init_msg.actSpaces.items():
            self.action_spaces[agent] = self._create_space(space)

        for agent, space in init_msg.obsSpaces.items():
            self.observation_spaces[agent] = self._create_space(space)
        self._agent_ids = list(self.action_spaces.keys())
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
            self.send_close_command()

        self.extraInfo = dict(state_msg.info)

        self.newStateRx = True

    def send_actions(self, actions: dict[str, Any]) -> bool:
        assert self.agent_selection
        reply = pb.EnvActMsg()

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

    def wrap(self, data: T) -> dict[str, T]:
        assert self.agent_selection is not None
        return {self.agent_selection: data}

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        acting_agent = self.agent_selection
        assert acting_agent is not None

        # Call parent step – returns raw (obs, reward, terminated, truncated, info)
        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = super().step(actions)

        # If agent is done and observation is None (simulation ended), fall back to last known obs
        if raw_term or raw_trunc:
            if raw_obs is None:
                raw_obs = self._last_obs.get(acting_agent)
            else:
                self._last_obs[acting_agent] = raw_obs
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
        # Call parent's reset – returns (raw_obs, raw_info) for the first agent
        raw_obs, raw_info = super().reset(seed=seed, options=options)

        # After super().reset(), self.agent_selection is set to the first agent
        first_agent = self.agent_selection
        assert first_agent is not None

        # Wrap the single observation with the first agent's ID
        obs_dict = {first_agent: raw_obs}
        info_dict = {first_agent: raw_info}

        # Store initial observation for step fallback
        self._last_obs = {first_agent: raw_obs}

        return obs_dict, info_dict

    def get_random_action(self) -> Any:
        assert self.agent_selection is not None
        return self.action_spaces[self.agent_selection].sample()
