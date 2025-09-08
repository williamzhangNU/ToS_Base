"""
Cognitive Map Utility Functions

This module provides utility functions for evaluating cognitive maps using turn logs
and LLM interfaces.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from verl import DataProto
from ragen.env.spatial.env import SpatialGym
from ..managers.cognitive_map_manager import CognitiveMapManager, COGMAP_INSTRUCTION_SHORTER
from ragen.llm_agent.agent_proxy import ApiCallingWrapperWg
import re


def evaluate_cognitive_maps_from_turnlogs(
    envs: List[SpatialGym],  # List of environment objects with .turn_logs attribute
    llm_wrapper: ApiCallingWrapperWg,
) -> None:
    """
    Evaluate cognitive maps for each turn in multiple environments' turn logs.
    
    This function processes all environments' turn logs, constructs message histories
    with COGMAP_EXP_REQUIRED_INSTRUCTION, makes batch LLM calls for all inputs,
    evaluates cognitive maps using CognitiveMapManager, and stores results back
    into the corresponding env turn logs.
    
    Args:
        envs: List of environment objects, each with .turn_logs attribute
        llm_wrapper: ApiCallingWrapperWg instance for LLM API calls
        cogmap_config: Optional configuration dict for CognitiveMapManager
    """

    # Collect all message sequences and metadata
    all_messages_list = []
    all_env_ids = []
    env_id_to_location = {}  # Mapping from env_id to (env_idx, turn_idx)
    
    env_id_counter = 0
    for env_idx, env in enumerate(envs):
        if not env.config.prompt_config.get("cogmap", False):
            continue
        turn_logs = env.turn_logs
        if env.config.exp_type == 'active':
            for turn_idx, turn_log in enumerate(turn_logs):
                # Build cumulative message history up to this turn
                messages = []
                if turn_idx==0 or not turn_logs[turn_idx-1].info['is_valid_action']:
                    continue
                # Add all previous turns to build context
                for prev_turn in turn_logs[:turn_idx + 1]:
                    if prev_turn.user_message:
                        messages.append({
                            "role": "user", 
                            "content": prev_turn.user_message
                        })
                    if prev_turn.assistant_raw_message:
                        messages.append({
                            "role": "assistant",
                            "content": prev_turn.assistant_raw_message
                        })
                
                # include Term
                if turn_logs[turn_idx-1].is_exploration_phase:
                    messages[-2] = {
                        "role": "user",
                        "content": turn_log.user_message + COGMAP_INSTRUCTION_SHORTER
                    }
                elif not turn_log.is_exploration_phase and turn_log.evaluation_log and turn_log.evaluation_log.evaluation_data.action:
                    messages[-2] = {
                        "role": "user",
                        "content": re.sub(r'## Evaluation Question.*', '', turn_log.user_message, flags=re.DOTALL) 
                            + turn_log.evaluation_log.evaluation_data.action + COGMAP_INSTRUCTION_SHORTER
                    }
                else:
                    continue
                # Remove the last assistant message since we're asking for cognitive map
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                
                # Add to batch
                all_messages_list.append(messages)
                all_env_ids.append(env_id_counter)
                env_id_to_location[env_id_counter] = (env_idx, turn_idx)
                env_id_counter += 1

        elif env.config.exp_type == 'passive':
            messages = [
                {
                    "role": "user", 
                    "content":  re.sub(r'## Evaluation Question.*', '', turn_logs[0].user_message, flags=re.DOTALL) + COGMAP_INSTRUCTION_SHORTER        
                }
            ]
            all_messages_list.append(messages)
            all_env_ids.append(env_id_counter)
            env_id_to_location[env_id_counter] = (env_idx, 0)
            env_id_counter += 1

    
    if all_messages_list:
        lm_outputs = _call_llm_batch(
            llm_wrapper, all_messages_list, all_env_ids, all_env_ids
        )

    for (response, original_env_id) in zip(lm_outputs.non_tensor_batch['response_texts'], lm_outputs.non_tensor_batch['env_ids']):
        env_idx, turn_idx = env_id_to_location[original_env_id]
        env = envs[env_idx]
        cognitive_map_manager=env.cognitive_map_manager
        assert cognitive_map_manager is not None, "CognitiveMapManager is not initialized in the environment."
        turn_log = env.turn_logs[turn_idx]
        
        # Evaluate the cognitive map using the manager
        cogmap_log = None
        if response and turn_log.room_state and turn_log.agent_state:
            cogmap_log = cognitive_map_manager.evaluate_cognitive_map(
                response,
                turn_log.room_state,
                turn_log.agent_state,
                turn_log.observed_items if turn_log.is_exploration_phase else [o.name for o in turn_log.room_state.all_objects]
            )
        
        # include Term
        if not turn_log.is_exploration_phase:
            turn_log.cogmap_final_log = cogmap_log
        else:
            turn_log.cogmap_log = cogmap_log

def _call_llm_batch(
    llm_wrapper: ApiCallingWrapperWg, 
    all_messages_list: List[List[Dict[str, str]]], 
    all_env_ids: List[int], 
    all_group_ids: List[int]
) -> List[str]:
    """
    Helper function to make batch LLM calls using ApiCallingWrapperWg.
    """
    # Create DataProto object for batch processing
    lm_inputs = DataProto()
    lm_inputs.non_tensor_batch = {
        'messages_list': np.array(all_messages_list, dtype=object),
        'env_ids': np.array(all_env_ids, dtype=object),
        'group_ids': np.array(all_group_ids, dtype=object)
    }
    
    # Call generate_sequences for batch processing
    lm_outputs = llm_wrapper.generate_sequences(lm_inputs)
    
    return lm_outputs

