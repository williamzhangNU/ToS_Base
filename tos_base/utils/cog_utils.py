"""
Cognitive Map Utility Functions

This module provides utility functions for evaluating cognitive maps using turn logs
and LLM interfaces.
"""

import numpy as np
from typing import List, Dict, Any
from verl import DataProto
from ..managers.cognitive_map_manager import CognitiveMapManager, COGMAP_INSTRUCTION_SHORTER
from ..managers.history_manager import HistoryManager
from .. import Room, Agent
import re


def _evaluate_and_store_cogmap_logs(turn_log, response, cognitive_map_manager, env_config):
    """Helper function to evaluate cognitive map response and store logs in turn_log"""
    cogmap_log = None
    cogmap_full_log = None
    if env_config.get('exp_type') == 'passive' or turn_log.get('is_exploration_phase'):
        mode = "explore"
    else:
        mode = "evaluate"
    

def evaluate_cognitive_maps_from_turnlogs(
    env_summarys: List[Dict[str, Any]],  # List of env summaries from get_env_summary()
    message_lists: List[List[Dict[str, str]]],  # Corresponding list of message histories
    llm_wrapper,
    override_cogmap: bool = False,
    cogmap_config: Dict[str, Any] = None,
    vagen: bool = False,  # Whether to use _generate_batch_responses interface
) -> None:
    """
    Evaluate cognitive maps for each turn in multiple environments' turn logs.
    
    This function processes all environments' turn logs, constructs message histories
    with COGMAP_EXP_REQUIRED_INSTRUCTION, makes batch LLM calls for all inputs,
    evaluates cognitive maps using CognitiveMapManager, and stores results back
    into the corresponding env turn logs.
    
    Args:
        env_summarys: List of environment summary dicts from get_env_summary(), 
                     each containing 'env_turn_logs' and 'env_info' keys
        message_lists: List of message histories for each environment
        llm_wrapper: ApiCallingWrapperWg instance for LLM API calls
        use_new_interface: If True, use _generate_batch_responses interface;
                          if False, use _call_llm_batch interface
    """
    assert len(env_summarys) == len(message_lists), "env_summarys and message_lists must have the same length"
    if not cogmap_config:
        cogmap_config = {
            "cogmap_type": "standard",
            "pos_allow_scale": True,
            "scope": "all",
        }
    
    # Collect all message sequences and metadata
    all_messages_list = []
    all_env_ids = []
    env_id_to_location = {}  # Mapping from env_id to (env_idx, turn_idx)
    env_cogmap_managers = {}  # Store CognitiveMapManager for each env
    env_history_managers = {}  # Store HistoryManager for each env
    
    env_id_counter = 0
    for env_idx, env_summary in enumerate(env_summarys):
        env_info = env_summary['env_info']
        env_config = env_info['config']
        room_config = env_info['initial_room']
        agent_config = env_info['initial_agent']
        # Create CognitiveMapManager
        cognitive_map_manager = CognitiveMapManager(
            **cogmap_config
        )
        env_cogmap_managers[env_idx] = cognitive_map_manager

        history_manager = HistoryManager(env_config['observation_config'], room_config, agent_config)
        env_history_managers[env_idx] = history_manager
        
        turn_logs = env_summary['env_turn_logs']
        env_messages = [msg for msg in message_lists[env_idx] if msg.get('role') != 'system']
        
        if env_config.get('exp_type') == 'active':
            for turn_idx, turn_log in enumerate(turn_logs):
                if turn_idx==0 or not turn_logs[turn_idx-1].get('info', {}).get('is_valid_action', False):
                    continue
                messages = []
                # Include all messages up to and including current turn
                max_message_idx = min((turn_idx+1) * 2, len(env_messages))
                for msg_idx in range(max_message_idx):
                    msg = env_messages[msg_idx]
                    messages.append(msg.copy())
                # Check if cogmap response already exists in history
                if turn_log['is_exploration_phase']:
                    target_turn_idx = turn_idx - 1
                    if history_manager.has_cogmap_response(target_turn_idx) and not override_cogmap:
                        continue
                    assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"
                    messages[-2]["content"] = turn_log.get('user_message', '') + COGMAP_INSTRUCTION_SHORTER
                # like false belief task
                elif not turn_log['is_exploration_phase'] and turn_log.get('evaluation_log') and turn_log.get('evaluation_log', {}).get('evaluation_data', {}).get('action'):
                    target_turn_idx = turn_idx
                    assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"
                    messages[-2]["content"] = re.sub(r'## Evaluation Question.*', '', turn_log.get('user_message', ''), flags=re.DOTALL) \
                        + turn_log['evaluation_log']['evaluation_data']['action'] + COGMAP_INSTRUCTION_SHORTER
                else:
                    continue
             
                messages.pop()
                
                # Add to batch
                env_id_to_location[env_id_counter] = (env_idx, target_turn_idx)
                all_messages_list.append(messages)
                all_env_ids.append(env_id_counter)
                env_id_counter += 1

        elif env_config.get('exp_type') == 'passive':
            # Check if cogmap response already exists in history
            if history_manager.has_cogmap_response(0) and not override_cogmap:
                continue

            assert env_messages[0]["role"] == "user", f"Expected user message but got {env_messages[0]['role']}"
            messages = [env_messages[0].copy()]
            messages[0]["content"] = re.sub(r'## Evaluation Question.*', '', turn_logs[0].get('user_message', ''), flags=re.DOTALL) + COGMAP_INSTRUCTION_SHORTER

            turn_log = {'turn_number': 1 ,"user_message": messages[0]["content"], "is_exploration_phase": True, "room_state": room_config, "agent_state": agent_config}
            turn_logs[0] = turn_log
            if not history_manager.is_history_exist():
                history_manager.update_turn_log(turn_log)

            all_messages_list.append(messages)
            all_env_ids.append(env_id_counter)
            env_id_to_location[env_id_counter] = (env_idx, 0)
            env_id_counter += 1

    
    if all_messages_list:
        response_texts = _call_llm_batch(
            llm_wrapper, all_messages_list, all_env_ids, vagen
        )

        for response, original_env_id in zip(response_texts, all_env_ids):
            env_idx, turn_idx = env_id_to_location[original_env_id]
            env_summary = env_summarys[env_idx]
            turn_log = env_summary['env_turn_logs'][turn_idx]
            cognitive_map_manager = env_cogmap_managers[env_idx]
            history_manager = env_history_managers[env_idx]
            
            # Store the cognitive map response in the turn log dictionary
            turn_log['cogmap_response'] = response
            cogmap_log, cogmap_full_log = None, None
            if response and turn_log.get('room_state') and turn_log.get('agent_state'):
                # Reconstruct Room and Agent states from turn log
                room_state = Room.from_dict(turn_log['room_state'])
                agent_state = Agent.from_dict(turn_log['agent_state'])
                
                # Get observed items
                if turn_log.get('exploration_log'):
                    cogmap_log = cognitive_map_manager.evaluate_cognitive_map(
                        response,
                        room_state,
                        agent_state,
                        turn_log['observed_items']
                    )
                cogmap_full_log = cognitive_map_manager.evaluate_cognitive_map(
                    response,
                    room_state,
                    agent_state,
                    [obj.name for obj in room_state.all_objects]
                )

            turn_log['cogmap_full_log'] = cogmap_full_log.to_dict() if cogmap_full_log else None
            turn_log['cogmap_log'] = cogmap_log.to_dict() if cogmap_log else None
            history_manager.update_cogmap(turn_log)
    
    # Save all history managers to persist cached cognitive map responses
    for env_idx, history_manager in env_history_managers.items():
        history_manager.save()  # Save cognitive map responses separately
        
    return env_summarys

def _call_llm_batch(
    llm_wrapper, 
    all_messages_list: List[List[Dict[str, str]]], 
    all_env_ids: List[int],
    vagen: bool = False
) -> List[str]:
    """
    Helper function to make batch LLM calls using ApiCallingWrapperWg.
    
    Args:
        llm_wrapper: The LLM wrapper instance
        all_messages_list: List of message lists for each request
        all_env_ids: List of environment IDs (used for ordering)
        vagen: If True, use generate interface

    Returns:
        List of response texts in the same order as input
    """
    if vagen:
        # Use _generate_batch_responses interface
        responses = llm_wrapper.generate(all_messages_list)
        
        # Return responses in the same order as input
        return [response['text'] for response in responses]
    else:
        # Use original generate_sequences interface
        lm_inputs = DataProto()
        lm_inputs.non_tensor_batch = {
            'messages_list': np.array(all_messages_list, dtype=object),
            'env_ids': np.array(all_env_ids, dtype=object),
            'group_ids': np.array(all_env_ids, dtype=object)  # Use env_ids as group_ids
        }
        
        # Call generate_sequences for batch processing
        lm_outputs = llm_wrapper.generate_sequences(lm_inputs)
        
        return lm_outputs.non_tensor_batch['response_texts']

