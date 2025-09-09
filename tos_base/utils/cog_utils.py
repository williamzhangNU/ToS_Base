"""
Cognitive Map Utility Functions

This module provides utility functions for evaluating cognitive maps using turn logs
and LLM interfaces.
"""

import numpy as np
from typing import List, Dict, Any
from verl import DataProto
from ..managers.cognitive_map_manager import CognitiveMapManager, COGMAP_INSTRUCTION_SHORTER
from .. import Room, Agent
import re


def evaluate_cognitive_maps_from_turnlogs(
    env_summarys: List[Dict[str, Any]],  # List of env summaries from get_env_summary()
    message_lists: List[List[Dict[str, str]]],  # Corresponding list of message histories
    llm_wrapper,
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
    cogmap_config = {
        "cogmap_type": "standard",
        "pos_allow_scale": True,
        "scope": "all"
    }
    
    # Collect all message sequences and metadata
    all_messages_list = []
    all_env_ids = []
    env_id_to_location = {}  # Mapping from env_id to (env_idx, turn_idx)
    env_cogmap_managers = {}  # Store CognitiveMapManager for each env
    
    env_id_counter = 0
    for env_idx, env_summary in enumerate(env_summarys):
        env_config = env_summary['env_info']['config']
        # Create CognitiveMapManager
        cognitive_map_manager = CognitiveMapManager(
            **cogmap_config
        )
        env_cogmap_managers[env_idx] = cognitive_map_manager
        
        turn_logs = env_summary['env_turn_logs']
        
        # Use messages from message_lists as base and add cogmap prompts
        if env_idx < len(message_lists):
            env_messages_raw = message_lists[env_idx] if message_lists[env_idx] else []
            # Filter out system messages at the beginning to ensure correct turn indexing
            env_messages = [msg for msg in env_messages_raw if msg.get('role') != 'system']
            
            if env_config.get('exp_type') == 'active':
                for turn_idx, turn_log in enumerate(turn_logs):
                    if turn_idx==0 or not turn_logs[turn_idx-1].get('info', {}).get('is_valid_action', False):
                        continue
                    
                    # Build cumulative message history up to this turn from message_lists
                    # Each turn has 2 messages (user + assistant), so turn_idx * 2 gives us the message index
                    messages = []
                    
                    # Include all messages up to and including current turn
                    max_message_idx = min((turn_idx+1) * 2, len(env_messages))
                    for msg_idx in range(max_message_idx):
                        msg = env_messages[msg_idx]
                        messages.append(msg.copy())
                    
                    # Add cogmap prompt to the appropriate message
                    if turn_logs[turn_idx-1].get('is_exploration_phase', False):
                        # Modify the second-to-last user message to include cogmap instruction
                        if len(messages) >= 2:
                            assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"
                            messages[-2]["content"] = turn_log.get('user_message', '') + COGMAP_INSTRUCTION_SHORTER
                    elif not turn_log.get('is_exploration_phase', False) and turn_log.get('evaluation_log') and turn_log.get('evaluation_log', {}).get('evaluation_data', {}).get('action'):
                        if len(messages) >= 2:
                            assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"
                            messages[-2]["content"] = re.sub(r'## Evaluation Question.*', '', turn_log.get('user_message', ''), flags=re.DOTALL) \
                                + turn_log['evaluation_log']['evaluation_data']['action'] + COGMAP_INSTRUCTION_SHORTER
                    else:
                        continue
                    
                    # Remove the last assistant message since we're asking for cognitive map
                    if messages and messages[-1]["role"] == "assistant":
                        messages.pop()
                    
                    # Add to batch
                    all_messages_list.append(messages)
                    all_env_ids.append(env_id_counter)
                    # very important: map env_id to (env_idx, turn_idx-1) since cogmap is for previous turn
                    env_id_to_location[env_id_counter] = (env_idx, turn_idx - 1)
                    env_id_counter += 1

            elif env_config.get('exp_type') == 'passive':
                assert env_messages[0]["role"] == "user", f"Expected user message but got {env_messages[0]['role']}"
                messages = [env_messages[0].copy()]
                messages[0]["content"] = re.sub(r'## Evaluation Question.*', '', turn_logs[0].get('user_message', ''), flags=re.DOTALL) + COGMAP_INSTRUCTION_SHORTER
                
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
            
            # Store the cognitive map response in the turn log dictionary
            turn_log['cognitive_map_response'] = response
            cogmap_log = None
            if response and turn_log.get('room_state') and turn_log.get('agent_state'):
                # Reconstruct Room and Agent states from turn log
                room_state = Room.from_dict(turn_log['room_state'])
                agent_state = Agent.from_dict(turn_log['agent_state'])
                
                # Get observed items
                observed_items = turn_log['observed_items']
                if env_summary['env_info']['config'].get('exp_type') == 'active':
                    cogmap_log = cognitive_map_manager.evaluate_cognitive_map(
                        response,
                        room_state,
                        agent_state,
                        observed_items
                    )
                cogmap_full_log = cognitive_map_manager.evaluate_cognitive_map(
                    response,
                    room_state,
                    agent_state,
                    [obj.name for obj in room_state.all_objects]
                )

            
            turn_log['cogmap_full_log'] = cogmap_full_log.to_dict() 
            turn_log['cogmap_log'] = cogmap_log.to_dict() if cogmap_log else None
    
    # After processing all cognitive maps, generate cogmap_summary for each environment
    for env_idx, cognitive_map_manager in env_cogmap_managers.items():
        env_summary = env_summarys[env_idx]
        
        # Generate cogmap summary using the manager
        cogmap_summary = cognitive_map_manager.get_cogmap_summary()
     
        env_summary['summary']['cogmap_summary'] = cogmap_summary
        
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

