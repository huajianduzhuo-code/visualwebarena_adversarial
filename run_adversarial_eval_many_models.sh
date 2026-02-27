# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_base_vs_gemma27b_100tasks


# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/imitation_old_prompt/gemma-3-12b-it-both_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_imitation_vs_gemma27b_100tasks


# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/enhanced_sft_task_focused/gemma-3-12b-it-both_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_enhanced_sft_task_focused_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_iter_10_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_15_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_iter_15_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_20_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_iter_20_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_10_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_15_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_15_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_20_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_20_vs_gemma27b_100tasks

# # against qwen model

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_base_vs_qwen8b_100tasks


# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/imitation_old_prompt/gemma-3-12b-it-both_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_imitation_vs_qwen8b_100tasks


# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/enhanced_sft_task_focused/gemma-3-12b-it-both_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_enhanced_sft_task_focused_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_15_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_iter_15_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_20_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_iter_20_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_15_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_15_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_20_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_no_enhanced_sft_iter_20_vs_qwen8b_100tasks


### baseline methods

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/spag_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_spag_iter_10_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_art_iter_10_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/online_sft_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --result-dir cache/adversarial_gemma12b_online_sft_iter_10_vs_gemma27b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/imitation_old_prompt/gemma-3-12b-it-both_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_imitation_prompt_defense_level1_vs_gemma27b_100tasks


# ---------- no attacker results ----------

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --workers 8 --no-attacker --result-dir cache/clean_gemma12b_base_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --workers 8 --no-attacker --result-dir cache/clean_gemma12b_iter_10_100tasks_run2

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/spag_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --workers 8 --no-attacker --result-dir cache/clean_gemma12b_spag_iter_10_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --workers 8 --no-attacker --result-dir cache/clean_gemma12b_art_iter_10_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/online_sft_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --workers 8 --no-attacker --result-dir cache/clean_gemma12b_online_sft_iter_10_100tasks_run2

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/imitation_old_prompt/gemma-3-12b-it-both_merged --workers 8 --no-attacker --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/clean_gemma12b_imitation_prompt_defense_level1_100tasks


# ---------- attacker qwen3-vl-8b-instruct results ----------

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_base_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_base_defense_level1_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/spag_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_spag_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_art_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/online_sft_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_online_sft_iter_10_vs_qwen8b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-8b-instruct --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_iter_10_defense_level1_vs_qwen8b_100tasks


# # # ---------- attacker qwen3-vl-32b-instruct results ----------
# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_base_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_base_defense_level1_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_iter_10_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/spag_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_spag_iter_10_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/grpo_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_art_iter_10_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_old_prompt_48/online_sft_gemma_3_12b_it_both_merged_art_agent_iterative_seed_0/rl/agent/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --result-dir cache/adversarial_gemma12b_online_sft_iter_10_vs_qwen32b_100tasks

# bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/qwen3-vl-32b-instruct --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_iter_10_defense_level1_vs_qwen32b_100tasks


# ---------- our method but use defensive prompt ----------

bash run_adversarial_eval.sh --agent /home/lhaoyu_google_com/spag_ckpt/multimodal_adversarial_html_enhanced_sft_task_focused_after_imitation/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative_seed_0/rl/iter_10_merged --attacker /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-27b-it --workers 8 --instruction-path agent/prompts/jsons/p_som_cot_id_actree_3s_defense.json --defense-level 1 --result-dir cache/adversarial_gemma12b_iter_10_defense_level1_vs_gemma27b_100tasks_run2