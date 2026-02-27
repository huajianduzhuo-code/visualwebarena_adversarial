#!/usr/bin/env python3
"""
Extract trajectory summaries from adversarial evaluation results.

This script traverses all trajectory.json files in an evaluation directory
and extracts the last step's agent actions and the full summary for each task.

Usage:
    python extract_trajectory_summary.py <result_dir>

Example:
    python extract_trajectory_summary.py cache/adversarial_gemma-3-12b-it_vs_gemma-3-12b-it_20260211_202804
"""

import json
import os
import sys
import glob
from pathlib import Path


def extract_trajectory_info(trajectory_path):
    """
    Extract relevant information from a single trajectory.json file.
    
    Args:
        trajectory_path: Path to the trajectory.json file
        
    Returns:
        dict with extracted information, or None if file is invalid
    """
    try:
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        # Extract the last step's agent action and raw prediction
        steps = data.get('steps', [])
        if steps:
            last_step = steps[-1]
            last_agent_action = last_step.get('agent_action', None)
            last_agent_raw_prediction = last_step.get('agent_raw_prediction', None)
        else:
            last_agent_action = None
            last_agent_raw_prediction = None
        
        # Extract the full summary
        summary = data.get('summary', {})
        
        # Extract metadata for context
        metadata = data.get('metadata', {})
        task_id = metadata.get('task_id')
        
        # Determine site from trajectory path
        # Path structure: .../worker_X/<site>/trajectories/<task_id>/trajectory.json
        parts = Path(trajectory_path).parts
        try:
            site_idx = [i for i, p in enumerate(parts) if 'worker_' in p][0] + 1
            site = parts[site_idx] if site_idx < len(parts) else 'unknown'
        except (IndexError, ValueError):
            site = 'unknown'
        
        return {
            'task_id': task_id,
            'site': site,
            'trajectory_path': str(trajectory_path),
            'last_step': {
                'agent_action': last_agent_action,
                'agent_raw_prediction': last_agent_raw_prediction
            },
            'summary': summary,
            'metadata': metadata
        }
    
    except Exception as e:
        print(f"Warning: Failed to process {trajectory_path}: {e}", file=sys.stderr)
        return None


def extract_all_trajectories(result_dir):
    """
    Extract information from all trajectory.json files in the result directory.
    
    Args:
        result_dir: Path to the evaluation result directory
        
    Returns:
        List of dictionaries containing extracted information
    """
    result_dir = Path(result_dir)
    
    if not result_dir.exists():
        print(f"Error: Directory {result_dir} does not exist", file=sys.stderr)
        return []
    
    # Find all trajectory.json files
    trajectory_pattern = str(result_dir / "worker_*" / "*" / "trajectories" / "*" / "trajectory.json")
    trajectory_files = glob.glob(trajectory_pattern)
    
    print(f"Found {len(trajectory_files)} trajectory files in {result_dir}")
    
    # Extract information from each trajectory
    all_trajectories = []
    for traj_path in sorted(trajectory_files):
        info = extract_trajectory_info(traj_path)
        if info is not None:
            all_trajectories.append(info)
    
    print(f"Successfully extracted {len(all_trajectories)} trajectories")
    
    return all_trajectories


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_trajectory_summary.py <result_dir>")
        print("\nExample:")
        print("  python extract_trajectory_summary.py cache/adversarial_gemma-3-12b-it_vs_gemma-3-12b-it_20260211_202804")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    # Extract all trajectories
    trajectories = extract_all_trajectories(result_dir)
    
    if not trajectories:
        print("No trajectories extracted. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Save to JSON file
    output_path = Path(result_dir) / "trajectory_summaries.json"
    with open(output_path, 'w') as f:
        json.dump(trajectories, f, indent=2, default=str)
    
    print(f"\nTrajectory summaries saved to: {output_path}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"  Total tasks: {len(trajectories)}")
    
    # Count by site
    from collections import Counter
    site_counts = Counter(t['site'] for t in trajectories)
    print("  Tasks by site:")
    for site, count in sorted(site_counts.items()):
        print(f"    {site}: {count}")
    
    # Count attack outcomes
    attack_success = sum(1 for t in trajectories if t['summary'].get('attack_success', False))
    defensive_stops = sum(1 for t in trajectories if t['summary'].get('defensive_stop', False))
    task_passed = sum(1 for t in trajectories if t['summary'].get('task_score', 0) == 1)
    
    print(f"  Attack success: {attack_success} ({attack_success/len(trajectories)*100:.1f}%)")
    print(f"  Defensive stops: {defensive_stops} ({defensive_stops/len(trajectories)*100:.1f}%)")
    print(f"  Tasks passed: {task_passed} ({task_passed/len(trajectories)*100:.1f}%)")


if __name__ == "__main__":
    main()
