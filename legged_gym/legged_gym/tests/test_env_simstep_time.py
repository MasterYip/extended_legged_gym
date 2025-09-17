import numpy as np
import matplotlib.pyplot as plt
import time
import isaacgym
import torch
import multiprocessing as mp
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import task_registry, get_args

def measure_step_time(env, num_steps=100):
    """Measure the mean simulation step time for a given environment."""
    start_time = time.time()
    for _ in range(num_steps):
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)*1.0
        env.step(actions)
    total_time = time.time() - start_time
    return total_time / num_steps

def measure_rollout_step_time(env, num_steps=100):
    """Measure the mean simulation step time for a batch rollout environment."""
    start_time = time.time()
    for _ in range(num_steps):
        actions = torch.randn(env.num_rollout_per_main * env.num_envs, env.num_actions, device=env.device)*1.0
        env.step_rollout(actions)
    total_time = time.time() - start_time
    return total_time / num_steps

def run_single_env_test(task, num_envs, num_steps):
    """Run a single environment test in a separate process"""
    print(f"Testing task: {task}, num_envs: {num_envs}")
    env_cfg, _ = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    env, _ = task_registry.make_env(name=task, args=None, env_cfg=env_cfg)
    env.reset()
    mean_step_time = measure_step_time(env, num_steps)
    print(f"Mean step time: {mean_step_time:.6f} seconds")
    return mean_step_time

def run_batch_env_test(task, num_main_envs, num_rollout_envs, num_steps,
                       measure_rollout_step=True):
    """Run a single batch rollout environment test in a separate process"""
    print(f"Testing task: {task}, num_main_envs: {num_main_envs}, num_rollout_envs: {num_rollout_envs}")
    env_cfg, _ = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_main_envs
    env_cfg.env.rollout_envs = num_rollout_envs
    env, _ = task_registry.make_env(name=task, args=None, env_cfg=env_cfg)
    env.reset()
    env._sync_main_to_rollout() # Add drift to rollout envs
    if measure_rollout_step:
        mean_step_time = measure_rollout_step_time(env, num_steps)
    else:
        mean_step_time = measure_step_time(env, num_steps)
    print(f"Mean step time: {mean_step_time:.6f} seconds")
    return mean_step_time

def worker_process(task, num_envs, num_steps, return_dict, key):
    """Worker function to run in a separate process"""
    try:
        result = run_single_env_test(task, num_envs, num_steps)
        return_dict[key] = result
    except Exception as e:
        print(f"Error in worker process: {e}")
        return_dict[key] = None

def worker_process_batch(task, num_main_envs, num_rollout_envs, num_steps, return_dict, key,
                         measure_rollout_step=True):
    """Worker function for batch rollout tests"""
    try:
        result = run_batch_env_test(task, num_main_envs, num_rollout_envs, num_steps, measure_rollout_step)
        return_dict[key] = result
    except Exception as e:
        print(f"Error in worker process: {e}")
        return_dict[key] = None

def test_basic_envs(task_list, num_envs_list, num_steps=100):
    """Test basic environments with different numbers of environments using multiprocessing."""
    results = {}
    
    for task in task_list:
        results[task] = []
        for num_envs in num_envs_list:
            # Use a Manager to share results between processes
            manager = mp.Manager()
            return_dict = manager.dict()
            key = f"{task}_{num_envs}"
            
            # Create and start a new process for this environment test
            p = mp.Process(target=worker_process, args=(task, num_envs, num_steps, return_dict, key))
            p.start()
            p.join()
            
            # Get the result
            if key in return_dict and return_dict[key] is not None:
                results[task].append(return_dict[key])
            else:
                print(f"Failed to get result for task: {task}, num_envs: {num_envs}")
                results[task].append(None)
    
    return results

def test_batch_rollout_envs(task, num_main_envs_list, num_rollout_envs_list, num_steps=100,
                            measure_rollout_step=True):
    """Test batch rollout environments with different numbers using multiprocessing."""
    results = []
    
    for i, num_main_envs in enumerate(num_main_envs_list):
        row_results = []
        for num_rollout_envs in num_rollout_envs_list:
            # Use a Manager to share results between processes
            manager = mp.Manager()
            return_dict = manager.dict()
            key = f"{num_main_envs}_{num_rollout_envs}"
            
            # Create and start a new process for this environment test
            p = mp.Process(
                target=worker_process_batch, 
                args=(task, num_main_envs, num_rollout_envs, num_steps, return_dict, key, measure_rollout_step)
            )
            p.start()
            p.join()
            
            # Get the result
            if key in return_dict and return_dict[key] is not None:
                row_results.append(return_dict[key])
            else:
                print(f"Failed to get result for main_envs: {num_main_envs}, rollout_envs: {num_rollout_envs}")
                row_results.append(None)
        
        results.append(row_results)
    
    return results

def test_batch_rollout_envs2(task, total_num_env_list, num_steps=100,
                             possible_main_envs = [1, 2, 4, 8, 16, 32, 64, 128],
                             measure_rollout_step=True):
    """
    Test batch rollout environments with different combinations that maintain similar total environment counts.
    Total environments = (rollout_envs + 1) * main_envs
    
    Args:
        task: The task name to test
        total_num_env_list: List of total environment counts to target
        num_steps: Number of simulation steps to measure
        
    Returns:
        Dictionary mapping total environment counts to lists of (main_envs, rollout_envs, step_time) tuples
    """    
    results = {}
    
    for total_envs in total_num_env_list:
        print(f"Testing for total environments ≈ {total_envs}")
        results[total_envs] = []
        
        # Find reasonable combinations of main_envs and rollout_envs
        for main_envs in possible_main_envs:
            # Calculate required rollout_envs to get close to total_envs
            # total = (rollout_envs + 1) * main_envs
            # rollout_envs = (total / main_envs) - 1
            rollout_envs = round((total_envs / main_envs) - 1)
            
            # Skip if rollout_envs is too small or negative
            if rollout_envs < 1:
                continue
                
            # Calculate actual total using this combination
            actual_total = (rollout_envs + 1) * main_envs
            
            # Skip if the actual total is too far from the target
            if abs(actual_total - total_envs) / total_envs > 0.2:  # Allow 20% deviation
                continue
                
            print(f"  Testing main_envs={main_envs}, rollout_envs={rollout_envs} (total≈{actual_total})")
            
            # Use a Manager to share results between processes
            manager = mp.Manager()
            return_dict = manager.dict()
            key = f"{main_envs}_{rollout_envs}"
            
            # Create and start a new process for this test
            p = mp.Process(
                target=worker_process_batch,
                args=(task, main_envs, rollout_envs, num_steps, return_dict, key, measure_rollout_step)
            )
            p.start()
            p.join()
            
            # Get the result
            if key in return_dict and return_dict[key] is not None:
                step_time = return_dict[key]
                results[total_envs].append((main_envs, rollout_envs, step_time))
                print(f"  Result: {step_time:.6f} seconds")
            else:
                print(f"  Failed to get result for main_envs={main_envs}, rollout_envs={rollout_envs}")
    
    return results

def plot_results_basic(results, num_envs_list):
    """Plot step time vs. num_envs for basic environments."""
    plt.figure()
    for task, times in results.items():
        # Filter out None values
        valid_times = []
        valid_envs = []
        for i, time_val in enumerate(times):
            if time_val is not None:
                valid_times.append(time_val)
                valid_envs.append(num_envs_list[i])
        
        if valid_times:
            plt.plot(valid_envs, valid_times, label=task)
    
    plt.xlabel("Number of Environments")
    plt.ylabel("Mean Step Time (s)")
    plt.title("Step Time vs. Number of Environments (Basic Envs)")
    plt.legend()
    plt.grid()
    plt.savefig("basic_env_step_time.png")
    plt.show()

def plot_results_batch_rollout(results, num_main_envs_list, num_rollout_envs_list):
    """Plot step time vs. num_envs for batch rollout environments."""
    results_array = np.array(results, dtype=object)
    plt.figure()
    
    for i, num_main_envs in enumerate(num_main_envs_list):
        # Filter out None values
        valid_times = []
        valid_rollouts = []
        for j, time_val in enumerate(results_array[i]):
            if time_val is not None:
                valid_times.append(time_val)
                valid_rollouts.append(num_rollout_envs_list[j])
                
        if valid_times:
            plt.plot(valid_rollouts, valid_times, label=f"Main Envs: {num_main_envs}")
    
    plt.xlabel("Number of Rollout Environments")
    plt.ylabel("Mean Step Time (s)")
    plt.title("Step Time vs. Rollout Environments (Batch Rollout)")
    plt.legend()
    plt.grid()
    plt.savefig("batch_rollout_step_time.png")
    plt.show()

def plot_results_constant_total(results):
    """
    Plot step time vs. num_main_envs for constant total environment counts.
    
    Args:
        results: Dictionary mapping total environment counts to lists of (main_envs, rollout_envs, step_time) tuples
    """
    plt.figure(figsize=(10, 6))
    
    # Create separate plot for each total environment count
    for total_envs, data_points in results.items():
        if not data_points:
            continue
            
        # Sort by number of main environments
        data_points.sort(key=lambda x: x[0])
        
        main_envs_values = [point[0] for point in data_points]
        step_times = [point[2] for point in data_points]
        
        plt.plot(main_envs_values, step_times, 'o-', label=f"Total ≈ {total_envs}")
        
        # Annotate each point with (main, rollout) values
        for i, (main, rollout, _) in enumerate(data_points):
            plt.annotate(f"({main}, {rollout})", 
                        (main_envs_values[i], step_times[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
    
    plt.xscale('log')
    plt.xlabel("Number of Main Environments")
    plt.ylabel("Mean Step Time (s)")
    plt.title("Step Time vs. Main Environments (Constant Total Environments)")
    plt.legend()
    plt.grid(True)
    plt.savefig("constant_total_step_time.png")
    plt.show()

if __name__ == "__main__":
    # Set multiprocessing start method at the module level
    mp.set_start_method('spawn')
    measure_rollout_step = True
    num_steps = 500
    # NOTE: To run headlessly, add --headless to the command line arguments

    # # Basic environment testing
    # basic_tasks = ["anymal_c_flat", 
    #                "elspider_air_flat",
    #                ]
    # num_envs_list = [1, 4, 16, 64, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    # # num_envs_list = [256, 1024, 2048, 4096, 8192]
    # print("Testing basic environments...")
    # basic_results = test_basic_envs(basic_tasks, num_envs_list, num_steps)
    # plot_results_basic(basic_results, num_envs_list)

    # # Original batch rollout environment testing
    # batch_task = "elspider_air_traj_grad_sampling"
    # num_main_envs_list = [1, 2, 4, 8]
    # num_rollout_envs_list = [16, 32, 64, 128]
    # print("Testing batch rollout environments...")
    # batch_results = test_batch_rollout_envs(batch_task, num_main_envs_list, num_rollout_envs_list, num_steps,
    #                                         measure_rollout_step)
    # plot_results_batch_rollout(batch_results, num_main_envs_list, num_rollout_envs_list)
    
    # New constant-total-environments testing
    batch_task = "anymal_c_batch_rollout_flat"
    # total_num_env_list = [512, 1024, 2048, 4096, 8192, 16384]
    total_num_env_list = [256, 1024, 4096]
    # possible_main_envs = [1, 2, 4, 8, 16, 32, 64, 128]
    possible_main_envs = [1, 4, 16, 64, 256]
    print("Testing batch rollout environments with constant total environments...")
    constant_total_results = test_batch_rollout_envs2(batch_task, total_num_env_list,
                                                      num_steps, possible_main_envs,
                                                      measure_rollout_step)
    plot_results_constant_total(constant_total_results)
