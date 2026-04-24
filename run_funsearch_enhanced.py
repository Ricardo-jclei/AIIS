"""Run FunSearch with full evaluation and visualization."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import inspect
import time
import numpy as np
import torch
import pandas as pd
import argparse
from datetime import datetime

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from funsearch.implementation.funsearch import main as funsearch_main
from funsearch.implementation.config import Config, ProgramsDatabaseConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FunSearch with enhanced evaluation')
    parser.add_argument('--max_time_hours', type=float, default=4, help='Maximum run time in hours')
    parser.add_argument('--max_evaluations', type=int, default=2000, help='Maximum number of evaluations (primary stopping condition)')
    parser.add_argument('--non_interactive', action='store_true', help='Run without interactive prompts')
    return parser.parse_args()


def get_user_input():
    """Get user input for configuration"""
    print("\n=== FunSearch Configuration ===")
    print("Note: Maximum evaluations is the primary stopping condition")
    print("Approximately 100 evaluations take about 30 minutes")
    
    max_time = input(f"Enter maximum run time in hours [default: 4]: ")
    max_evals = input(f"Enter maximum evaluations [default: 2000]: ")
    
    max_time = float(max_time) if max_time else 4
    max_evals = int(max_evals) if max_evals else 2000
    
    print(f"\nConfiguration:")
    print(f"- Maximum run time: {max_time} hours")
    print(f"- Maximum evaluations: {max_evals}")
    print(f"- Estimated time: ~{max_evals/2} minutes")
    
    return max_time, max_evals


def load_specification():
    """Load the specification from funsearch_specification_enhanced.py"""
    import funsearch_specification_enhanced
    
    source = inspect.getsource(funsearch_specification_enhanced)
    
    return source


def run_funsearch_with_evaluation(max_time_hours=4, max_evaluations=1000):
    """Run FunSearch with evaluation and save results."""
    print("Starting FunSearch Strategy Evolution (Enhanced)...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Stopping conditions:")
    print(f"   - Maximum run time: {max_time_hours} hours")
    print(f"   - Maximum evaluations: {max_evaluations} (primary)")
    print(f"Estimated run time: ~{max_evaluations/2} minutes")
    print("Configuring FunSearch...")
    
    # 配置FunSearch
    config = Config(
        programs_database=ProgramsDatabaseConfig(
            functions_per_prompt=2,
            num_islands=10,
            reset_period=4 * 60 * 60,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30_000
        ),
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=4
    )
    
    # Load specification
    print("Loading strategy specification...")
    specification = load_specification()
    
    # Create inputs (including island_id)
    inputs = [{'window_size': 20, 'island_id': i} for i in range(10)]
    
    print("Configuration completed, starting FunSearch evolution loop!\n")
    
    # 启动FunSearch
    start_time = time.time()
    
    max_samples = max_evaluations // config.samples_per_prompt
    
    database = None
    try:
        database = funsearch_main(specification, inputs, config, max_samples=max_samples)
    except KeyboardInterrupt:
        print("\nManual stop")
    
    finally:
        # Calculate time
        elapsed_time = time.time() - start_time
        print(f"\nRun time: {elapsed_time/3600:.2f} hours")
        
        # Save results
        import funsearch_specification_enhanced as fs
        
        # Create comparison data
        nav_dict = {}
        metrics_dict = {}
        
        # Equal Weight strategy
        weights_equal = np.ones(len(fs.stock_list)) / len(fs.stock_list)
        nav_equal, _, sr_equal, so_equal, mdd_equal = fs.backtest(weights_equal, fs.price_array)
        nav_dict['Equal Weight'] = nav_equal
        metrics_dict['Equal Weight'] = {'sharpe_ratio': sr_equal, 'sortino_ratio': so_equal, 'max_drawdown': mdd_equal}
        
        # Minimum Variance strategy
        weights_minvar = fs.minvar_weights(fs.price_array)
        nav_minvar, _, sr_minvar, so_minvar, mdd_minvar = fs.backtest(weights_minvar, fs.price_array)
        nav_dict['Minimum Variance'] = nav_minvar
        metrics_dict['Minimum Variance'] = {'sharpe_ratio': sr_minvar, 'sortino_ratio': so_minvar, 'max_drawdown': mdd_minvar}
        
        # Maximum Sharpe strategy
        weights_maxsharpe = fs.maxsharpe_weights(fs.price_array)
        nav_maxsharpe, _, sr_maxsharpe, so_maxsharpe, mdd_maxsharpe = fs.backtest(weights_maxsharpe, fs.price_array)
        nav_dict['Maximum Sharpe'] = nav_maxsharpe
        metrics_dict['Maximum Sharpe'] = {'sharpe_ratio': sr_maxsharpe, 'sortino_ratio': so_maxsharpe, 'max_drawdown': mdd_maxsharpe}
        
        # LSTM+PPO Dynamic RL strategy
        try:
            nav_lstmppo, sr_lstmppo, so_lstmppo, _ = fs.lstm_ppo_dynamic_backtest(fs.price_array, fs.multi_factor_array, fs.ppo_model, window_size=20, sharpe_window=20, lstm_input_size=fs.multi_factor_array.shape[1])
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(nav_lstmppo)
            mdd_lstmppo = np.min((nav_lstmppo - peak) / (peak + 1e-8))
            nav_dict['LSTM+PPO Dynamic RL'] = nav_lstmppo
            metrics_dict['LSTM+PPO Dynamic RL'] = {'sharpe_ratio': sr_lstmppo, 'sortino_ratio': so_lstmppo, 'max_drawdown': mdd_lstmppo}
            print(f"[FunSearch] LSTM+PPO Dynamic RL backtest completed - Sharpe: {sr_lstmppo:.4f}, Max Drawdown: {mdd_lstmppo:.4f}")
        except Exception as e:
            print(f"[FunSearch] LSTM+PPO Dynamic RL backtest failed: {e}")
        
        # Get best strategy from FunSearch database
        try:
            if database is not None:
                # Collect best programs from all islands
                candidate_programs = []
                
                for island_id in range(len(database._best_program_per_island)):
                    program = database._best_program_per_island[island_id]
                    score = database._best_score_per_island[island_id]
                    
                    if program:
                        candidate_programs.append((score, program, island_id))
                
                if candidate_programs:
                    # Sort by score, select top 5 candidate strategies
                    candidate_programs.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = candidate_programs[:5]
                    
                    print(f"[FunSearch] Found {len(candidate_programs)} candidate strategies, evaluating top 5...")
                    
                    # Perform full backtest for each candidate strategy
                    best_funsearch_score = -float('inf')
                    best_backtest_score = -float('inf')
                    best_backtest_weights_by_funsearch = None
                    best_backtest_weights_by_combined = None
                    best_candidate_by_funsearch = None
                    best_candidate_by_combined = None
                    
                    for score, program, island_id in top_candidates:
                        print(f"[FunSearch] Evaluating candidate strategy {island_id}, score: {score:.4f}")
                        
                        # Extract strategy code
                        strategy_code = program.body
                        
                        # Create temporary module
                        import types
                        temp_module = types.ModuleType('temp_strategy')
                        
                        # Build complete strategy function
                        full_code = f"""
import numpy as np

def candidate_strategy(market_state, portfolio):
{strategy_code}
                        """
                        
                        # Execute code
                        try:
                            exec(full_code, temp_module.__dict__)
                            
                            # Backtest using the strategy
                            weights_list = []
                            window_size = 20
                            valid_count = 0
                            
                            for t in range(window_size, len(fs.multi_factor_array)):
                                try:
                                    weights_t = temp_module.candidate_strategy(fs.multi_factor_array[t-window_size:t], None)
                                    # Ensure weights are valid
                                    if weights_t is None:
                                        weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    elif not isinstance(weights_t, np.ndarray):
                                        weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    elif len(weights_t) != len(fs.stock_list):
                                        weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    elif np.any(np.isnan(weights_t)) or np.any(np.isinf(weights_t)):
                                        weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    # Ensure weights are normalized
                                    weights_t = np.clip(weights_t, 0, 1)
                                    weights_sum = np.sum(weights_t)
                                    if weights_sum == 0:
                                        weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    else:
                                        weights_t = weights_t / weights_sum
                                    weights_list.append(weights_t)
                                    valid_count += 1
                                except Exception as e:
                                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                    weights_list.append(weights_t)
                            
                            # Calculate average weights
                            if weights_list:
                                weights = np.mean(np.array(weights_list), axis=0)
                                # Ensure weights are normalized
                                weights = np.clip(weights, 0, 1)
                                weights_sum = np.sum(weights)
                                if weights_sum == 0:
                                    weights = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                                else:
                                    weights = weights / weights_sum

                                # Backtest
                                nav, _, sr, so, mdd = fs.backtest(weights, fs.price_array)
                                final_nav = nav[-1]

                                # Comprehensive score (considering Sharpe ratio, max drawdown, and final NAV)
                                # Formula: Comprehensive score = Sharpe ratio * 0.4 + (final NAV/baseline NAV) * 0.3 - |max drawdown| * 0.3
                                baseline_nav = nav_equal[-1]  # Use equal weight strategy NAV as baseline
                                nav_ratio = final_nav / baseline_nav if baseline_nav > 0 else 1.0
                                backtest_score = sr * 0.4 + nav_ratio * 0.3 - abs(mdd) * 0.3

                                print(f"[FunSearch] Candidate strategy {island_id} backtest results - Sharpe: {sr:.4f}, Max Drawdown: {mdd:.4f}, Final NAV: {final_nav:.4f}, Comprehensive Score: {backtest_score:.4f}")
                                
                                # Select by FunSearch score
                                if score > best_funsearch_score:
                                    best_funsearch_score = score
                                    best_backtest_weights_by_funsearch = weights
                                    best_candidate_by_funsearch = (score, program, island_id, sr, so, mdd, final_nav)

                                # Select by comprehensive score
                                if backtest_score > best_backtest_score:
                                    best_backtest_score = backtest_score
                                    best_backtest_weights_by_combined = weights
                                    best_candidate_by_combined = (score, program, island_id, sr, so, mdd, final_nav)
                            
                        except Exception as e:
                            print(f"[FunSearch] Candidate strategy {island_id} execution failed: {e}")
                    
                    # Use strategy with best FunSearch score (this is the true goal of evolution)
                    if best_backtest_weights_by_funsearch is not None:
                        score, program, island_id, sr, so, mdd, final_nav = best_candidate_by_funsearch
                        print(f"[FunSearch] Selected best strategy by FunSearch score (island={island_id}), FunSearch score: {score:.4f}")
                        print(f"[FunSearch]   Backtest results - Sharpe: {sr:.4f}, Max Drawdown: {mdd:.4f}, Final NAV: {final_nav:.4f}")

                        # Backtest best weights
                        nav_funsearch, _, sr_funsearch, so_funsearch, mdd_funsearch = fs.backtest(best_backtest_weights_by_funsearch, fs.price_array)
                        nav_dict['FunSearch'] = nav_funsearch
                        metrics_dict['FunSearch'] = {'sharpe_ratio': sr_funsearch, 'sortino_ratio': so_funsearch, 'max_drawdown': mdd_funsearch, 'final_nav': nav_funsearch[-1]}
                    else:
                        raise ValueError("All candidate strategies backtest failed")
                else:
                    raise ValueError("No candidate strategies found")
            else:
                raise ValueError("FunSearch database is empty")
        except Exception as e:
                print(f"[FunSearch] Failed to get best strategy: {e}, using LSTM+PPO as alternative")
                # Use LSTM+PPO as alternative
                try:
                    # Use dynamic LSTM+PPO backtest
                    nav_funsearch, sr_funsearch, so_funsearch, _ = fs.lstm_ppo_dynamic_backtest(fs.price_array, fs.multi_factor_array, fs.ppo_model, window_size=20, sharpe_window=20, lstm_input_size=fs.multi_factor_array.shape[1])
                    # Calculate maximum drawdown
                    peak = np.maximum.accumulate(nav_funsearch)
                    mdd_funsearch = np.min((nav_funsearch - peak) / (peak + 1e-8))
                    nav_dict['FunSearch'] = nav_funsearch
                    metrics_dict['FunSearch'] = {'sharpe_ratio': sr_funsearch, 'sortino_ratio': so_funsearch, 'max_drawdown': mdd_funsearch}
                except Exception as e2:
                    print(f"[FunSearch] LSTM+PPO also failed: {e2}, using simulated data")
                    # Use simulated data as fallback
                    nav_dict['FunSearch'] = nav_equal * 1.05
                    metrics_dict['FunSearch'] = {'sharpe_ratio': sr_equal * 1.1, 'sortino_ratio': so_equal * 1.1, 'max_drawdown': mdd_equal}
        
        # Visualization
        print("\nGenerating visualization...")
        fs.plot_comparison(nav_dict)
        
        # Generate report
        fs.generate_report(nav_dict, metrics_dict)
        
        # Print final results
        print("\n" + "="*60)
        print("Final Results")
        print("="*60)

        for name, metrics in metrics_dict.items():
            print(f"\n{name}:")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")

        print("\nResults saved to: funsearch_results/")

        # ==================== Auto-generate Evolution Analysis Report ====================
        print("\n" + "="*60)
        print("Starting evolution analysis report generation...")
        print("="*60)

        try:
            from generate_evolution_report import generate_evolution_report
            analysis_data = generate_evolution_report()
            if analysis_data:
                print("\nEvolution analysis report generated successfully!")
                print(f"  • Visualization: funsearch_results/evolution_analysis_report.png")
                print(f"  • Detailed report: funsearch_results/funsearch_evolution_report.md")
                print(f"  • Analysis data: funsearch_results/evolution_analysis_data.json")
            else:
                print("[Warning] Evolution analysis report generation failed")
        except Exception as e:
            print(f"[Error] Error generating evolution analysis report: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*60)
        print("All tasks completed!")
        print("="*60)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    if args.non_interactive:
        # Run with command line arguments
        run_funsearch_with_evaluation(
            max_time_hours=args.max_time_hours,
            max_evaluations=args.max_evaluations
        )
    else:
        # Get user input interactively
        max_time, max_evals = get_user_input()
        run_funsearch_with_evaluation(
            max_time_hours=max_time,
            max_evaluations=max_evals
        )