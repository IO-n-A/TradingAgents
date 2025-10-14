# MLOps Backtesting Deployment

This directory contains scripts and configurations for running backtests on trained Reinforcement Learning (RL) agents. Backtesting is a crucial step to evaluate the performance of a trading strategy on historical data before considering paper or live trading.

## Files

*   **`run_backtest.py`**:
    *   This script is responsible for loading a trained RL agent, setting up the trading environment with historical test data, and running the backtesting process.
    *   It will utilize FinRL's testing and plotting capabilities.
    *   **Inputs:**
        *   Path to a trained RL agent model (from the model registry).
        *   Path to processed historical test data (tracked by DVC).
        *   Configuration for the trading environment (e.g., from `MLOps/config/environments/stock_trading_env_config.yaml`).
        *   Date range for backtesting (can be specified via `MLOps/config/common/global_vars.yaml` or command-line arguments).
    *   **Outputs:**
        *   Performance metrics (e.g., Cumulative Returns, Sharpe Ratio, Max Drawdown, Sortino Ratio, Calmar Ratio).
        *   Portfolio value over time plot.
        *   Trade history/log.
        *   Results will be saved to a structured directory, for example, `MLOps/results/backtesting/<agent_name>/<timestamp>/`.

## Workflow

1.  **Prepare Data:** Ensure that the historical test dataset is processed, includes all necessary features (including sentiment scores if applicable), and is tracked by DVC. The test data period should not overlap with the training data period.
2.  **Select Model:** Choose a trained RL agent model from the `MLOps/model_registry/`.
3.  **Configure Backtest:**
    *   Specify the agent model path in `run_backtest.py` or via command-line arguments.
    *   Ensure the environment configuration in `MLOps/config/environments/stock_trading_env_config.yaml` matches the one used for training (or is appropriate for testing).
    *   Define the backtesting period.
4.  **Run Backtest Script:** Execute `python MLOps/deployment/backtesting/run_backtest.py`.
5.  **Analyze Results:**
    *   Examine the generated performance metrics and plots.
    *   Compare results against benchmarks or different model versions.
    *   Store results systematically for future reference and comparison. MLflow can also be used to log backtesting results as artifacts of a "testing" run associated with a trained model.

## Key Metrics to Evaluate:

*   **Cumulative Returns:** Total return over the backtesting period.
*   **Annualized Return:** Return scaled to a yearly basis.
*   **Annualized Volatility:** Standard deviation of returns, indicating risk.
*   **Sharpe Ratio:** Risk-adjusted return (excess return per unit of volatility).
*   **Sortino Ratio:** Similar to Sharpe, but only considers downside volatility.
*   **Max Drawdown:** Largest peak-to-trough decline during a specific period.
*   **Calmar Ratio:** Annualized return divided by the absolute value of the maximum drawdown.
*   **Win/Loss Ratio:** Ratio of winning trades to losing trades.
*   **Average Win/Loss Amount:** Average profit from winning trades versus average loss from losing trades.

## Considerations:

*   **Look-ahead Bias:** Ensure no future information is used in making decisions at any point in the backtest.
*   **Transaction Costs:** Accurately model transaction costs (brokerage fees, slippage).
*   **Data Snooping:** Avoid overfitting the strategy to the historical test data by repeatedly testing and tweaking. Use a separate out-of-sample validation set if possible.
*   **Benchmarking:** Compare the strategy's performance against relevant benchmarks (e.g., buy-and-hold the S&P 500).
*   **Robustness Checks:** Test the strategy on different time periods or market conditions.

This backtesting setup aims to provide a standardized way to evaluate trading strategies developed within the FinAI_algo project.