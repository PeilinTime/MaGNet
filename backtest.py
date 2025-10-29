import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from Dataset import StockDataset
from MaGNet import MaGNet
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def generate_close_price(data_path, T):
    data = torch.load(data_path)

    total_date = data.shape[1]
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)

    test_data = data[:, valid_cutoff:]

    test_close_price = test_data[:, T - 1:, 0]

    return test_close_price


def generate_logits(data_path, weight_path, dim, num_experts, num_heads_mha, num_channels, num_heads_CausalMHA,
         T, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
         num_S2DAttn, num_GPH, M2):
    print("Loading data...")
    data = torch.load(data_path).to(device)

    # Data parameters
    N, total_date, F = data.shape
    # Data split
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)

    test_data = data[:, valid_cutoff:]

    # Standardize test data
    epsilon = 1e-6
    test_data_mean = test_data.mean(dim=1, keepdim=True)
    test_data_std = test_data.std(dim=1, keepdim=True)
    test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

    # Create datasets
    test_dataset = StockDataset(test_data, T, device)

    # Calculate the number of trading days
    T_test = test_data.shape[1] - T  # Number of test trading days

    # Initialize model with same parameters as in training
    model = MaGNet(N, T, F, dim, num_MAGE, num_experts,
                 num_heads_mha, num_F2DAttn, num_channels,
                 num_heads_CausalMHA, num_TCH, TopK, M1,
                 num_S2DAttn, num_GPH, M2, device=device, dropout=0.1).to(device)

    # Load model weights
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()

    # Initialize output tensors
    test_logits = torch.zeros(N, T_test, 2, device=device)

    # Initialize label tensors for metrics calculation
    test_labels = torch.zeros(N, T_test, dtype=torch.long, device=device)

    with torch.no_grad():
        for idx in tqdm(range(T_test), desc="Test"):
            X, y = test_dataset[idx]  # X: [N, T, F], y: [N]
            outputs, _, _, _, _, _ = model(X)  # outputs: [N, 2]
            test_logits[:, idx, :] = outputs
            test_labels[:, idx] = y

    return test_logits


class DailyPortfolioTradingStrategy:
    """
    Dynamic Daily Portfolio Trading Strategy and Backtesting Framework

    This class implements a trading strategy that:
    - Predicts stock price movements using logits
    - Constructs equal capital portfolios
    - Handles daily rebalancing with transaction costs
    - Includes stop-loss mechanisms
    - Calculates comprehensive performance metrics
    """

    def __init__(self, data_name,
                 initial_capital: float = 1_000_000,
                 transaction_cost_rate: float = 0.0025,
                 k_stocks: Optional[int] = None,
                 p_ratio: Optional[float] = None,
                 q_stop_loss: float = 0.5,
                 r_rising_ratio: float = 1.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize the trading strategy

        Args:
            initial_capital: Starting capital (default: 1,000,000)
            transaction_cost_rate: Cost per trade as percentage (default: 0.25%)
            p_ratio: Proportion of stocks to hold (0 < p <= 1)
            q_stop_loss: Stop loss threshold ratio (0 < q < 1)
            r_rising_ratio: Ratio of rising stocks to buy when stop-loss not triggered (0 <= r <= 1)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.k_stocks = k_stocks
        self.p_ratio = p_ratio
        self.q_stop_loss = q_stop_loss
        self.r_rising_ratio = r_rising_ratio
        self.risk_free_rate = risk_free_rate

        # Validate parameters
        if k_stocks is None and p_ratio is None:
            raise ValueError("Either k_stocks or p_ratio must be specified")
        if p_ratio is not None and not (0 < p_ratio <= 1):
            raise ValueError("p_ratio must be 0 < p_ratio <= 1")
        if not (0 < q_stop_loss < 1):
            raise ValueError("q_stop_loss must be 0 < q_stop_loss < 1")
        if not (0 <= r_rising_ratio <= 1):
            raise ValueError("r_rising_ratio must be 0 <= r_rising_ratio <= 1")

        # Initialize tracking variables
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking variables for a new backtest"""
        self.portfolio_values = []
        self.daily_returns = []
        self.cumulative_returns = []
        self.positions = []
        self.cash_history = []
        self.transaction_costs = []
        self.transaction_counts = []
        self.daily_holdings = []

    def calculate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities using softmax

        Args:
            logits: Tensor of shape [N, 2] for N stocks

        Returns:
            Probabilities tensor of shape [N, 2]
        """
        return torch.softmax(logits, dim=-1)

    def select_stocks(self, rise_probs: torch.Tensor, n_stocks: int) -> Tuple[List[int], int]:
        """
        Select stocks based on rise probability and stop-loss mechanism

        Args:
            rise_probs: Tensor of rise probabilities for each stock
            n_stocks: Total number of stocks

        Returns:
            Tuple of (selected stock indices, target number of stocks)
        """
        # Determine target number of stocks
        if self.k_stocks is not None:
            target_stocks = min(self.k_stocks, n_stocks)
        else:
            target_stocks = int(self.p_ratio * n_stocks)

        # Find stocks predicted to rise (prob > 0.5)
        rising_stocks = torch.where(rise_probs > 0.5)[0]
        m_rising = len(rising_stocks) # m_rising: number of stocks predicted to rise

        # Apply stop-loss mechanism
        stop_loss_threshold = int(target_stocks * self.q_stop_loss) # target_stocks = K, 0<q_stop_loss<1, 0<=stop_loss_threshold<K

        if m_rising >= target_stocks:
            # Buy top K stocks
            top_k_indices = torch.topk(rise_probs, target_stocks).indices
            selected_stocks = top_k_indices.tolist()
        elif m_rising >= stop_loss_threshold:
            # Buy only M*r stocks predicted to rise (select the top M*r by probability), 0<r_rising_ratio<=1, 0<=num_to_select<=M
            num_to_select = int(m_rising * self.r_rising_ratio)
            if num_to_select > 0:
                # Get the rise probabilities for the rising stocks
                rising_probs = rise_probs[rising_stocks]
                # Get the top M*r stocks among the rising stocks
                top_indices = torch.topk(rising_probs, num_to_select).indices
                selected_stocks = rising_stocks[top_indices].tolist()
                target_stocks = num_to_select
            else:
                selected_stocks = []
                target_stocks = 0
        else:
            # Stop buying today
            selected_stocks = []
            target_stocks = 0

        return selected_stocks, target_stocks

    def calculate_transaction_cost(self, value: float) -> float:
        """Calculate transaction cost for a given trade value"""
        return value * self.transaction_cost_rate

    def execute_trades(self,
                       current_holdings: Dict[int, float],
                       target_stocks: List[int],
                       stock_prices: np.ndarray,
                       cash: float) -> Tuple[Dict[int, float], float, float, int]:
        """
        Execute trades to transition from current to target holdings

        Args:
            current_holdings: Dict of {stock_idx: num_shares}
            target_stocks: List of target stock indices
            stock_prices: Current stock prices
            cash: Available cash

        Returns:
            Tuple of (new_holdings, new_cash, total_transaction_cost, num_transactions)
        """
        new_holdings = {}
        total_cost = 0
        num_transactions = 0

        # Step 1: Liquidate positions not in target
        for stock_idx, shares in current_holdings.items():
            if stock_idx not in target_stocks:
                # Sell position
                sale_value = shares * stock_prices[stock_idx]
                transaction_cost = self.calculate_transaction_cost(sale_value)
                cash += sale_value - transaction_cost
                total_cost += transaction_cost
                num_transactions += 1
            else:
                # Keep position for rebalancing
                new_holdings[stock_idx] = shares

        # Calculate portfolio value including cash
        portfolio_value = cash
        for stock_idx, shares in new_holdings.items():
            portfolio_value += shares * stock_prices[stock_idx]

        # Step 2: Calculate target allocation
        if len(target_stocks) > 0:
            target_value_per_stock = portfolio_value / (1 + self.transaction_cost_rate) / len(target_stocks)

            # Step 3: Rebalance existing positions and buy new ones
            for stock_idx in target_stocks:
                current_value = new_holdings.get(stock_idx, 0) * stock_prices[stock_idx]
                target_value = target_value_per_stock

                if abs(current_value - target_value) > 1e-6:  # Avoid tiny trades, can set eps = 1e-6 here
                    if current_value < target_value:
                        # Buy more shares
                        buy_value = target_value - current_value
                        shares_to_buy = buy_value / stock_prices[stock_idx]
                        transaction_cost = self.calculate_transaction_cost(buy_value)
                        new_holdings[stock_idx] = new_holdings.get(stock_idx, 0) + shares_to_buy
                        cash -= buy_value + transaction_cost
                        total_cost += transaction_cost
                        num_transactions += 1

                    else:
                        # Sell excess shares
                        sell_value = current_value - target_value
                        shares_to_sell = sell_value / stock_prices[stock_idx]
                        transaction_cost = self.calculate_transaction_cost(sell_value)
                        new_holdings[stock_idx] -= shares_to_sell
                        cash += sell_value - transaction_cost
                        total_cost += transaction_cost
                        num_transactions += 1


        return new_holdings, cash, total_cost, num_transactions

    def run_backtest(self,
                     logits: torch.Tensor,
                     close_prices: torch.Tensor) -> Dict[str, float]:
        """
        Run the complete backtest

        Args:
            logits: Tensor of shape [N, T, 2] - price movement predictions
            close_prices: Tensor of shape [N, T+1] - actual close prices

        Returns:
            Dictionary of performance metrics
        """
        # Reset tracking
        self.reset_tracking()

        # Convert tensors to numpy for easier manipulation
        logits_np = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
        prices_np = close_prices.cpu().numpy() if isinstance(close_prices, torch.Tensor) else close_prices

        n_stocks, t_days = logits_np.shape[0], logits_np.shape[1]

        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}  # {stock_idx: num_shares}

        # Initial portfolio value
        self.portfolio_values.append(self.initial_capital)
        self.cash_history.append(cash)

        # Run daily trading cycle
        for day in range(t_days):
            # Get today's predictions and prices
            day_logits = torch.tensor(logits_np[:, day, :])
            day_probs = self.calculate_probabilities(day_logits)
            rise_probs = day_probs[:, 1]  # Index 1 is for rise
            current_prices = prices_np[:, day]

            # Select target stocks
            target_stocks, _ = self.select_stocks(rise_probs, n_stocks)

            # Execute trades
            holdings, cash, transaction_cost, num_trades = self.execute_trades(
                holdings, target_stocks, current_prices, cash
            )

            # Track metrics
            self.transaction_costs.append(transaction_cost)
            self.transaction_counts.append(num_trades)
            self.daily_holdings.append(list(holdings.keys()))

            # Calculate end-of-day portfolio value using next day's prices
            next_prices = prices_np[:, day + 1]
            portfolio_value = cash
            for stock_idx, shares in holdings.items():
                portfolio_value += shares * next_prices[stock_idx]

            self.portfolio_values.append(portfolio_value)
            self.cash_history.append(cash)

            # Calculate daily return
            daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)

            # Calculate cumulative return
            cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.cumulative_returns.append(cumulative_return)

            # Store positions
            self.positions.append(holdings.copy())


        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()

        return metrics

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics:

        final_portfolio_value
        cumulative_return
        annual_return
        sharpe_ratio
        max_drawdown
        calmar_ratio
        """
        metrics = {}

        # Basic metrics
        metrics['final_portfolio_value'] = self.portfolio_values[-1]
        metrics['cumulative_return'] = self.cumulative_returns[-1] if self.cumulative_returns else 0

        # Annual return (252 trading days per year)
        trading_days = len(self.daily_returns)
        if trading_days > 0:
            metrics['annual_return'] = metrics['cumulative_return'] * (252 / trading_days)
        else:
            metrics['annual_return'] = 0

        # Risk metrics
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)

            # Volatility
            metrics['volatility'] = np.std(returns_array) * np.sqrt(252)  # annualized volatility

            # Sharpe Ratio
            if metrics['volatility'] > 0:
                excess_return = metrics['annual_return'] - self.risk_free_rate
                metrics['sharpe_ratio'] = excess_return / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0

            # Maximum Drawdown
            peak = self.portfolio_values[0]
            max_drawdown = 0
            for value in self.portfolio_values[1:]:
                peak = max(peak, value)
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            metrics['max_drawdown'] = max_drawdown

            # Calmar Ratio
            if metrics['max_drawdown'] > 0:
                metrics['calmar_ratio'] = metrics['annual_return'] / metrics['max_drawdown']
            else:
                metrics['calmar_ratio'] = float('inf')


        else:
            metrics['volatility'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
            metrics['calmar_ratio'] = 0

        return metrics

    def plot_results(self, data_name, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of trading results

        Args:
            save_path: Path to save the plot (if None, display only)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'MaGNet Backtesting Results on {data_name}', fontsize=16)

        # 1. Portfolio Value Over Time
        ax = axes[0]
        ax.plot(self.portfolio_values, linewidth=2, color='blue')
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.legend()

        # 2. Daily Returns Distribution
        ax = axes[1]
        if self.daily_returns:
            ax.hist([r * 100 for r in self.daily_returns], bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax.set_title('Daily Returns Distribution')
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_return = np.mean(self.daily_returns) * 100
            ax.axvline(x=mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.2f}%')
            ax.legend()

        # 3. Drawdown Chart
        ax = axes[2]
        drawdowns = []
        peak = self.portfolio_values[0]
        for value in self.portfolio_values:
            peak = max(peak, value)
            drawdown = (value - peak) / peak * 100
            drawdowns.append(drawdown)

        ax.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        ax.plot(drawdowns, color='darkred', linewidth=1)
        ax.set_title('Drawdown Over Time')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"Backtesting_result_{data_name}.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            base, ext = save_path.rsplit('.', 1)
            new_save_path = f"{base}_{data_name}.{ext}"
            plt.savefig(new_save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def print_performance_summary(self, metrics: Dict[str, float]):
        """Print a formatted summary of performance metrics"""
        print("\n" + "=" * 60)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print("=" * 60)

        print("\nRETURN METRICS:")
        print(f"  Initial Capital:         {self.initial_capital:,.2f}")
        print(f"  Final Portfolio Value:   {metrics['final_portfolio_value']:,.2f}")
        print(f"  Annual Return (AR):      {metrics['annual_return'] * 100:.2f}%")

        print("\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio (SR):       {metrics['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio (CR):       {metrics['calmar_ratio']:.3f}")

        print("\nRISK METRICS:")
        print(f"  Maximum Drawdown (MDD):  {metrics['max_drawdown'] * 100:.2f}%")

        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    dim = 32 # size of Feature Embedding, D
    num_experts = 4  # number of experts in MoE in MAGE block
    num_heads_MHA = 2 # number of heads in MHA in MAGE block
    num_channel = 4 # number of channels in Feature-wise/Stock-wise 2D Spatiotemporal Attention
    num_heads_CausalMHA = 2 # number of heads in CausalMHA in TCH


    ###  For dataset DJIA:

    data_name = "DJIA"
    data_path = 'djia_alpha158_alpha360.pt'
    weight_path = 'djia_weight.pt'
    T = 20
    num_MAGE = 2 # number of MAGE block
    num_F2DAttn = 1 # number of Feature-wise 2D Spatiotemporal Attention
    num_TCH = 1  # number of TCH
    M1 = 32 # number of hyperedges in TCH
    TopK = 32 # TopK sparsification in TCH
    num_S2DAttn = 1 # number of  Stock-wise 2D Spatiotemporal Attention
    num_GPH = 1 # number of GPH
    M2 = 16 # number of hyperedges in GPH
    p_ratio = 1
    q_stop_loss = 0.05
    r_rising_ratio = 0
    test_logits = generate_logits(data_path, weight_path, dim, num_experts, num_heads_MHA, num_channel, num_heads_CausalMHA,
         T, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
         num_S2DAttn, num_GPH, M2)
    test_close_prices = generate_close_price(data_path, T)
    strategy_p = DailyPortfolioTradingStrategy(
        initial_capital=1_000_000,
        transaction_cost_rate=0.0025,
        p_ratio=p_ratio,
        q_stop_loss=q_stop_loss,
        r_rising_ratio=r_rising_ratio,
        risk_free_rate=0.02,
        data_name = data_name
    )
    # Run backtest
    metrics_p = strategy_p.run_backtest(test_logits, test_close_prices)
    # Print results
    strategy_p.print_performance_summary(metrics_p)
    # Plot results
    strategy_p.plot_results(data_name)




    ###  For dataset NASDAQ100:

    # data_name = "NASDAQ100"
    # data_path = 'nas100_alpha158_alpha360.pt'
    # weight_path = 'nas100_weight.pt'
    # T = 10
    # num_MAGE = 1 # number of MAGE block
    # num_F2DAttn = 1 # number of Feature-wise 2D Spatiotemporal Attention
    # num_TCH = 2  # number of TCH
    # M1 = 64  # number of hyperedges in TCH
    # TopK = 64  # TopK sparsification in TCH
    # num_S2DAttn = 1 # number of  Stock-wise 2D Spatiotemporal Attention
    # num_GPH = 2 # number of GPH
    # M2 = 32 # number of hyperedges in GPH
    # p_ratio = 1
    # q_stop_loss = 0.4
    # r_rising_ratio = 1
    # test_logits = generate_logits(data_path, weight_path, dim, num_experts, num_heads_MHA, num_channel, num_heads_CausalMHA,
    #      T, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
    #      num_S2DAttn, num_GPH, M2)
    # test_close_prices = generate_close_price(data_path, T)
    # strategy_p = DailyPortfolioTradingStrategy(
    #     initial_capital=1_000_000,
    #     transaction_cost_rate=0.0025,
    #     p_ratio=p_ratio,
    #     q_stop_loss=q_stop_loss,
    #     r_rising_ratio=r_rising_ratio,
    #     risk_free_rate=0.02,
    #     data_name = data_name
    # )
    # metrics_p = strategy_p.run_backtest(test_logits, test_close_prices)
    # strategy_p.print_performance_summary(metrics_p)
    # strategy_p.plot_results(data_name)





    ###  For dataset CSI300:

    # data_name = "CSI300"
    # data_path = 'csi300_alpha158_alpha360.pt'
    # weight_path = 'csi300_weight.pt'
    # T = 10
    # num_MAGE = 1 # number of MAGE block
    # num_F2DAttn = 1 # number of Feature-wise 2D Spatiotemporal Attention
    # num_TCH = 2  # number of TCH
    # M1 = 64  # number of hyperedges in TCH
    # TopK = 64  # TopK sparsification in TCH
    # num_S2DAttn = 1 # number of  Stock-wise 2D Spatiotemporal Attention
    # num_GPH = 1 # number of GPH
    # M2 = 32 # number of hyperedges in GPH
    # p_ratio = 1  # Hold top p*N stocks
    # q_stop_loss = 0.05
    # r_rising_ratio = 0  # Only buy r*100% of rising stocks in middle condition
    # test_logits = generate_logits(data_path, weight_path, dim, num_experts, num_heads_MHA, num_channel, num_heads_CausalMHA,
    #      T, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
    #      num_S2DAttn, num_GPH, M2)
    # test_close_prices = generate_close_price(data_path, T)
    # strategy_p = DailyPortfolioTradingStrategy(
    #     initial_capital=1_000_000,
    #     transaction_cost_rate=0.0025,
    #     p_ratio=p_ratio,  # Hold top p*N stocks
    #     q_stop_loss=q_stop_loss,
    #     r_rising_ratio=r_rising_ratio,  # Only buy r*100% of rising stocks in middle condition
    #     risk_free_rate=0.02,
    #     data_name = data_name
    # )
    # metrics_p = strategy_p.run_backtest(test_logits, test_close_prices)
    # strategy_p.print_performance_summary(metrics_p)
    # strategy_p.plot_results(data_name)