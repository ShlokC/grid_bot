"""
Enhanced Grid Trading Strategy with Self-Adaptive Market Intelligence (SAMIG)
Combines traditional grid trading with advanced market analysis and adaptive parameters.
"""
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import traceback
from core.exchange import Exchange

@dataclass
class MarketSnapshot:
    """Real-time market condition snapshot"""
    timestamp: float
    price: float
    volume: float
    spread: float
    order_book_imbalance: float
    price_velocity: float
    volatility_regime: float
    momentum: float
    mean_reversion_strength: float

@dataclass
class GridPerformanceMetric:
    """Performance tracking for adaptive learning"""
    timestamp: float
    pnl: float
    win_rate: float
    trades_count: int
    market_conditions: MarketSnapshot
    grid_config: Dict

class AdaptiveParameterManager:
    """Manages dynamic parameter adjustment based on performance feedback"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.learning_rate = 0.1
        self.param_bounds = {
            'grid_density': {'min': 0.5, 'max': 3.0, 'current': 1.0},
            'trend_sensitivity': {'min': 0.1, 'max': 2.0, 'current': 1.0},
            'volatility_response': {'min': 0.3, 'max': 2.5, 'current': 1.0},
            'momentum_threshold': {'min': 0.1, 'max': 0.9, 'current': 0.5},
            'mean_reversion_factor': {'min': 0.5, 'max': 2.0, 'current': 1.0},
            'exposure_asymmetry': {'min': 0.5, 'max': 2.0, 'current': 1.0}
        }
    
    def update_parameter(self, param_name: str, performance_score: float, market_conditions: MarketSnapshot):
        """Update parameter based on performance feedback"""
        if param_name not in self.param_bounds:
            return
        
        bounds = self.param_bounds[param_name]
        current = bounds['current']
        
        # Performance-based adjustment
        if performance_score > 0:
            adjustment = self.learning_rate * (performance_score / 100)
        else:
            adjustment = -self.learning_rate * (abs(performance_score) / 100)
        
        # Market volatility modulation
        adjustment *= (1 + market_conditions.volatility_regime)
        
        # Apply bounds
        new_value = current + adjustment
        bounds['current'] = max(bounds['min'], min(bounds['max'], new_value))
    
    def get_parameter(self, param_name: str) -> float:
        """Get current adaptive parameter value"""
        return self.param_bounds.get(param_name, {}).get('current', 1.0)

class MarketIntelligenceEngine:
    """Advanced market analysis engine"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.spread_history = deque(maxlen=50)
        self.snapshot_history = deque(maxlen=100)
        self.asset_characteristics = {
            'typical_volatility': None,
            'typical_spread': None,
            'price_precision': None
        }
    
    def analyze_market_conditions(self, exchange: Exchange) -> MarketSnapshot:
        """Analyze current market microstructure"""
        try:
            ticker = exchange.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            volume = float(ticker.get('quoteVolume', 0))
            
            # Calculate spread
            bid = float(ticker.get('bid', current_price))
            ask = float(ticker.get('ask', current_price))
            spread = (ask - bid) / current_price if current_price > 0 else 0
            
            # Update histories
            self.price_history.append(current_price)
            self.volume_history.append(volume)
            self.spread_history.append(spread)
            
            # Calculate analytics
            order_book_imbalance = self._calculate_order_book_imbalance(ticker)
            price_velocity = self._calculate_price_velocity()
            volatility_regime = self._calculate_volatility_regime()
            momentum = self._calculate_momentum()
            mean_reversion_strength = self._calculate_mean_reversion_strength()
            
            snapshot = MarketSnapshot(
                timestamp=time.time(),
                price=current_price,
                volume=volume,
                spread=spread,
                order_book_imbalance=order_book_imbalance,
                price_velocity=price_velocity,
                volatility_regime=volatility_regime,
                momentum=momentum,
                mean_reversion_strength=mean_reversion_strength
            )
            
            self.snapshot_history.append(snapshot)
            self._update_asset_characteristics()
            
            return snapshot
            
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            return MarketSnapshot(time.time(), 0, 0, 0, 0, 0, 1, 0, 1)
    
    def _calculate_order_book_imbalance(self, ticker: Dict) -> float:
        """Calculate order book imbalance"""
        try:
            bid_size = float(ticker.get('bidVolume', 0))
            ask_size = float(ticker.get('askVolume', 0))
            total_size = bid_size + ask_size
            
            if total_size == 0:
                return 0
            return (bid_size - ask_size) / total_size
        except:
            return 0
    
    def _calculate_price_velocity(self) -> float:
        """Calculate rate of price change"""
        if len(self.price_history) < 10:
            return 0
        
        prices = np.array(list(self.price_history)[-10:])
        if len(prices) > 2:
            velocity = np.gradient(prices)[-1]
            return velocity / prices[-1] if prices[-1] > 0 else 0
        return 0
    
    def _calculate_volatility_regime(self) -> float:
        """Calculate current volatility relative to recent history"""
        if len(self.price_history) < 20:
            return 1.0
        
        prices = np.array(list(self.price_history))
        recent_returns = np.diff(prices[-10:]) / prices[-10:-1]
        all_returns = np.diff(prices) / prices[:-1]
        
        recent_vol = np.std(recent_returns) if len(recent_returns) > 1 else 0
        hist_vol = np.std(all_returns) if len(all_returns) > 1 else recent_vol
        
        return recent_vol / hist_vol if hist_vol > 0 else 1.0
    
    def _calculate_momentum(self) -> float:
        """Calculate adaptive momentum"""
        if len(self.price_history) < 20:
            return 0
        
        prices = np.array(list(self.price_history))
        
        # Multiple timeframe momentum
        short_mom = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        medium_mom = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        long_mom = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Weighted momentum
        momentum = short_mom * 0.5 + medium_mom * 0.3 + long_mom * 0.2
        
        # Normalize
        momentum_std = np.std([short_mom, medium_mom, long_mom])
        if momentum_std > 0:
            return momentum / (momentum_std * 3)
        return momentum
    
    def _calculate_mean_reversion_strength(self) -> float:
        """Calculate mean reversion tendency"""
        if len(self.price_history) < 30:
            return 1.0
        
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 16:
            return 1.0
        
        var_1 = np.var(returns)
        var_2 = np.var(returns[::2] + returns[1::2]) / 2
        
        if var_1 == 0:
            return 1.0
        
        variance_ratio = var_2 / var_1
        return 2 - variance_ratio if variance_ratio < 2 else 0.1
    
    def _update_asset_characteristics(self):
        """Update learned asset characteristics"""
        if len(self.price_history) < 50:
            return
        
        prices = np.array(list(self.price_history))
        spreads = np.array(list(self.spread_history))
        
        self.asset_characteristics['typical_volatility'] = np.std(np.diff(prices) / prices[:-1])
        self.asset_characteristics['typical_spread'] = np.mean(spreads)

class GridStrategy:
    def __init__(self, 
                 exchange: Exchange, 
                 symbol: str, 
                 price_lower: float, 
                 price_upper: float,
                 grid_number: int,
                 investment: float,
                 take_profit_pnl: float,
                 stop_loss_pnl: float,
                 grid_id: str,
                 leverage: float = 20.0,
                 enable_grid_adaptation: bool = True,
                 enable_samig: bool = False):
        """Initialize enhanced grid strategy with optional SAMIG features"""
        
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        # Core parameters
        self.price_lower = float(price_lower)
        self.price_upper = float(price_upper)
        self.grid_number = int(grid_number)
        self.investment = float(investment)
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.leverage = float(leverage)
        self.enable_grid_adaptation = enable_grid_adaptation
        self.enable_samig = enable_samig
        
        # Base parameters for SAMIG reference
        self.base_price_lower = self.price_lower
        self.base_price_upper = self.price_upper
        self.base_grid_number = self.grid_number
        
        # Grid calculations
        self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
        self.investment_per_grid = self.investment / self.grid_number
        
        # SAMIG components (optional)
        if self.enable_samig:
            self.market_intelligence = MarketIntelligenceEngine(symbol)
            self.parameter_manager = AdaptiveParameterManager()
            self.performance_tracker = deque(maxlen=50)
            self.current_market_snapshot = None
            self.adaptation_count = 0
        
        # Grid state
        self.grid_orders = {}
        self.pnl = 0.0
        self.initial_investment = investment
        self.trades_count = 0
        self.running = False
        self.grid_adjustments_count = 0
        
        # Market information
        self._fetch_market_info()
    
    def _fetch_market_info(self):
        """Fetch market information for the trading pair"""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            self.price_precision = market_info['precision']['price']
            self.amount_precision = market_info['precision']['amount']
            self.min_amount = market_info['limits']['amount']['min']
            self.min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 0)
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            self.price_precision = 2
            self.amount_precision = 6
            self.min_amount = 0.0
            self.min_cost = 0.0
    
    def _calculate_dynamic_grid_parameters(self) -> Dict:
        """Calculate dynamic grid parameters using SAMIG if enabled"""
        if not self.enable_samig:
            return {
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'long_exposure_limit': self.investment,
                'short_exposure_limit': self.investment,
                'grid_interval': self.grid_interval,
                'order_size_multiplier': 1.0
            }
        
        try:
            # Get market intelligence
            market_snapshot = self.market_intelligence.analyze_market_conditions(self.exchange)
            self.current_market_snapshot = market_snapshot
            
            # Get adaptive parameters
            grid_density = self.parameter_manager.get_parameter('grid_density')
            trend_sensitivity = self.parameter_manager.get_parameter('trend_sensitivity')
            volatility_response = self.parameter_manager.get_parameter('volatility_response')
            momentum_threshold = self.parameter_manager.get_parameter('momentum_threshold')
            mean_reversion_factor = self.parameter_manager.get_parameter('mean_reversion_factor')
            exposure_asymmetry = self.parameter_manager.get_parameter('exposure_asymmetry')
            
            # Calculate dynamic parameters
            current_price = market_snapshot.price
            price_range = self.base_price_upper - self.base_price_lower
            
            # Volatility adjustment
            vol_adjustment = 1 + (market_snapshot.volatility_regime - 1) * volatility_response
            adjusted_range = price_range * vol_adjustment
            
            # Momentum-based center shift
            momentum_shift = market_snapshot.momentum * trend_sensitivity * price_range * 0.1
            grid_center = current_price + momentum_shift
            
            dynamic_price_lower = grid_center - adjusted_range / 2
            dynamic_price_upper = grid_center + adjusted_range / 2
            
            # Ensure positive prices
            if dynamic_price_lower <= 0:
                dynamic_price_lower = current_price * 0.5
                dynamic_price_upper = dynamic_price_lower + adjusted_range
            
            # Dynamic grid density
            if abs(market_snapshot.momentum) > momentum_threshold:
                dynamic_grid_number = max(3, int(self.base_grid_number * 0.7))
            else:
                dynamic_grid_number = int(self.base_grid_number * grid_density * mean_reversion_factor)
            
            # Asymmetric exposure limits
            base_exposure = self.investment
            
            if market_snapshot.order_book_imbalance > 0:
                long_exposure_limit = base_exposure * exposure_asymmetry
                short_exposure_limit = base_exposure / exposure_asymmetry
            elif market_snapshot.order_book_imbalance < 0:
                long_exposure_limit = base_exposure / exposure_asymmetry
                short_exposure_limit = base_exposure * exposure_asymmetry
            else:
                long_exposure_limit = short_exposure_limit = base_exposure
            
            # Momentum bias
            if market_snapshot.momentum > momentum_threshold:
                long_exposure_limit *= (1 + abs(market_snapshot.momentum))
            elif market_snapshot.momentum < -momentum_threshold:
                short_exposure_limit *= (1 + abs(market_snapshot.momentum))
            
            return {
                'price_lower': dynamic_price_lower,
                'price_upper': dynamic_price_upper,
                'grid_number': dynamic_grid_number,
                'long_exposure_limit': long_exposure_limit,
                'short_exposure_limit': short_exposure_limit,
                'grid_interval': (dynamic_price_upper - dynamic_price_lower) / dynamic_grid_number,
                'order_size_multiplier': 1 + market_snapshot.volatility_regime * 0.2
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating dynamic parameters: {e}")
            # Fallback to static parameters
            return {
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'long_exposure_limit': self.investment,
                'short_exposure_limit': self.investment,
                'grid_interval': self.grid_interval,
                'order_size_multiplier': 1.0
            }
    
    def _round_price(self, price: float) -> float:
        """Round price according to market precision"""
        try:
            if hasattr(self, 'price_precision') and isinstance(self.price_precision, int):
                decimals = self.price_precision
            else:
                if price < 0.1:
                    decimals = 8
                elif price < 10:
                    decimals = 6
                elif price < 1000:
                    decimals = 4
                else:
                    decimals = 2
            
            return round(price, decimals)
        except Exception as e:
            self.logger.error(f"Error rounding price {price}: {e}")
            return float(price)
    
    def _round_amount(self, amount: float) -> float:
        """Round amount according to market precision"""
        try:
            if hasattr(self, 'amount_precision') and isinstance(self.amount_precision, int):
                decimals = self.amount_precision
            else:
                decimals = 6
            
            return round(amount, decimals)
        except Exception as e:
            self.logger.error(f"Error rounding amount {amount}: {e}")
            return float(amount)
    
    def _calculate_grid_levels(self, dynamic_params: Dict = None) -> List[float]:
        """Calculate grid price levels"""
        try:
            if dynamic_params:
                price_lower = dynamic_params['price_lower']
                price_upper = dynamic_params['price_upper']
                grid_number = dynamic_params['grid_number']
            else:
                price_lower = self.price_lower
                price_upper = self.price_upper
                grid_number = self.grid_number
            
            grid_interval = (price_upper - price_lower) / grid_number
            levels = []
            
            for i in range(grid_number + 1):
                level = price_lower + (i * grid_interval)
                levels.append(self._round_price(level))
            
            return levels
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []
    
    def _calculate_order_amount(self, dynamic_params: Dict = None) -> float:
        """Calculate order amount per grid level"""
        try:
            multiplier = dynamic_params.get('order_size_multiplier', 1.0) if dynamic_params else 1.0
            
            ticker = self.exchange.get_ticker(self.symbol)
            price = float(ticker['last'])
            
            amount = (self.investment_per_grid * self.leverage * multiplier) / price
            amount = max(amount, self.min_amount)
            
            return self._round_amount(amount)
        except Exception as e:
            self.logger.error(f"Error calculating order amount: {e}")
            return self.min_amount
    
    def _get_directional_exposure(self) -> Dict[str, float]:
        """Get current directional exposure breakdown"""
        try:
            positions = self.exchange.get_positions(self.symbol)
            net_position_value = 0.0
            
            for position in positions:
                initial_margin = float(position.get('initialMargin', 0))
                side = position.get('side', '')
                
                if side == 'long':
                    net_position_value += initial_margin
                elif side == 'short':
                    net_position_value -= initial_margin
            
            open_orders = self.exchange.get_open_orders(self.symbol)
            potential_long_exposure = 0.0
            potential_short_exposure = 0.0
            
            for order in open_orders:
                if order['id'] in self.grid_orders:
                    if order['side'] == 'buy':
                        potential_long_exposure += self.investment_per_grid
                    elif order['side'] == 'sell':
                        potential_short_exposure += self.investment_per_grid
            
            total_potential_long = max(0, net_position_value) + potential_long_exposure
            total_potential_short = abs(min(0, net_position_value)) + potential_short_exposure
            
            return {
                'net_position_value': net_position_value,
                'current_long_value': max(0, net_position_value),
                'current_short_value': abs(min(0, net_position_value)),
                'potential_long_exposure': potential_long_exposure,
                'potential_short_exposure': potential_short_exposure,
                'total_potential_long': total_potential_long,
                'total_potential_short': total_potential_short,
                'remaining_long_budget': max(0, self.investment - total_potential_long),
                'remaining_short_budget': max(0, self.investment - total_potential_short)
            }
        except Exception as e:
            self.logger.error(f"Error getting directional exposure: {e}")
            return {
                'net_position_value': 0, 'current_long_value': 0, 'current_short_value': 0,
                'potential_long_exposure': 0, 'potential_short_exposure': 0,
                'total_potential_long': 0, 'total_potential_short': 0,
                'remaining_long_budget': self.investment, 'remaining_short_budget': self.investment
            }
    
    def _can_place_buy_order(self, dynamic_params: Dict = None) -> bool:
        """Check if we can place a buy order"""
        try:
            exposure = self._get_directional_exposure()
            limit = dynamic_params.get('long_exposure_limit', self.investment) if dynamic_params else self.investment
            return exposure['remaining_long_budget'] >= self.investment_per_grid and exposure['total_potential_long'] < limit
        except Exception as e:
            self.logger.error(f"Error checking buy order capability: {e}")
            return False
    
    def _can_place_sell_order(self, dynamic_params: Dict = None) -> bool:
        """Check if we can place a sell order"""
        try:
            exposure = self._get_directional_exposure()
            limit = dynamic_params.get('short_exposure_limit', self.investment) if dynamic_params else self.investment
            return exposure['remaining_short_budget'] >= self.investment_per_grid and exposure['total_potential_short'] < limit
        except Exception as e:
            self.logger.error(f"Error checking sell order capability: {e}")
            return False
    
    def setup_grid(self) -> None:
        """Setup the grid with enhanced adaptive capabilities"""
        try:
            # Get dynamic parameters
            dynamic_params = self._calculate_dynamic_grid_parameters()
            
            if self.enable_samig and self.current_market_snapshot:
                self.logger.info(f"SAMIG Setup - Market: Vol={self.current_market_snapshot.volatility_regime:.2f}, "
                               f"Momentum={self.current_market_snapshot.momentum:.3f}")
                self.logger.info(f"Dynamic: Grids={dynamic_params['grid_number']}, "
                               f"Range=[{dynamic_params['price_lower']:.6f}, {dynamic_params['price_upper']:.6f}]")
            
            # Update current parameters with dynamic values
            self.price_lower = dynamic_params['price_lower']
            self.price_upper = dynamic_params['price_upper']
            self.grid_number = dynamic_params['grid_number']
            self.grid_interval = dynamic_params['grid_interval']
            
            grid_levels = self._calculate_grid_levels(dynamic_params)
            amount = self._calculate_order_amount(dynamic_params)
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            self.logger.info(f"Setting up enhanced grid for {self.symbol}")
            self.logger.info(f"Price range: {self.price_lower:.6f} - {self.price_upper:.6f}")
            self.logger.info(f"Grid levels: {self.grid_number}, Current price: {current_price}")
            
            # Cancel existing orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Error cancelling orders: {e}")
            
            self.grid_orders = {}
            orders_placed = 0
            
            # Place initial grid orders
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Place buy orders below current price
                if buy_price < current_price and self._can_place_buy_order(dynamic_params):
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                        time.sleep(1)
                        
                        order_status = self.exchange.get_order_status(order['id'], self.symbol)
                        if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                            self.grid_orders[order['id']] = {
                                'type': 'buy',
                                'price': buy_price,
                                'amount': amount,
                                'status': 'open',
                                'grid_level': i
                            }
                            orders_placed += 1
                    except Exception as e:
                        self.logger.error(f"Failed to place buy order at {buy_price}: {e}")
                
                # Place sell orders above current price
                if sell_price > current_price and self._can_place_sell_order(dynamic_params):
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                        time.sleep(1)
                        
                        order_status = self.exchange.get_order_status(order['id'], self.symbol)
                        if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                            self.grid_orders[order['id']] = {
                                'type': 'sell',
                                'price': sell_price,
                                'amount': amount,
                                'status': 'open',
                                'grid_level': i + 1
                            }
                            orders_placed += 1
                    except Exception as e:
                        self.logger.error(f"Failed to place sell order at {sell_price}: {e}")
            
            if orders_placed > 0:
                self.running = True
                self.logger.info(f"Grid setup complete with {orders_placed} orders")
            else:
                self.running = False
                self.logger.warning("Grid setup failed - no orders placed")
        
        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.running = False
    
    def update_grid(self) -> None:
        """Update grid with SAMIG intelligence if enabled"""
        try:
            if not self.running:
                return
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # SAMIG regime change detection
            if self.enable_samig:
                market_snapshot = self.market_intelligence.analyze_market_conditions(self.exchange)
                
                if self._detect_regime_change(market_snapshot):
                    self.logger.info("SAMIG: Regime change detected - reconfiguring grid")
                    self.setup_grid()
                    return
                
                # Performance evaluation and adaptation
                self._evaluate_performance_and_adapt()
            
            # Standard grid adaptation
            elif self.enable_grid_adaptation and self._is_price_outside_grid(current_price):
                self._adapt_grid_to_price(current_price)
                return
            
            self._update_orders()
            self._check_tp_sl()
            
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _detect_regime_change(self, new_snapshot: MarketSnapshot) -> bool:
        """Detect significant market regime changes"""
        if not self.current_market_snapshot:
            return False
        
        old = self.current_market_snapshot
        new = new_snapshot
        
        # Calculate relative changes
        vol_change = abs(new.volatility_regime - old.volatility_regime) / (old.volatility_regime + 0.1)
        momentum_change = abs(new.momentum - old.momentum)
        mr_change = abs(new.mean_reversion_strength - old.mean_reversion_strength) / (old.mean_reversion_strength + 0.1)
        
        # Adaptive threshold
        typical_vol = self.market_intelligence.asset_characteristics.get('typical_volatility', 0.02)
        threshold = min(0.5, max(0.1, typical_vol * 10))
        
        regime_change = (vol_change > threshold or 
                        momentum_change > threshold or
                        mr_change > threshold)
        
        if regime_change:
            self.logger.info(f"Regime change: Vol Δ={vol_change:.3f}, Mom Δ={momentum_change:.3f}")
        
        return regime_change
    
    def _evaluate_performance_and_adapt(self):
        """Evaluate performance and adapt SAMIG parameters"""
        if not self.enable_samig or len(self.performance_tracker) < 10:
            return
        
        try:
            recent_pnl = sum([p.pnl for p in list(self.performance_tracker)[-10:]])
            performance_score = recent_pnl / self.investment * 100
            
            if self.current_market_snapshot:
                self.parameter_manager.update_parameter('grid_density', performance_score, self.current_market_snapshot)
                self.parameter_manager.update_parameter('trend_sensitivity', performance_score, self.current_market_snapshot)
                self.parameter_manager.update_parameter('volatility_response', performance_score, self.current_market_snapshot)
                
                self.adaptation_count += 1
                self.logger.info(f"SAMIG Adaptation #{self.adaptation_count}: Performance={performance_score:.2f}%")
        
        except Exception as e:
            self.logger.error(f"Error in performance evaluation: {e}")
    
    def _is_price_outside_grid(self, current_price: float) -> bool:
        """Check if price is outside grid boundaries"""
        buffer_size = (self.price_upper - self.price_lower) * 0.005
        return (current_price < (self.price_lower - buffer_size) or 
                current_price > (self.price_upper + buffer_size))
    
    def _adapt_grid_to_price(self, current_price: float) -> None:
        """Adapt grid to price movement"""
        try:
            self.logger.info(f"Grid adaptation: Price {current_price} outside range")
            
            grid_size = self.price_upper - self.price_lower
            self.price_lower = current_price - (grid_size / 2)
            self.price_upper = current_price + (grid_size / 2)
            
            if self.price_lower <= 0:
                self.price_lower = 0.00001
                self.price_upper = self.price_lower + grid_size
            
            self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
            
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            self.grid_orders = {}
            self.setup_grid()
            self.grid_adjustments_count += 1
            
        except Exception as e:
            self.logger.error(f"Error in grid adaptation: {e}")
    
    def _update_orders(self):
        """Update order status and handle fills"""
        try:
            open_orders = self.exchange.get_open_orders(self.symbol)
            current_order_ids = {order['id']: order for order in open_orders}
            
            for order_id in list(self.grid_orders.keys()):
                order_info = self.grid_orders[order_id]
                
                if order_id not in current_order_ids and order_info['status'] == 'open':
                    try:
                        order_status = self.exchange.get_order_status(order_id, self.symbol)
                        
                        if order_status['status'] in ['filled', 'closed']:
                            order_info['status'] = 'filled'
                            self.trades_count += 1
                            
                            # Track performance for SAMIG
                            if self.enable_samig:
                                self._track_performance()
                            
                            self._calculate_pnl()
                            self._place_counter_order(order_info)
                        
                        elif order_status['status'] == 'canceled':
                            order_info['status'] = 'cancelled'
                            
                    except Exception as e:
                        self.logger.error(f"Error checking order {order_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error updating orders: {e}")
    
    def _track_performance(self):
        """Track performance metrics for SAMIG learning"""
        if not self.enable_samig or not self.current_market_snapshot:
            return
        
        try:
            metric = GridPerformanceMetric(
                timestamp=time.time(),
                pnl=self.pnl,
                win_rate=self.trades_count / max(1, len(self.grid_orders)),
                trades_count=self.trades_count,
                market_conditions=self.current_market_snapshot,
                grid_config={
                    'price_lower': self.price_lower,
                    'price_upper': self.price_upper,
                    'grid_number': self.grid_number,
                    'leverage': self.leverage
                }
            )
            self.performance_tracker.append(metric)
        
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
    
    def _place_counter_order(self, filled_order: Dict) -> None:
        """Place counter order after fill"""
        try:
            grid_level = filled_order['grid_level']
            price = filled_order['price']
            amount = filled_order['amount']
            
            if filled_order['type'] == 'buy':
                sell_price = self._round_price(price + self.grid_interval)
                if sell_price <= self.price_upper and self._can_place_sell_order():
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                        self.grid_orders[order['id']] = {
                            'type': 'sell',
                            'price': sell_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': grid_level + 1
                        }
                    except Exception as e:
                        self.logger.error(f"Error placing counter sell order: {e}")
            
            elif filled_order['type'] == 'sell':
                buy_price = self._round_price(price - self.grid_interval)
                if buy_price >= self.price_lower and self._can_place_buy_order():
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                        self.grid_orders[order['id']] = {
                            'type': 'buy',
                            'price': buy_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': grid_level - 1
                        }
                    except Exception as e:
                        self.logger.error(f"Error placing counter buy order: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in counter order placement: {e}")
    
    def _calculate_pnl(self) -> None:
        """Calculate PnL from positions"""
        try:
            positions = self.exchange.get_positions(self.symbol)
            unrealized_pnl = sum(float(pos.get('unrealizedPnl', 0)) for pos in positions)
            self.pnl = unrealized_pnl
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
    
    def _check_tp_sl(self) -> None:
        """Check take profit and stop loss conditions"""
        try:
            pnl_percentage = (self.pnl / self.initial_investment) * 100
            
            if pnl_percentage >= self.take_profit_pnl:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}%")
                self.stop_grid()
            elif pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}%")
                self.stop_grid()
        except Exception as e:
            self.logger.error(f"Error checking TP/SL: {e}")
    
    def stop_grid(self) -> None:
        """Stop the grid strategy"""
        try:
            if not self.running:
                return
            
            self.logger.info(f"Stopping grid for {self.symbol}")
            
            # Cancel orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                for order_id in self.grid_orders:
                    if self.grid_orders[order_id]['status'] == 'open':
                        self.grid_orders[order_id]['status'] = 'cancelled'
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            # Close positions
            try:
                positions = self.exchange.get_positions(self.symbol)
                for position in positions:
                    if abs(float(position['contracts'])) > 0:
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        self.exchange.create_market_order(self.symbol, side, abs(float(position['contracts'])))
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
            
            self.running = False
            self.logger.info(f"Grid stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.running = False
    
    def get_status(self) -> Dict:
        """Get comprehensive grid status including SAMIG metrics"""
        try:
            exposure = self._get_directional_exposure()
            
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol,
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'grid_interval': self.grid_interval,
                'investment': self.investment,
                'investment_per_grid': self.investment_per_grid,
                'current_long_value': exposure['current_long_value'],
                'current_short_value': exposure['current_short_value'],
                'remaining_long_budget': exposure['remaining_long_budget'],
                'remaining_short_budget': exposure['remaining_short_budget'],
                'take_profit_pnl': self.take_profit_pnl,
                'stop_loss_pnl': self.stop_loss_pnl,
                'leverage': self.leverage,
                'enable_grid_adaptation': self.enable_grid_adaptation,
                'enable_samig': self.enable_samig,
                'grid_adjustments_count': self.grid_adjustments_count,
                'pnl': self.pnl,
                'pnl_percentage': (self.pnl / self.initial_investment) * 100 if self.initial_investment else 0,
                'trades_count': self.trades_count,
                'running': self.running,
                'orders_count': len([o for o in self.grid_orders.values() if o.get('status') == 'open'])
            }
            
            # Add SAMIG-specific metrics
            if self.enable_samig and self.current_market_snapshot:
                status.update({
                    'samig_active': True,
                    'adaptation_count': self.adaptation_count,
                    'volatility_regime': self.current_market_snapshot.volatility_regime,
                    'momentum': self.current_market_snapshot.momentum,
                    'mean_reversion_strength': self.current_market_snapshot.mean_reversion_strength,
                    'order_book_imbalance': self.current_market_snapshot.order_book_imbalance,
                    'grid_density': self.parameter_manager.get_parameter('grid_density'),
                    'trend_sensitivity': self.parameter_manager.get_parameter('trend_sensitivity'),
                    'volatility_response': self.parameter_manager.get_parameter('volatility_response')
                })
            else:
                status['samig_active'] = False
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {}