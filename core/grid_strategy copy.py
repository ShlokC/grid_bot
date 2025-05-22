"""
Grid trading strategy implementation with proper directional exposure management.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
import traceback
from core.exchange import Exchange

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
                 enable_grid_adaptation: bool = True):
        """
        Initialize the grid strategy with proper directional exposure management.
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        self.logger.debug(f"Initialized GridStrategy with symbol: {self.original_symbol}, ID: {self.symbol}")
        
        self.price_lower = float(price_lower)
        self.price_upper = float(price_upper)
        self.grid_number = int(grid_number)
        self.investment = float(investment)
        self.take_profit_pnl = float(take_profit_pnl)
        self.stop_loss_pnl = float(stop_loss_pnl)
        self.grid_id = grid_id
        self.leverage = float(leverage)
        self.enable_grid_adaptation = enable_grid_adaptation
        
        # Calculate grid parameters
        self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
        self.original_grid_interval = self.grid_interval
        
        # Investment per grid level (used for order sizing)
        self.investment_per_grid = self.investment / self.grid_number
        
        # Store grid orders
        self.grid_orders = {}
        
        # Performance metrics
        self.pnl = 0.0
        self.initial_investment = investment
        self.trades_count = 0
        self.running = False
        self.grid_adjustments_count = 0
        
        # Market information
        self._fetch_market_info()
        
    def _fetch_market_info(self):
        """Fetch market information for the trading pair."""
        try:
            market_info = self.exchange.get_market_info(self.symbol)
            self.price_precision = market_info['precision']['price']
            self.amount_precision = market_info['precision']['amount']
            self.min_amount = market_info['limits']['amount']['min']
            self.min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 0)
            self.logger.debug(f"Fetched market info for {self.symbol}: precision={self.price_precision}, min_amount={self.min_amount}")
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            self.price_precision = 2
            self.amount_precision = 6
            self.min_amount = 0.0
            self.min_cost = 0.0
        
    def _round_price(self, price: float) -> float:
        """Round price according to market precision."""
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
            
            format_str = f"{{:.{decimals}f}}"
            formatted_price = format_str.format(price)
            precise_price = float(formatted_price)
            
            self.logger.debug(f"Rounded price: {price} → {precise_price} (using {decimals} decimals)")
            return precise_price
        except Exception as e:
            self.logger.error(f"Error rounding price {price}: {e}")
            return float(price)
    
    def _round_amount(self, amount: float) -> float:
        """Round amount according to market precision."""
        try:
            if hasattr(self, 'amount_precision') and isinstance(self.amount_precision, int):
                decimals = self.amount_precision
            else:
                if amount < 0.001:
                    decimals = 8
                elif amount < 1:
                    decimals = 6 
                elif amount < 1000:
                    decimals = 4
                else:
                    decimals = 2
            
            format_str = f"{{:.{decimals}f}}"
            formatted_amount = format_str.format(amount)
            precise_amount = float(formatted_amount)
            
            self.logger.debug(f"Rounded amount: {amount} → {precise_amount} (using {decimals} decimals)")
            return precise_amount
        except Exception as e:
            self.logger.error(f"Error rounding amount {amount}: {e}")
            return float(amount)
    
    def _calculate_grid_levels(self) -> List[float]:
        """Calculate grid price levels with proper precision."""
        try:
            price_lower = float(self.price_lower)
            price_upper = float(self.price_upper)
            grid_number = int(self.grid_number)
            
            grid_interval = (price_upper - price_lower) / grid_number
            levels = []
            
            if price_lower < 0.1:
                format_str = "{:.8f}"
            elif price_lower < 10:
                format_str = "{:.6f}"
            elif price_lower < 1000:
                format_str = "{:.4f}"
            else:
                format_str = "{:.2f}"
                
            self.logger.debug(f"Using price format: {format_str} for price range {price_lower}-{price_upper}")
            
            for i in range(grid_number + 1):
                exact_level = price_lower + (i * grid_interval)
                formatted_level = format_str.format(exact_level)
                precise_level = float(formatted_level)
                levels.append(precise_level)
                self.logger.debug(f"Grid level {i}: {exact_level} → {precise_level}")
                
            if levels and len(levels) > 1:
                levels[0] = price_lower
                levels[-1] = price_upper
                
            return levels
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _calculate_order_amount(self) -> float:
        """Calculate order amount for each grid level based on investment per grid."""
        try:
            leverage = float(self.leverage)
            
            # Get current price
            ticker = self.exchange.get_ticker(self.symbol)
            price = float(ticker['last'])
            self.logger.info(f"Current price for {self.symbol}: {price}")
            
            # Calculate amount based on investment per grid level
            amount = (self.investment_per_grid * leverage) / price
            self.logger.info(f"Order amount per grid (investment: {self.investment_per_grid}, leverage: {leverage}x): {amount}")
            
            # Ensure amount meets minimum requirements
            min_amount = self.min_amount if hasattr(self, 'min_amount') and self.min_amount > 0 else 0.00001
            amount = max(amount, min_amount)
            
            # Round amount according to market precision
            rounded_amount = self._round_amount(amount)
            self.logger.info(f"Final rounded order amount: {rounded_amount}")
            
            return rounded_amount
        except Exception as e:
            self.logger.error(f"Error calculating order amount: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self.min_amount if hasattr(self, 'min_amount') and self.min_amount > 0 else 0.00001
    
    def _get_directional_exposure(self) -> Dict[str, float]:
        """Get current directional exposure breakdown."""
        try:
            # Get current net position value
            positions = self.exchange.get_positions(self.symbol)
            net_position_value = 0.0
            
            for position in positions:
                initial_margin = float(position.get('initialMargin', 0))
                side = position.get('side', '')
                
                if side == 'long':
                    net_position_value += initial_margin
                elif side == 'short':
                    net_position_value -= initial_margin
            
            # Get potential exposure from open orders
            open_orders = self.exchange.get_open_orders(self.symbol)
            potential_long_exposure = 0.0
            potential_short_exposure = 0.0
            
            for order in open_orders:
                if order['id'] in self.grid_orders:
                    if order['side'] == 'buy':
                        potential_long_exposure += self.investment_per_grid
                    elif order['side'] == 'sell':
                        potential_short_exposure += self.investment_per_grid
            
            # Calculate total potential exposure
            total_potential_long = max(0, net_position_value) + potential_long_exposure
            total_potential_short = abs(min(0, net_position_value)) + potential_short_exposure
            
            exposure = {
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
            
            self.logger.debug(f"Directional exposure: Long=${exposure['current_long_value']:.2f}+${exposure['potential_long_exposure']:.2f}=${exposure['total_potential_long']:.2f}, Short=${exposure['current_short_value']:.2f}+${exposure['potential_short_exposure']:.2f}=${exposure['total_potential_short']:.2f}")
            
            return exposure
        except Exception as e:
            self.logger.error(f"Error getting directional exposure: {e}")
            return {
                'net_position_value': 0, 'current_long_value': 0, 'current_short_value': 0,
                'potential_long_exposure': 0, 'potential_short_exposure': 0,
                'total_potential_long': 0, 'total_potential_short': 0,
                'remaining_long_budget': self.investment, 'remaining_short_budget': self.investment
            }
    
    def _can_place_buy_order(self) -> bool:
        """Check if we can place a buy order without exceeding long exposure limit."""
        try:
            exposure = self._get_directional_exposure()
            can_place = exposure['remaining_long_budget'] >= self.investment_per_grid
            
            if not can_place:
                self.logger.debug(f"Cannot place buy order: remaining long budget ${exposure['remaining_long_budget']:.2f} < required ${self.investment_per_grid:.2f}")
            
            return can_place
        except Exception as e:
            self.logger.error(f"Error checking if can place buy order: {e}")
            return False
    
    def _can_place_sell_order(self) -> bool:
        """Check if we can place a sell order without exceeding short exposure limit."""
        try:
            exposure = self._get_directional_exposure()
            can_place = exposure['remaining_short_budget'] >= self.investment_per_grid
            
            if not can_place:
                self.logger.debug(f"Cannot place sell order: remaining short budget ${exposure['remaining_short_budget']:.2f} < required ${self.investment_per_grid:.2f}")
            
            return can_place
        except Exception as e:
            self.logger.error(f"Error checking if can place sell order: {e}")
            return False
    
    def _cancel_existing_orders(self):
        """Cancel any existing orders for this symbol."""
        try:
            open_orders = self.exchange.get_open_orders(self.symbol)
            
            if open_orders:
                self.logger.info(f"Found {len(open_orders)} existing open orders for {self.symbol}. Cancelling...")
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
                
                open_orders_after = self.exchange.get_open_orders(self.symbol)
                if open_orders_after:
                    self.logger.warning(f"Some orders ({len(open_orders_after)}) still remain after cancellation attempt.")
                else:
                    self.logger.info("All existing orders successfully cancelled.")
                    
        except Exception as e:
            self.logger.error(f"Error cancelling existing orders: {str(e)}")

    def setup_grid(self) -> None:
        """Setup the initial grid orders with proper directional exposure management."""
        try:
            self.price_lower = float(self.price_lower)
            self.price_upper = float(self.price_upper)
            self.grid_number = int(self.grid_number)
            self.investment = float(self.investment)
            
            grid_levels = self._calculate_grid_levels()
            amount = self._calculate_order_amount()
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            self.logger.info(f"Setting up grid for {self.symbol} with {self.grid_number} levels")
            self.logger.info(f"Price range: {self.price_lower} - {self.price_upper}")
            self.logger.info(f"Current price: {current_price}")
            self.logger.info(f"Investment per grid: {self.investment_per_grid}")
            self.logger.info(f"Total investment budget: {self.investment}")
            
            # Check current directional exposure
            exposure = self._get_directional_exposure()
            self.logger.info(f"Current exposure: Long=${exposure['current_long_value']:.2f}, Short=${exposure['current_short_value']:.2f}")
            self.logger.info(f"Remaining budgets: Long=${exposure['remaining_long_budget']:.2f}, Short=${exposure['remaining_short_budget']:.2f}")
            
            self._cancel_existing_orders()
            self.grid_orders = {}
            
            orders_placed = 0
            retry_count = 3
            
            # Place initial grid orders with directional limits
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Place buy orders below current price (if long budget allows)
                if buy_price < current_price and self._can_place_buy_order():
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing buy order at price {buy_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'buy', amount, buy_price
                            )
                            
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
                                self.logger.info(f"Confirmed buy order at level {i}: {buy_price}, ID: {order['id']}")
                                order_placed = True
                                orders_placed += 1
                            else:
                                self.logger.warning(f"Order status for buy order at {buy_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place buy order at level {i} price {buy_price}: {str(e)}")
                            retry += 1
                            time.sleep(2)
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place buy order at level {i} after {retry_count} attempts")
                elif buy_price < current_price:
                    self.logger.info(f"Skipping buy order at {buy_price} - long exposure limit reached")
                
                # Place sell orders above current price (if short budget allows)
                if sell_price > current_price and self._can_place_sell_order():
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing sell order at price {sell_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'sell', amount, sell_price
                            )
                            
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
                                self.logger.info(f"Confirmed sell order at level {i+1}: {sell_price}, ID: {order['id']}")
                                order_placed = True
                                orders_placed += 1
                            else:
                                self.logger.warning(f"Order status for sell order at {sell_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place sell order at level {i+1} price {sell_price}: {str(e)}")
                            retry += 1
                            time.sleep(2)
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place sell order at level {i+1} after {retry_count} attempts")
                elif sell_price > current_price:
                    self.logger.info(f"Skipping sell order at {sell_price} - short exposure limit reached")
            
            # Final exposure report
            final_exposure = self._get_directional_exposure()
            self.logger.info(f"Grid setup complete. Placed {orders_placed} orders.")
            self.logger.info(f"Final exposure: Long=${final_exposure['total_potential_long']:.2f}, Short=${final_exposure['total_potential_short']:.2f}")
            
            open_orders = [o for o in self.grid_orders.values() if o['status'] == 'open']
            if len(open_orders) > 0:
                self.running = True
                self.logger.info(f"Grid is now running with {len(open_orders)} active orders.")
            else:
                self.running = False
                self.logger.warning("Grid setup failed to place any orders. Grid is not running.")
        
        except Exception as e:
            self.logger.error(f"Error setting up grid: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.running = False

    def update_grid(self) -> None:
        """Update grid orders and check for grid adaptation if needed."""
        try:
            if not self.running:
                return
            
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            if self.enable_grid_adaptation and self._is_price_outside_grid(current_price):
                self._adapt_grid_to_price(current_price)
                return
            
            self._update_orders()
            self._check_tp_sl()
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _is_price_outside_grid(self, current_price: float) -> bool:
        """Check if price is outside the grid boundaries."""
        buffer_size = (self.price_upper - self.price_lower) * 0.005
        
        if current_price < (self.price_lower - buffer_size) or current_price > (self.price_upper + buffer_size):
            self.logger.info(f"Price {current_price} is outside grid range [{self.price_lower} - {self.price_upper}]")
            return True
        return False
    
    def _adapt_grid_to_price(self, current_price: float) -> None:
        """Adapt the grid to follow the price movement while preserving directional exposure management."""
        try:
            self.logger.info(f"GRID ADAPTATION TRIGGERED: Price {current_price} moved outside boundaries")
            
            # Log current exposure before adaptation
            pre_exposure = self._get_directional_exposure()
            self.logger.info(f"Pre-adaptation exposure: Long=${pre_exposure['current_long_value']:.2f}, Short=${pre_exposure['current_short_value']:.2f}")
            
            old_lower = self.price_lower
            old_upper = self.price_upper
            
            # Calculate new grid boundaries centered around current price
            grid_size = self.price_upper - self.price_lower
            grid_center = current_price
            
            new_lower = grid_center - (grid_size / 2)
            new_upper = grid_center + (grid_size / 2)
            
            # Ensure positive prices
            if new_lower <= 0:
                new_lower = 0.00001
                new_upper = new_lower + grid_size
            
            self.logger.info(f"GRID SHIFT: [{old_lower:.6f} - {old_upper:.6f}] → [{new_lower:.6f} - {new_upper:.6f}]")
            
            # Update grid parameters
            self.price_lower = new_lower
            self.price_upper = new_upper
            self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
            
            # Cancel all existing orders (this frees up order budget but keeps positions)
            self.logger.info("Cancelling all existing orders before grid adaptation")
            try:
                self.exchange.cancel_all_orders(self.symbol)
                time.sleep(2)
                self.logger.info("Orders cancelled - order budget freed for new grid")
            except Exception as cancel_error:
                self.logger.error(f"Error cancelling orders during grid adaptation: {cancel_error}")
            
            # Setup new grid with adapted boundaries
            # This will respect directional exposure limits based on existing positions
            self.logger.info("Setting up NEW GRID with adapted boundaries and directional limits")
            
            old_grid_orders = self.grid_orders.copy()
            self.grid_orders = {}
            
            try:
                # Log exposure available for new grid
                adaptation_exposure = self._get_directional_exposure()
                self.logger.info(f"Available for new grid: Long budget=${adaptation_exposure['remaining_long_budget']:.2f}, Short budget=${adaptation_exposure['remaining_short_budget']:.2f}")
                
                # Setup new grid (this respects directional limits)
                self.setup_grid()
                
                self.grid_adjustments_count += 1
                
                # Log successful adaptation
                post_exposure = self._get_directional_exposure()
                self.logger.info(f"GRID ADAPTATION #{self.grid_adjustments_count} COMPLETE")
                self.logger.info(f"Post-adaptation exposure: Long=${post_exposure['total_potential_long']:.2f}, Short=${post_exposure['total_potential_short']:.2f}")
                
            except Exception as setup_error:
                self.logger.error(f"Failed to set up adapted grid: {setup_error}")
                
                # Rollback to old grid parameters
                self.price_lower = old_lower
                self.price_upper = old_upper
                self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
                self.grid_orders = old_grid_orders
                
                self.logger.warning("ROLLBACK: Restored previous grid parameters after adaptation failure")
        
        except Exception as e:
            self.logger.error(f"Error in grid adaptation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_orders(self):
        """Update order status and handle filled orders."""
        open_orders = self.exchange.get_open_orders(self.symbol)
        current_order_ids = {order['id']: order for order in open_orders}
        
        active_buy_levels = set()
        active_sell_levels = set()
        
        for order_id in list(self.grid_orders.keys()):
            order_info = self.grid_orders[order_id]
            grid_level = order_info['grid_level']
            
            if order_id in current_order_ids:
                if order_info['status'] != 'open':
                    order_info['status'] = 'open'
                
                if order_info['type'] == 'buy':
                    active_buy_levels.add(grid_level)
                else:
                    active_sell_levels.add(grid_level)
            
            elif order_info['status'] == 'open':
                try:
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    
                    if order_status['status'] in ['filled', 'closed']:
                        order_info['status'] = order_status['status']
                        self.trades_count += 1
                        
                        fill_price = float(order_status.get('average', order_info['price']))
                        self.logger.info(f"Order filled: {order_id} ({order_info['type']} at {fill_price})")
                        
                        # Log directional exposure after order fill
                        exposure = self._get_directional_exposure()
                        self.logger.info(f"After fill - Long: ${exposure['current_long_value']:.2f}, Short: ${exposure['current_short_value']:.2f}")
                        
                        # Calculate PnL from position changes
                        self._calculate_pnl()
                        
                        # Place counter order with directional limits
                        self._place_counter_order_with_confirmation(order_info)
                    
                    elif order_status['status'] == 'canceled':
                        order_info['status'] = 'cancelled'
                        self.logger.info(f"Order cancelled: {order_id}")
                    
                    else:
                        self.logger.warning(f"Unexpected order status for {order_id}: {order_status['status']}")
                        
                except Exception as e:
                    self.logger.error(f"Error checking order {order_id} status: {e}")
        
        # Ensure grid coverage with directional limits
        self._ensure_grid_coverage(active_buy_levels, active_sell_levels)
    
    def _place_counter_order_with_confirmation(self, filled_order: Dict) -> None:
        """Place a counter order after an order is filled, respecting directional limits."""
        try:
            grid_level = filled_order['grid_level']
            price = filled_order['price']
            amount = filled_order['amount']
            retry_count = 3
            
            if filled_order['type'] == 'buy':
                # Buy order filled (long position created) -> place sell order above it
                sell_price = self._round_price(price + self.grid_interval)
                
                if sell_price <= self.price_upper and self._can_place_sell_order():
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing counter sell order at {sell_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'sell', amount, sell_price
                            )
                            
                            time.sleep(1)
                            order_status = self.exchange.get_order_status(order['id'], self.symbol)
                            
                            if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                                self.grid_orders[order['id']] = {
                                    'type': 'sell',
                                    'price': sell_price,
                                    'amount': amount,
                                    'status': 'open',
                                    'grid_level': grid_level + 1
                                }
                                self.logger.info(f"Confirmed counter sell order at {sell_price}, ID: {order['id']}")
                                order_placed = True
                            else:
                                self.logger.warning(f"Order status for counter sell order at {sell_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place counter sell order at {sell_price}: {e}")
                            retry += 1
                            time.sleep(2)
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place counter sell order after {retry_count} attempts")
                elif sell_price > self.price_upper:
                    self.logger.info(f"Counter sell order at {sell_price} exceeds upper price limit")
                else:
                    self.logger.info(f"Counter sell order skipped - short exposure limit reached")
            
            elif filled_order['type'] == 'sell':
                # Sell order filled (position closed/short created) -> place buy order below it
                buy_price = self._round_price(price - self.grid_interval)
                
                if buy_price >= self.price_lower and self._can_place_buy_order():
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing counter buy order at {buy_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'buy', amount, buy_price
                            )
                            
                            time.sleep(1)
                            order_status = self.exchange.get_order_status(order['id'], self.symbol)
                            
                            if order_status and order_status['status'] in ['open', 'new', 'partially_filled']:
                                self.grid_orders[order['id']] = {
                                    'type': 'buy',
                                    'price': buy_price,
                                    'amount': amount,
                                    'status': 'open',
                                    'grid_level': grid_level - 1
                                }
                                self.logger.info(f"Confirmed counter buy order at {buy_price}, ID: {order['id']}")
                                order_placed = True
                            else:
                                self.logger.warning(f"Order status for counter buy order at {buy_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place counter buy order at {buy_price}: {e}")
                            retry += 1
                            time.sleep(2)
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place counter buy order after {retry_count} attempts")
                elif buy_price < self.price_lower:
                    self.logger.info(f"Counter buy order at {buy_price} below lower price limit")
                else:
                    self.logger.info(f"Counter buy order skipped - long exposure limit reached")
        
        except Exception as e:
            self.logger.error(f"Error in _place_counter_order_with_confirmation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _calculate_pnl(self) -> None:
        """Calculate PnL from actual position changes."""
        try:
            positions = self.exchange.get_positions(self.symbol)
            unrealized_pnl = 0.0
            
            for position in positions:
                unrealized_pnl += float(position.get('unrealizedPnl', 0))
            
            self.pnl = unrealized_pnl
            pnl_percentage = (self.pnl / self.initial_investment) * 100 if self.initial_investment > 0 else 0
            
            self.logger.debug(f"Current PnL: {self.pnl:.6f} ({pnl_percentage:.4f}%)")
            
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
    
    def _ensure_grid_coverage(self, active_buy_levels: Set[int], active_sell_levels: Set[int]) -> None:
        """Ensure grid coverage by replacing missing orders, respecting directional limits."""
        try:
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            grid_levels = self._calculate_grid_levels()
            amount = self._calculate_order_amount()
            
            if not grid_levels or len(grid_levels) <= 1:
                return
                
            self.logger.debug(f"Checking grid coverage. Current price: {current_price}")
            self.logger.debug(f"Active buy levels: {active_buy_levels}")
            self.logger.debug(f"Active sell levels: {active_sell_levels}")
            
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Check for missing buy orders below current price (if long budget allows)
                if (buy_price < current_price and 
                    i not in active_buy_levels and 
                    self._can_place_buy_order()):
                    
                    self.logger.info(f"Missing buy order at level {i} (price: {buy_price}). Placing order...")
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'buy', amount, buy_price)
                        self.grid_orders[order['id']] = {
                            'type': 'buy',
                            'price': buy_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': i
                        }
                        self.logger.info(f"Placed missing buy order at level {i}: {buy_price}, ID: {order['id']}")
                    except Exception as e:
                        self.logger.error(f"Failed to place missing buy order at level {i}: {e}")
                
                # Check for missing sell orders above current price (if short budget allows)
                if (sell_price > current_price and 
                    (i + 1) not in active_sell_levels and 
                    self._can_place_sell_order()):
                    
                    self.logger.info(f"Missing sell order at level {i+1} (price: {sell_price}). Placing order...")
                    try:
                        order = self.exchange.create_limit_order(self.symbol, 'sell', amount, sell_price)
                        self.grid_orders[order['id']] = {
                            'type': 'sell',
                            'price': sell_price,
                            'amount': amount,
                            'status': 'open',
                            'grid_level': i + 1
                        }
                        self.logger.info(f"Placed missing sell order at level {i+1}: {sell_price}, ID: {order['id']}")
                    except Exception as e:
                        self.logger.error(f"Failed to place missing sell order at level {i+1}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error ensuring grid coverage: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _check_tp_sl(self) -> None:
        """Check if take profit or stop loss conditions are met."""
        try:
            pnl_percentage = (self.pnl / self.initial_investment) * 100
            if pnl_percentage >= self.take_profit_pnl:
                self.logger.info(f"Take profit reached: {pnl_percentage:.2f}% >= {self.take_profit_pnl:.2f}%")
                self.stop_grid()
                return
            if pnl_percentage <= -self.stop_loss_pnl:
                self.logger.info(f"Stop loss reached: {pnl_percentage:.2f}% <= -{self.stop_loss_pnl:.2f}%")
                self.stop_grid()
                return
        except Exception as e:
            self.logger.error(f"Error in _check_tp_sl: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def stop_grid(self) -> None:
        """Stop the grid strategy and cancel all orders."""
        try:
            if not self.running:
                self.logger.info(f"Grid is already stopped for {self.symbol}")
                return
                
            self.logger.info(f"Stopping grid strategy for {self.symbol}")
            
            try:
                self.exchange.cancel_all_orders(self.symbol)
                self.logger.info(f"All orders cancelled for {self.symbol}")
                
                for order_id in self.grid_orders:
                    if self.grid_orders[order_id]['status'] == 'open':
                        self.grid_orders[order_id]['status'] = 'cancelled'
                        
            except Exception as e:
                self.logger.error(f"Error cancelling orders for {self.symbol}: {e}")
            
            try:
                positions = self.exchange.get_positions(self.symbol)
                for position in positions:
                    if abs(float(position['contracts'])) > 0:
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        try:
                            self.exchange.create_market_order(
                                self.symbol, 
                                side, 
                                abs(float(position['contracts']))
                            )
                            self.logger.info(f"Closed position: {position['side']} {position['contracts']}")
                        except Exception as e:
                            self.logger.error(f"Error closing position {position['side']}: {e}")
                    
            except Exception as e:
                self.logger.error(f"Error getting positions for {self.symbol}: {e}")

            self.running = False
            self.logger.info(f"Grid strategy stopped for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.running = False
        
    def get_status(self) -> Dict:
        """Get the current status of the grid strategy."""
        try:
            exposure = self._get_directional_exposure()
            
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol if hasattr(self, 'original_symbol') else self.symbol,
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'grid_interval': self.grid_interval,
                'investment': self.investment,
                'investment_per_grid': self.investment_per_grid,
                'current_long_value': exposure['current_long_value'],
                'current_short_value': exposure['current_short_value'],
                'potential_long_exposure': exposure['potential_long_exposure'],
                'potential_short_exposure': exposure['potential_short_exposure'],
                'total_potential_long': exposure['total_potential_long'],
                'total_potential_short': exposure['total_potential_short'],
                'remaining_long_budget': exposure['remaining_long_budget'],
                'remaining_short_budget': exposure['remaining_short_budget'],
                'take_profit_pnl': self.take_profit_pnl,
                'stop_loss_pnl': self.stop_loss_pnl,
                'leverage': self.leverage,
                'enable_grid_adaptation': self.enable_grid_adaptation,
                'grid_adjustments_count': self.grid_adjustments_count,
                'pnl': self.pnl,
                'pnl_percentage': (self.pnl / self.initial_investment) * 100 if self.initial_investment else 0,
                'trades_count': self.trades_count,
                'running': self.running,
                'orders_count': len([o for o in self.grid_orders.values() if o['status'] == 'open'])
            }
            return status
        except Exception as e:
            self.logger.error(f"Error getting grid status: {e}")
            return {}