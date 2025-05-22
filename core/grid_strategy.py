"""
Grid trading strategy implementation.
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
                 take_profit_pnl: float,  # Take profit PnL percentage
                 stop_loss_pnl: float,    # Stop loss PnL percentage
                 grid_id: str,
                 leverage: float = 20.0,   # Default to 1x leverage
                 enable_grid_adaptation: bool = True):  # Enable grid to follow price movements
        """
        Initialize the grid strategy.
        
        Args:
            exchange: Exchange instance
            symbol: Trading pair (e.g., 'BTC/USDT')
            price_lower: Lower price boundary
            price_upper: Upper price boundary
            grid_number: Number of grid levels
            investment: Total investment amount
            take_profit_pnl: Take profit PnL percentage
            stop_loss_pnl: Stop loss PnL percentage
            grid_id: Unique identifier for this grid
            leverage: Trading leverage (e.g., 1.0, 10.0, etc.)
            enable_grid_adaptation: Whether to adapt grid when price moves outside boundaries
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        # Store original symbol for display and the ID for API calls
        self.original_symbol = symbol
        self.symbol = exchange._get_symbol_id(symbol) if hasattr(exchange, '_get_symbol_id') else symbol
        
        self.logger.debug(f"Initialized GridStrategy with symbol: {self.original_symbol}, ID: {self.symbol}")
        
        # Convert all numeric inputs to proper types
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
        self.original_grid_interval = self.grid_interval  # Store original interval for reference
        
        # Store grid orders and positions
        self.grid_orders = {}  # order_id -> order_info
        self.positions = {}    # position_id -> position_info
        
        # Performance metrics
        self.pnl = 0.0
        self.initial_investment = investment
        self.trades_count = 0
        self.running = False
        self.grid_adjustments_count = 0  # Track number of grid adjustments made
        
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
        """
        Round price according to market precision using string formatting.
        This avoids problems with the built-in round() function.
        """
        try:
            # Determine appropriate precision for the price
            # Start with exchange-specified precision if available
            if hasattr(self, 'price_precision') and isinstance(self.price_precision, int):
                # Use exchange-defined precision
                decimals = self.price_precision
            else:
                # Default to asset-appropriate precision if exchange info not available
                if price < 0.1:
                    # Very low-priced assets need more precision
                    decimals = 8
                elif price < 10:
                    # Low-priced assets need medium precision
                    decimals = 6
                elif price < 1000:
                    # Medium-priced assets need less precision
                    decimals = 4
                else:
                    # High-priced assets need minimal precision
                    decimals = 2
            
            # Create format string with appropriate precision
            format_str = f"{{:.{decimals}f}}"
            
            # Format to string with specified precision then back to float
            formatted_price = format_str.format(price)
            precise_price = float(formatted_price)
            
            self.logger.debug(f"Rounded price: {price} → {precise_price} (using {decimals} decimals)")
            return precise_price
        except Exception as e:
            self.logger.error(f"Error rounding price {price}: {e}")
            # Fallback - return original value but ensure it's a float
            return float(price)
    
    def _round_amount(self, amount: float) -> float:
        """
        Round amount according to market precision using string formatting.
        This avoids problems with the built-in round() function.
        """
        try:
            # Determine appropriate precision for the amount
            # Start with exchange-specified precision if available
            if hasattr(self, 'amount_precision') and isinstance(self.amount_precision, int):
                # Use exchange-defined precision
                decimals = self.amount_precision
            else:
                # Default to asset-appropriate precision if exchange info not available
                if amount < 0.001:
                    # Very small amounts (like BTC) need more precision
                    decimals = 8
                elif amount < 1:
                    # Small amounts need medium precision
                    decimals = 6 
                elif amount < 1000:
                    # Medium amounts need less precision
                    decimals = 4
                else:
                    # Large amounts need minimal precision
                    decimals = 2
            
            # Create format string with appropriate precision
            format_str = f"{{:.{decimals}f}}"
            
            # Format to string with specified precision then back to float
            formatted_amount = format_str.format(amount)
            precise_amount = float(formatted_amount)
            
            self.logger.debug(f"Rounded amount: {amount} → {precise_amount} (using {decimals} decimals)")
            return precise_amount
        except Exception as e:
            self.logger.error(f"Error rounding amount {amount}: {e}")
            # Fallback - return original value but ensure it's a float
            return float(amount)
    
    def _calculate_grid_levels(self) -> List[float]:
        """Calculate grid price levels with proper precision."""
        try:
            # Ensure we're working with the correct types
            price_lower = float(self.price_lower)
            price_upper = float(self.price_upper)
            grid_number = int(self.grid_number)
            
            # Calculate grid interval
            grid_interval = (price_upper - price_lower) / grid_number
            
            # Generate levels with explicit string formatting for precise decimal control
            levels = []
            
            # Determine appropriate string format based on the asset's typical pricing
            # Use more decimal places for lower-priced assets
            if price_lower < 0.1:
                # Use 8 decimals for very low-priced assets (like many altcoins)
                format_str = "{:.8f}"
            elif price_lower < 10:
                # Use 6 decimals for low-priced assets
                format_str = "{:.6f}"
            elif price_lower < 1000:
                # Use 4 decimals for medium-priced assets
                format_str = "{:.4f}"
            else:
                # Use 2 decimals for high-priced assets
                format_str = "{:.2f}"
                
            # Log the format we're using
            self.logger.debug(f"Using price format: {format_str} for price range {price_lower}-{price_upper}")
            
            for i in range(grid_number + 1):
                # Calculate exact level
                exact_level = price_lower + (i * grid_interval)
                
                # Format to string with appropriate precision then back to float
                # This ensures we have exact decimal representation
                formatted_level = format_str.format(exact_level)
                precise_level = float(formatted_level)
                
                levels.append(precise_level)
                self.logger.debug(f"Grid level {i}: {exact_level} → {precise_level}")
                
            # Verify we have the correct number of levels
            if len(levels) != grid_number + 1:
                self.logger.warning(f"Expected {grid_number + 1} grid levels, but got {len(levels)}")
                
            # Force correct endpoints to ensure grid boundaries match exactly
            if levels and len(levels) > 1:
                levels[0] = price_lower  # First level is always exactly price_lower
                levels[-1] = price_upper  # Last level is always exactly price_upper
                
            return levels
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _calculate_order_amount(self) -> float:
        """Calculate order amount for each grid level."""
        try:
            # Ensure we have the correct types
            investment = float(self.investment)
            grid_number = int(self.grid_number)
            leverage = float(self.leverage) if hasattr(self, 'leverage') else 20.0
            
            # Simple distribution: equal investment per grid level
            investment_per_grid = investment / grid_number
            self.logger.info(f"Investment per grid: {investment_per_grid}")
            
            # Get current price to estimate amount
            ticker = self.exchange.get_ticker(self.symbol)
            price = float(ticker['last'])
            self.logger.info(f"Current price for {self.symbol}: {price}")
            
            # Calculate base amount with leverage
            amount = (investment_per_grid * leverage) / price
            self.logger.info(f"Base order amount (with {leverage}x leverage): {amount}")
            
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
    
    def _cancel_existing_orders(self):
        """Cancel any existing orders for this symbol to avoid duplicates."""
        try:
            # Get existing open orders
            open_orders = self.exchange.get_open_orders(self.symbol)
            
            if open_orders:
                self.logger.info(f"Found {len(open_orders)} existing open orders for {self.symbol}. Cancelling...")
                
                # Cancel all orders
                self.exchange.cancel_all_orders(self.symbol)
                
                # Wait for cancellation to complete
                time.sleep(2)
                
                # Verify cancellation
                open_orders_after = self.exchange.get_open_orders(self.symbol)
                if open_orders_after:
                    self.logger.warning(f"Some orders ({len(open_orders_after)}) still remain after cancellation attempt.")
                else:
                    self.logger.info("All existing orders successfully cancelled.")
                    
        except Exception as e:
            self.logger.error(f"Error cancelling existing orders: {str(e)}")
            # Don't re-raise, we'll still try to set up new orders
    
    def setup_grid(self) -> None:
        """Setup the initial grid orders with confirmation checks."""
        try:
            # Make sure we're using the correct types
            self.price_lower = float(self.price_lower)
            self.price_upper = float(self.price_upper)
            self.grid_number = int(self.grid_number)
            self.investment = float(self.investment)
            
            # Calculate grid levels
            grid_levels = self._calculate_grid_levels()
            
            # Debug log the grid levels to verify they're calculated correctly
            self.logger.info(f"Grid levels: {grid_levels}")
            
            # Calculate order amount
            amount = self._calculate_order_amount()
            self.logger.info(f"Order amount per grid: {amount}")
            
            # Get current price
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            self.logger.info(f"Setting up grid for {self.symbol} with {self.grid_number} levels")
            self.logger.info(f"Price range: {self.price_lower} - {self.price_upper}")
            self.logger.info(f"Current price: {current_price}")
            self.logger.info(f"Leverage: {self.leverage}x")
            
            # Check for existing orders and cancel if needed
            self._cancel_existing_orders()
            
            # Clear existing orders tracking
            self.grid_orders = {}
            
            # Loop through grid levels and place orders with confirmation
            order_count = 0
            retry_count = 3  # Maximum retries for order placement
            
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Only place buy orders below current price
                if buy_price < current_price:
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing buy order at price {buy_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'buy', amount, buy_price
                            )
                            
                            # Wait briefly for the order to be processed by the exchange
                            time.sleep(1)
                            
                            # Verify the order was created
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
                                order_count += 1
                            else:
                                self.logger.warning(f"Order status for buy order at {buy_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place buy order at level {i} price {buy_price}: {str(e)}")
                            retry += 1
                            time.sleep(2)  # Wait before retrying
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place buy order at level {i} after {retry_count} attempts")
                else:
                    self.logger.info(f"Skipping buy order at price {buy_price} (above current price)")
                    
                # Only place sell orders above current price
                if sell_price > current_price:
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing sell order at price {sell_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'sell', amount, sell_price
                            )
                            
                            # Wait briefly for the order to be processed by the exchange
                            time.sleep(1)
                            
                            # Verify the order was created
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
                                order_count += 1
                            else:
                                self.logger.warning(f"Order status for sell order at {sell_price} is {order_status.get('status', 'unknown')}. Retrying...")
                                retry += 1
                        except Exception as e:
                            self.logger.error(f"Failed to place sell order at level {i+1} price {sell_price}: {str(e)}")
                            retry += 1
                            time.sleep(2)  # Wait before retrying
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place sell order at level {i+1} after {retry_count} attempts")
                else:
                    self.logger.info(f"Skipping sell order at price {sell_price} (below current price)")
            
            # Log the number of orders placed
            open_orders = [o for o in self.grid_orders.values() if o['status'] == 'open']
            self.logger.info(f"Grid setup complete. Successfully placed {len(open_orders)} orders.")
            
            # Only mark the grid as running if we successfully placed at least one order
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
            
            # Get current price
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            
            # Check if price is outside grid boundaries and adaptation is enabled
            if self.enable_grid_adaptation and self._is_price_outside_grid(current_price):
                self._adapt_grid_to_price(current_price)
                return  # Skip regular update since we've reset the grid
            
            # Regular grid update logic
            self._update_orders()
            
            # Check for take profit or stop loss
            self._check_tp_sl()
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _is_price_outside_grid(self, current_price: float) -> bool:
        """Check if price is outside the grid boundaries."""
        # Add a small buffer (0.5% of grid range) to avoid triggering adaptation too frequently
        buffer_size = (self.price_upper - self.price_lower) * 0.005
        
        if current_price < (self.price_lower - buffer_size) or current_price > (self.price_upper + buffer_size):
            self.logger.info(f"Price {current_price} is outside grid range [{self.price_lower} - {self.price_upper}]")
            return True
        return False
    
    def _adapt_grid_to_price(self, current_price: float) -> None:
        """
        Adapt the grid to follow the price movement.
        
        This method will:
        1. Calculate new grid boundaries centered around current price
        2. Cancel all existing orders
        3. Set up a new grid with the new boundaries
        """
        try:
            self.logger.info(f"Adapting grid to follow price movement. Current price: {current_price}")
            
            # Store old boundaries for logging
            old_lower = self.price_lower
            old_upper = self.price_upper
            
            # Calculate grid size and boundaries
            grid_size = self.price_upper - self.price_lower
            grid_center = current_price
            
            # Calculate new boundaries ensuring we maintain the same grid size
            new_lower = grid_center - (grid_size / 2)
            new_upper = grid_center + (grid_size / 2)
            
            # Ensure new boundaries are valid (positive prices)
            if new_lower <= 0:
                new_lower = 0.00001  # Set to small positive value
                new_upper = new_lower + grid_size
            
            self.logger.info(f"New grid boundaries: [{new_lower} - {new_upper}] (old: [{old_lower} - {old_upper}])")
            
            # Update grid parameters
            self.price_lower = new_lower
            self.price_upper = new_upper
            self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
            
            # Cancel all existing orders
            self.logger.info("Cancelling all existing orders before adapting grid")
            try:
                self.exchange.cancel_all_orders(self.symbol)
                # Wait for cancellation to complete
                time.sleep(2)
            except Exception as cancel_error:
                self.logger.error(f"Error cancelling orders during grid adaptation: {cancel_error}")
            
            # Set up new grid with updated boundaries
            self.logger.info("Setting up new grid with adapted boundaries")
            
            # Reset grid orders tracking
            old_grid_orders = self.grid_orders.copy()
            self.grid_orders = {}
            
            # Try to setup the new grid
            try:
                self.setup_grid()
                
                # Increment adjustment counter
                self.grid_adjustments_count += 1
                self.logger.info(f"Successfully adapted grid to new boundaries. Adjustment #{self.grid_adjustments_count}")
                
            except Exception as setup_error:
                # If setting up new grid fails, log error and try to restore old grid
                self.logger.error(f"Failed to set up adapted grid: {setup_error}")
                
                # Restore old grid parameters
                self.price_lower = old_lower
                self.price_upper = old_upper
                self.grid_interval = (self.price_upper - self.price_lower) / self.grid_number
                self.grid_orders = old_grid_orders
                
                self.logger.warning("Restored previous grid parameters after adaptation failure")
        
        except Exception as e:
            self.logger.error(f"Error adapting grid: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_orders(self):
        """Update order status and handle filled orders."""
        # Check open orders
        open_orders = self.exchange.get_open_orders(self.symbol)
        
        # Create a map of order IDs for quick lookup
        current_order_ids = {order['id']: order for order in open_orders}
        
        # Track which grid levels already have active orders to avoid duplicates
        active_buy_levels = set()
        active_sell_levels = set()
        
        # Update status of all tracked orders
        for order_id in list(self.grid_orders.keys()):
            order_info = self.grid_orders[order_id]
            grid_level = order_info['grid_level']
            
            # If order is in current open orders, it's still active
            if order_id in current_order_ids:
                # Update status if needed
                if order_info['status'] != 'open':
                    order_info['status'] = 'open'
                
                # Track active grid levels
                if order_info['type'] == 'buy':
                    active_buy_levels.add(grid_level)
                else:
                    active_sell_levels.add(grid_level)
            
            # If order was open but not in current open orders, it may be filled
            elif order_info['status'] == 'open':
                try:
                    # Check order status
                    order_status = self.exchange.get_order_status(order_id, self.symbol)
                    
                    if order_status['status'] == 'filled':
                        # Order was filled
                        order_info['status'] = 'filled'
                        self.trades_count += 1
                        
                        # Calculate profit/loss for this trade
                        if order_info['type'] == 'sell':
                            # Simple profit calculation
                            fill_price = float(order_status.get('average', order_info['price']))
                            profit = order_info['amount'] * (fill_price - self.price_lower)
                            self.pnl += profit
                            self.logger.info(f"Order filled: {order_id} ({order_info['type']} at {fill_price}). Profit: {profit:.2f}")
                        else:
                            self.logger.info(f"Order filled: {order_id} ({order_info['type']} at {order_info['price']})")
                        
                        # Place counter order with confirmation
                        self._place_counter_order_with_confirmation(order_info)
                    
                    elif order_status['status'] == 'canceled':
                        # Order was cancelled
                        order_info['status'] = 'cancelled'
                        self.logger.info(f"Order cancelled: {order_id}")
                    
                    else:
                        # Unexpected status
                        self.logger.warning(f"Unexpected order status for {order_id}: {order_status['status']}")
                        
                except Exception as e:
                    self.logger.error(f"Error checking order {order_id} status: {e}")
        
        # Check for missing grid orders (in case some were lost or not created)
        self._ensure_grid_coverage(active_buy_levels, active_sell_levels)
    
    def _place_counter_order_with_confirmation(self, filled_order: Dict) -> None:
        """Place a counter order after an order is filled, with confirmation."""
        try:
            grid_level = filled_order['grid_level']
            price = filled_order['price']
            amount = filled_order['amount']
            retry_count = 3
            
            if filled_order['type'] == 'buy':
                # If buy order filled, place sell order one level up
                sell_price = self._round_price(price + self.grid_interval)
                
                if sell_price <= self.price_upper:
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing counter sell order at {sell_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'sell', amount, sell_price
                            )
                            
                            # Wait briefly for the order to be processed
                            time.sleep(1)
                            
                            # Verify the order was created
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
                            time.sleep(2)  # Wait before retrying
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place counter sell order after {retry_count} attempts")
            
            elif filled_order['type'] == 'sell':
                # If sell order filled, place buy order one level down
                buy_price = self._round_price(price - self.grid_interval)
                
                if buy_price >= self.price_lower:
                    order_placed = False
                    retry = 0
                    
                    while not order_placed and retry < retry_count:
                        try:
                            self.logger.info(f"Placing counter buy order at {buy_price}")
                            order = self.exchange.create_limit_order(
                                self.symbol, 'buy', amount, buy_price
                            )
                            
                            # Wait briefly for the order to be processed
                            time.sleep(1)
                            
                            # Verify the order was created
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
                            time.sleep(2)  # Wait before retrying
                    
                    if not order_placed:
                        self.logger.error(f"Failed to place counter buy order after {retry_count} attempts")
        
        except Exception as e:
            self.logger.error(f"Error in _place_counter_order_with_confirmation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _ensure_grid_coverage(self, active_buy_levels: Set[int], active_sell_levels: Set[int]) -> None:
        """Ensure all grid levels have active orders if needed."""
        try:
            # Current price is needed to determine which levels should have orders
            current_price = float(self.exchange.get_ticker(self.symbol)['last'])
            grid_levels = self._calculate_grid_levels()
            amount = self._calculate_order_amount()
            
            # Only check for missing orders if we have grid levels
            if not grid_levels or len(grid_levels) <= 1:
                return
                
            self.logger.debug(f"Checking grid coverage. Current price: {current_price}")
            self.logger.debug(f"Active buy levels: {active_buy_levels}")
            self.logger.debug(f"Active sell levels: {active_sell_levels}")
            
            # Check each grid level
            for i in range(len(grid_levels) - 1):
                buy_price = grid_levels[i]
                sell_price = grid_levels[i + 1]
                
                # Check for missing buy orders below current price
                if buy_price < current_price and i not in active_buy_levels:
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
                
                # Check for missing sell orders above current price
                if sell_price > current_price and (i + 1) not in active_sell_levels:
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
            # Calculate PnL percentage
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
            
            # Cancel all open orders
            try:
                self.exchange.cancel_all_orders(self.symbol)
                self.logger.info(f"All orders cancelled for {self.symbol}")
                
                # Mark all orders as cancelled in our local tracking
                for order_id in self.grid_orders:
                    if self.grid_orders[order_id]['status'] == 'open':
                        self.grid_orders[order_id]['status'] = 'cancelled'
                        
            except Exception as e:
                self.logger.error(f"Error cancelling orders for {self.symbol}: {e}")
            
            # Close all positions with enhanced error handling
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
            # Ensure grid is marked as stopped even if there are errors
            self.running = False
        
    def get_status(self) -> Dict:
        """Get the current status of the grid strategy with adaptation info."""
        try:
            status = {
                'grid_id': self.grid_id,
                'symbol': self.symbol,
                'display_symbol': self.original_symbol if hasattr(self, 'original_symbol') else self.symbol,
                'price_lower': self.price_lower,
                'price_upper': self.price_upper,
                'grid_number': self.grid_number,
                'grid_interval': self.grid_interval,
                'investment': self.investment,
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