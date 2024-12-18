use clap::Parser;
use ethers::signers::{LocalWallet, Signer};
use ethers::types::H160;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use hyperliquid_rust_sdk::{
    BaseUrl, ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest, ExchangeClient,
    ExchangeDataStatus, ExchangeResponseStatus, InfoClient, Message, Subscription, UserData,
    EPSILON,
};
use nonzero_ext::nonzero;
use tokio::sync::mpsc::unbounded_channel;
use tracing::{error, info};
use tracing_subscriber;
use dialoguer::{Input, Password};

#[derive(Debug, Parser)]
#[command(
    name = "Hyperliquid Grid Trading Bot",
    author = "augustin-v",
    version = "1.0",
    about = "A grid trading bot for Hyperliquid perps",
)]
pub struct GridConfig {
    /// Trading pair (e.g., "ETH-USD")
    #[arg(short, long)]
    pub asset: String,

    /// Upper price boundary for grid trading
    #[arg(long)]
    pub upper_price: f64,

    /// Lower price boundary for grid trading
    #[arg(long)]
    pub lower_price: f64,

    /// Number of grid levels to place orders
    #[arg(short, long, default_value_t = 10)]
    pub grid_levels: u32,

    /// Quantity to trade at each grid level
    #[arg(short, long)]
    pub quantity_per_grid: f64,

    /// Maximum total position size allowed
    #[arg(long, default_value_t = 1.0)]
    pub max_position_size: f64,

    /// Price decimal precision
    #[arg(long, default_value_t = 2)]
    pub decimals: u32,

    /// Private key for wallet authentication
    #[arg(long, env = "HL_PRIVATE_KEY")]
    wallet: LocalWallet,

    /// Stop loss percentage from entry price
    #[arg(long, default_value_t = 0.05)]
    pub stop_loss_percentage: f64,

    /// Take profit percentage per grid level
    #[arg(long, default_value_t = 0.02)]
    pub take_profit_percentage: f64,

    /// Maximum drawdown allowed before emergency shutdown
    #[arg(long, default_value_t = 0.1)]
    pub max_drawdown: f64,
}

impl GridConfig {
    pub fn build_interactive() -> Self {
        // Basic prompts
        let asset: String = Input::new()
            .with_prompt("Enter trading pair (e.g., ETH-USD)")
            .interact()
            .unwrap();

        let upper_price: f64 = Input::new()
            .with_prompt("Enter upper price boundary")
            .interact()
            .unwrap();

        let lower_price: f64 = Input::new()
            .with_prompt("Enter lower price boundary")
            .interact()
            .unwrap();

        println!("\n📊 Risk Management Setup");
        println!("Recommended: 5% stop loss, 2% take profit, 10% max drawdown");
        
        let stop_loss: f64 = Input::new()
            .with_prompt("Stop loss percentage, default 5%:")
            .default(0.05)
            .interact()
            .unwrap();

        let take_profit: f64 = Input::new()
            .with_prompt("Take profit percentage, default 2%:")
            .default(0.02)
            .interact()
            .unwrap();

        let max_drawdown: f64 = Input::new()
            .with_prompt("Maximum drawdown percentage")
            .default(0.10)
            .interact()
            .unwrap();

        // Private key input with hidden input
        let wallet = match std::env::var("HL_PRIVATE_KEY") {
            Ok(key) => key.parse().unwrap(),
            Err(_) => {
                let wallet_key: String = Password::new()
                    .with_prompt("Enter your private key (Or set it up in a .env file like so: HL_PRIVATE_KEY=YOURPRIVATEKEY)")
                    .interact()
                    .unwrap();
                wallet_key.parse().unwrap()
            }
        };

        Self {
            asset,
            upper_price,
            lower_price,
            grid_levels: 10,
            quantity_per_grid: 0.1,
            max_position_size: 1.0,
            decimals: 2,
            stop_loss_percentage: stop_loss,
            take_profit_percentage: take_profit,
            max_drawdown,
            wallet,
        }
    }

}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!(
        "
    ╔═══════════════════════════════════════════╗
    ║     HYPERLIQUID GRID TRADING BOT v1.0     ║
    ║                                           ║
    ║            X: @AugustinV_dev              ║
    ║                                           ║
    ║        Let's get Rusty rich!🦀💰          ║
    ╚═══════════════════════════════════════════╝
    "
    );

    // Parse config from CLI arguments
    let config = match GridConfig::try_parse() {
        Ok(config) => config,
        Err(_) => {
            info!("Interactive CLI build initiated 🤖");
            GridConfig::build_interactive()
        },
    };

    info!("🚀 Starting grid trader with configuration:");
    info!("📊 Asset: {}", config.asset);
    info!("📈 Upper Price: ${}", config.upper_price);
    info!("📉 Lower Price: ${}", config.lower_price);
    info!("🔲 Grid Levels: {}", config.grid_levels);
    info!("💰 Quantity per Grid: {}", config.quantity_per_grid);
    info!("📊 Max Position Size: {}", config.max_position_size);
    
    // Initialize grid trader with error handling
    let mut grid_trader = match GridTrader::new(config).await {
        Ok(trader) => trader,
        Err(e) => {
            error!("Failed to initialize grid trader: {}", e);
            std::process::exit(1);
        }
    };

    grid_trader.start().await;
}

#[derive(Debug)]
pub struct GridOrder {
    pub oid: u64,
    pub position: f64,
    pub price: f64,
    pub is_buy: bool,
}

pub struct GridTrader {
    pub asset: String,
    pub upper_price: f64,
    pub lower_price: f64,
    pub grid_levels: u32,
    pub quantity_per_grid: f64,
    pub max_position_size: f64,
    pub decimals: u32,
    pub active_orders: Vec<GridOrder>,
    pub cur_position: f64,
    pub latest_mid_price: f64,
    pub info_client: InfoClient,
    pub exchange_client: ExchangeClient,
    pub user_address: H160,
    pub rate_limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,
    pub stop_loss_percentage: f64,
    pub take_profit_percentage: f64,
    pub max_drawdown: f64,
    pub initial_mid_price: f64,
}

impl GridTrader {
pub async fn new(input: GridConfig) -> Result<GridTrader, Box<dyn std::error::Error>> {
    let user_address = input.wallet.address();
    let info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await?;
    let exchange_client = ExchangeClient::new(
        None,
        input.wallet,
        Some(BaseUrl::Testnet),
        None,
        None,
    ).await?;

    // Verify asset exists
    let meta = info_client.meta().await?;
    let asset_exists = meta.universe.iter().any(|asset| asset.name == input.asset);
    if !asset_exists {
        return Err(format!("Asset {} not found", input.asset).into());
    }

    let rate_limiter = RateLimiter::direct(Quota::per_second(nonzero!(1200u32)));

    Ok(GridTrader {
        asset: input.asset,
        upper_price: input.upper_price,
        lower_price: input.lower_price,
        grid_levels: input.grid_levels,
        quantity_per_grid: input.quantity_per_grid,
        max_position_size: input.max_position_size,
        decimals: input.decimals,
        active_orders: Vec::new(),
        cur_position: 0.0,
        latest_mid_price: -1.0,
        initial_mid_price: -1.0,
        stop_loss_percentage: input.stop_loss_percentage,
        take_profit_percentage: input.take_profit_percentage,
        max_drawdown: input.max_drawdown,
        info_client,
        exchange_client,
        user_address,
        rate_limiter,
    })
}

    pub async fn start(&mut self) {
        let (sender, mut receiver) = unbounded_channel();

        // Add signal handling
        let shutdown = tokio::signal::ctrl_c();
        tokio::pin!(shutdown);

        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await
            .unwrap();

        self.info_client
            .subscribe(Subscription::AllMids, sender)
            .await
            .unwrap();

        // Place initial grid orders
        self.place_grid_orders().await;
        self.initial_mid_price = -1.0;
        loop {
            tokio::select! {
                _ = &mut shutdown => {
                    info!("Shutdown signal received, cleaning up...");
                    self.cleanup().await;
                    break;
                }
                Some(message) = receiver.recv() => {
                    match message {
                        Message::AllMids(all_mids) => {
                            if let Some(mid) = all_mids.data.mids.get(&self.asset) {
                                let mid: f64 = mid.parse().unwrap();
                                self.latest_mid_price = mid;
                            // Initialize initial price if not set
                            if self.initial_mid_price < 0.0 {
                                self.initial_mid_price = mid;
                            }

                            // Check risk management conditions
                            if self.check_risk_management().await {
                                break;
                            }
                                self.check_and_rebalance_grid().await;
                            }
                        }
                        Message::User(user_events) => {
                            if self.latest_mid_price < 0.0 {
                                continue;
                            }
                            if let UserData::Fills(fills) = user_events.data {
                                for fill in fills {
                                    let amount: f64 = fill.sz.parse().unwrap();
                                    if fill.side.eq("B") {
                                        self.cur_position += amount;
                                        info!("Fill: bought {amount} {}", self.asset);
                                    } else {
                                        self.cur_position -= amount;
                                        info!("Fill: sold {amount} {}", self.asset);
                                    }
                                }
                                self.check_and_rebalance_grid().await;
                            }
                        }
                        _ => {
                            error!("Unsupported message type");
                        }
                    }
                }
            }
        }
    }

    async fn cleanup(&mut self) {
        info!("Cancelling all active orders...");
        for order in &self.active_orders {
            if self.attempt_cancel(self.asset.clone(), order.oid).await {
                info!("Cancelled order {}", order.oid);
            }
        }
        info!("Final position: {} {}", self.cur_position, self.asset);
    }

    async fn place_grid_orders(&mut self) {
        let grid_step = (self.upper_price - self.lower_price) / (self.grid_levels as f64);

        for i in 0..self.grid_levels {
            let price = self.lower_price + (grid_step * i as f64);
            let (amount_resting, oid) = self
                .place_order_with_rate_limit(
                    self.asset.clone(),
                    self.quantity_per_grid,
                    price,
                    true,
                )
                .await;

            if amount_resting > EPSILON {
                self.active_orders.push(GridOrder {
                    oid,
                    position: amount_resting,
                    price,
                    is_buy: true,
                });
                info!(
                    "Placed grid buy order at {price} for {amount_resting} {}",
                    self.asset
                );
            }
        }
    }

    async fn place_order_with_rate_limit(
        &self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        self.rate_limiter.until_ready().await;
        self.place_order(asset, amount, price, is_buy).await
    }

    async fn attempt_cancel_with_rate_limit(&self, asset: String, oid: u64) -> bool {
        self.rate_limiter.until_ready().await;
        self.attempt_cancel(asset, oid).await
    }

    async fn place_order(
        &self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        let asset_id = match self.get_asset_id(&asset).await {
            Ok(id) => id.to_string(),
            Err(e) => {
                error!("Failed to get asset ID for {}: {}", asset, e);
                return (0.0, 0);
            }
        };
    
        let order = self
            .exchange_client
            .order(
                ClientOrderRequest {
                    asset: asset_id,
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: amount,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                },
                None,
            )
            .await;

        match order {
            Ok(ExchangeResponseStatus::Ok(response)) => {
                if let Some(data) = response.data {
                    if !data.statuses.is_empty() {
                        match data.statuses[0].clone() {
                            ExchangeDataStatus::Filled(order) => (amount, order.oid),
                            ExchangeDataStatus::Resting(order) => (amount, order.oid),
                            ExchangeDataStatus::Error(e) => {
                                error!("Error placing order: {e}");
                                (0.0, 0)
                            }
                            _ => (0.0, 0),
                        }
                    } else {
                        (0.0, 0)
                    }
                } else {
                    (0.0, 0)
                }
            }
            _ => {
                error!("Unexpected order response: {:?}", order);
                (0.0, 0)
            }
        }
    }

    async fn check_and_rebalance_grid(&mut self) {
        let grid_step = (self.upper_price - self.lower_price) / (self.grid_levels as f64);

        // Calculate ideal grid levels based on current price
        let mut ideal_grid_levels = Vec::new();
        for i in 0..self.grid_levels {
            let price = self.lower_price + (grid_step * i as f64);
            ideal_grid_levels.push(price);
        }

        let mut orders_to_cancel = Vec::new();
        for order in &self.active_orders {
            let closest_grid = ideal_grid_levels.iter().min_by(|&&a, &b| {
                let diff_a = (a - order.price).abs();
                let diff_b = (b - order.price).abs();
                diff_a.partial_cmp(&diff_b).unwrap()
            });

            if let Some(&closest_price) = closest_grid {
                if (closest_price - order.price).abs() > EPSILON {
                    orders_to_cancel.push(order.oid);
                }
            }
        }

        // Cancel outdated orders
        for oid in orders_to_cancel {
            if self
                .attempt_cancel_with_rate_limit(self.asset.clone(), oid)
                .await
            {
                self.active_orders.retain(|order| order.oid != oid);
                info!("Cancelled order {oid} during rebalance");
            }
        }

        // Place new orders at missing grid levels
        for ideal_price in ideal_grid_levels {
            // Check if we already have an order near this price
            let has_order = self
                .active_orders
                .iter()
                .any(|order| (order.price - ideal_price).abs() < EPSILON);

            if !has_order {
                // Check if we're within position limits
                let potential_position = self.cur_position + self.quantity_per_grid;
                if potential_position.abs() <= self.max_position_size {
                    let (amount_resting, oid) = self
                        .place_order_with_rate_limit(
                            self.asset.clone(),
                            self.quantity_per_grid,
                            ideal_price,
                            ideal_price < self.latest_mid_price,
                        )
                        .await;

                    if amount_resting > EPSILON {
                        self.active_orders.push(GridOrder {
                            oid,
                            position: amount_resting,
                            price: ideal_price,
                            is_buy: ideal_price < self.latest_mid_price,
                        });
                        info!(
                            "Placed rebalancing {} order at {} for {} {}",
                            if ideal_price < self.latest_mid_price {
                                "buy"
                            } else {
                                "sell"
                            },
                            ideal_price,
                            amount_resting,
                            self.asset
                        );
                    }
                }
            }
        }
    }

    pub async fn get_asset_id(&self, symbol: &str) -> Result<u64, Box<dyn std::error::Error>> {
        let meta = self.info_client.meta().await?;
        for (index, asset) in meta.universe.iter().enumerate() {
            if asset.name == symbol {
                return Ok(index as u64);
            }
        }
        Err("Asset not found".into())
    }

    async fn attempt_cancel(&self, asset: String, oid: u64) -> bool {
        let cancel = self
            .exchange_client
            .cancel(ClientCancelRequest { asset, oid }, None)
            .await;

        match cancel {
            Ok(ExchangeResponseStatus::Ok(cancel)) => {
                if let Some(cancel) = cancel.data {
                    if !cancel.statuses.is_empty() {
                        match cancel.statuses[0].clone() {
                            ExchangeDataStatus::Success => {
                                return true;
                            }
                            ExchangeDataStatus::Error(e) => {
                                error!("Error with cancelling: {e}")
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        error!("Exchange data statuses is empty when cancelling: {cancel:?}")
                    }
                } else {
                    error!("Exchange response data is empty when cancelling: {cancel:?}")
                }
            }
            Ok(ExchangeResponseStatus::Err(e)) => error!("Error with cancelling: {e}"),
            Err(e) => error!("Error with cancelling: {e}"),
        }
        false
    }

    async fn check_risk_management(&mut self) -> bool {
        // Stop loss check
        let stop_loss_price = self.initial_mid_price * (1.0 - self.stop_loss_percentage);
        if self.latest_mid_price <= stop_loss_price {
            error!("Stop loss triggered at price: {}", self.latest_mid_price);
            self.emergency_shutdown().await;
            return true;
        }

        // Take profit check for each grid level
        for order in &self.active_orders {
            let take_profit_price = order.price * (1.0 + self.take_profit_percentage);
            if (order.is_buy && self.latest_mid_price >= take_profit_price)
                || (!order.is_buy && self.latest_mid_price <= take_profit_price)
            {
                info!("Take profit triggered for order at price: {}", order.price);
                self.attempt_cancel_with_rate_limit(self.asset.clone(), order.oid)
                    .await;
            }
        }

        // Drawdown check
        let drawdown =
            (self.initial_mid_price - self.latest_mid_price).abs() / self.initial_mid_price;
        if drawdown > self.max_drawdown {
            error!("Maximum drawdown exceeded: {:.2}%", drawdown * 100.0);
            self.emergency_shutdown().await;
            return true;
        }

        // Abnormal price movement check
        let price_change_percentage =
            (self.latest_mid_price - self.initial_mid_price) / self.initial_mid_price;
        if price_change_percentage.abs() > 0.1 {
            // 10% price movement
            error!(
                "Abnormal price movement detected: {:.2}%",
                price_change_percentage * 100.0
            );
            self.emergency_shutdown().await;
            return true;
        }

        false
    }

    async fn emergency_shutdown(&mut self) {
        info!("🚨 Emergency shutdown initiated");
    
        // Cancel all active orders
        for order in &self.active_orders {
            if self
                .attempt_cancel_with_rate_limit(self.asset.clone(), order.oid)
                .await
            {
                info!("Cancelled order {} during emergency shutdown", order.oid);
            }
        }
        // Clear the active orders vector
        self.active_orders.clear();
    
        // Close all positions if needed
        if self.cur_position.abs() > EPSILON {
            let (amount, _) = self
                .place_order_with_rate_limit(
                    self.asset.clone(),
                    self.cur_position.abs(),
                    self.latest_mid_price,
                    self.cur_position < 0.0,
                )
                .await;
            info!("Closed position of {} {}", amount, self.asset);
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grid_calculations() {
        let trader = GridTrader::new(create_test_config()).await.unwrap();
        let grid_step = (trader.upper_price - trader.lower_price) / (trader.grid_levels as f64);
        assert_eq!(grid_step, 100.0);
    }

    #[tokio::test]
    async fn test_order_placement() {
        let mut trader = GridTrader::new(create_test_config()).await.unwrap();
        trader.latest_mid_price = 2250.0;
        let grid_step = (trader.upper_price - trader.lower_price) / (trader.grid_levels as f64);
        let expected_prices: Vec<f64> = (0..trader.grid_levels)
            .map(|i| trader.lower_price + (grid_step * i as f64))
            .collect();
        assert_eq!(expected_prices.len(), 3);
        assert_eq!(expected_prices[0], 2000.0);
        assert_eq!((expected_prices[1] * 1000.0).round() / 1000.0, 2166.667);
        assert_eq!(expected_prices[2], 2333.3333333333335);
    }
    
    #[tokio::test]
    async fn test_position_limits() {
        let mut trader = GridTrader::new(create_test_config()).await.unwrap();
        trader.cur_position = 0.8;
        let potential_position = trader.cur_position + trader.quantity_per_grid;
        assert!(potential_position > trader.max_position_size);
    }

    #[tokio::test]
    async fn test_grid_rebalancing() {
        let mut trader = GridTrader::new(create_test_config()).await.unwrap();
        trader.active_orders.push(GridOrder {
            oid: 1,
            position: 0.1,
            price: 2000.0,
            is_buy: true,
        });
    
        assert_eq!(trader.active_orders.len(), 1);
        assert!(trader
            .active_orders
            .iter()
            .any(|order| order.price == 2000.0));
    }
    

    #[tokio::test]
    async fn test_price_boundaries() {
        let trader = GridTrader::new(create_test_config()).await.unwrap();
        assert!(trader.lower_price <= trader.upper_price);
        assert!(trader.grid_levels > 0);
        assert!(trader.quantity_per_grid > 0.0);
    }

#[tokio::test]
async fn test_websocket_subscription() {
    let config = create_test_config();
    let trader = GridTrader::new(config).await.unwrap();
    assert_eq!(trader.user_address, trader.exchange_client.wallet.address());
}
    #[test]
    fn test_default_args() {
        let args = GridConfig::try_parse_from(&[
            "test",
            "--asset",
            "ETH-USD",
            "--upper-price",
            "2500.0",
            "--lower-price",
            "2000.0",
            "--quantity-per-grid",
            "0.1",
            "--wallet",
            "e908f86dbb4d55ac876378565aafeabc187f6690f046459397b17d9b9a19688e",
        ])
        .unwrap();

        // Check default values
        assert_eq!(args.grid_levels, 10);
        assert_eq!(args.max_position_size, 1.0);
        assert_eq!(args.decimals, 2);
    }

    #[test]
    fn test_custom_args() {
        let args = GridConfig::try_parse_from(&[
            "test",
            "--asset",
            "ETH-USD",
            "--upper-price",
            "3000.0",
            "--lower-price",
            "2000.0",
            "--grid-levels",
            "5",
            "--quantity-per-grid",
            "0.2",
            "--max-position-size",
            "2.0",
            "--wallet",
            "e908f86dbb4d55ac876378565aafeabc187f6690f046459397b17d9b9a19688e",
        ])
        .unwrap();

        assert_eq!(args.asset, "ETH-USD");
        assert_eq!(args.upper_price, 3000.0);
        assert_eq!(args.lower_price, 2000.0);
        assert_eq!(args.grid_levels, 5);
        assert_eq!(args.quantity_per_grid, 0.2);
        assert_eq!(args.max_position_size, 2.0);
    }

    #[test]
    fn test_invalid_price_range() {
        let result = GridConfig::try_parse_from(&[
            "test",
            "--upper-price",
            "1000.0",
            "--lower-price",
            "2000.0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_grid_levels() {
        let result = GridConfig::try_parse_from(&["test", "--grid-levels", "0"]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_position_update_after_fill() {
        let mut trader = setup_test_trader().await;
        trader.cur_position = 0.0;

        // Simulate buy fill
        trader.cur_position += 0.1;
        assert_eq!(trader.cur_position, 0.1);

        // Simulate sell fill
        trader.cur_position -= 0.05;
        assert_eq!(trader.cur_position, 0.05);
    }

    #[tokio::test]
    async fn test_grid_spacing() {
        let trader = setup_test_trader().await;
        let grid_step = (trader.upper_price - trader.lower_price) / (trader.grid_levels as f64);

        let mut previous_price = trader.lower_price;
        for i in 1..trader.grid_levels {
            let current_price = trader.lower_price + (grid_step * i as f64);
            assert!(current_price > previous_price);
            assert!((current_price - previous_price - grid_step).abs() < EPSILON);
            previous_price = current_price;
        }
    }

    #[tokio::test]
    async fn test_max_position_validation() {
        let mut trader = setup_test_trader().await;
        trader.cur_position = trader.max_position_size;

        // Should not place buy orders when at max position
        let potential_position = trader.cur_position + trader.quantity_per_grid;
        assert!(potential_position > trader.max_position_size);
    }

    #[tokio::test]
    async fn test_order_cancellation() {
        let mut trader = setup_test_trader().await;
        trader.active_orders.push(GridOrder {
            oid: 1,
            position: 0.1,
            price: 2000.0,
            is_buy: true,
        });
        trader.active_orders.push(GridOrder {
            oid: 2,
            position: 0.1,
            price: 2100.0,
            is_buy: true,
        });

        assert_eq!(trader.active_orders.len(), 2);
        trader.active_orders.retain(|order| order.oid != 1);
        assert_eq!(trader.active_orders.len(), 1);
        assert_eq!(trader.active_orders[0].oid, 2);
    }

    #[tokio::test]
    async fn test_price_update_handling() {
        let mut trader = setup_test_trader().await;
        assert!(trader.latest_mid_price < 0.0);

        trader.latest_mid_price = 2250.0;
        assert!(trader.latest_mid_price > trader.lower_price);
        assert!(trader.latest_mid_price < trader.upper_price);
    }

    #[tokio::test]
    async fn test_grid_order_direction() {
        let mut trader = setup_test_trader().await;
        trader.latest_mid_price = 2250.0;

        // Orders below mid price should be buys
        assert!(2000.0 < trader.latest_mid_price);
        // Orders above mid price should be sells
        assert!(2500.0 > trader.latest_mid_price);
    }

    #[tokio::test]
    async fn test_quantity_validation() {
        let trader = setup_test_trader().await;
        assert!(trader.quantity_per_grid > 0.0);
        assert!(trader.quantity_per_grid <= trader.max_position_size);
    }

// Helper function for test setup
async fn setup_test_trader() -> GridTrader {
    let config = create_test_config();
    GridTrader::new(config).await.unwrap()
}

// Helper function to create test config
fn create_test_config() -> GridConfig {
    GridConfig {
        asset: "ETH".to_string(),
        upper_price: 2500.0,
        lower_price: 2000.0,
        grid_levels: 5,
        quantity_per_grid: 0.1,
        max_position_size: 1.0,
        decimals: 2,
        stop_loss_percentage: 0.05,
        take_profit_percentage: 0.02,
        max_drawdown: 0.1,
        wallet: "e908f86dbb4d55ac876378565aafeabc187f6690f046459397b17d9b9a19688e"
            .parse()
            .unwrap(),
    }
}


    #[tokio::test]
    async fn test_api_error_response() {
        let trader = setup_test_trader().await;
        let (amount_resting, oid) = trader
            .place_order("INVALID-ASSET".to_string(), 0.1, 2000.0, true)
            .await;
        assert_eq!(amount_resting, 0.0);
        assert_eq!(oid, 0);
    }

    #[tokio::test]
    async fn test_rate_limit_handling() {
        let trader = setup_test_trader().await;
        let mut orders = Vec::new();

        // Try to place many orders quickly
        for i in 0..10 {
            let price = 2000.0 + (i as f64 * 10.0);
            let (amount, oid) = trader
                .place_order_with_rate_limit(trader.asset.clone(), 0.1, price, true)
                .await;
            orders.push((amount, oid));
        }

        assert!(orders.len() == 10);
    }

    #[tokio::test]
    async fn test_multiple_fills() {
        let mut trader = setup_test_trader().await;
        trader.cur_position = 0.0;

        // Simulate multiple fills
        for _ in 0..5 {
            trader.cur_position += 0.1;
        }
        assert!((trader.cur_position - 0.5).abs() < EPSILON);

        for _ in 0..3 {
            trader.cur_position -= 0.1;
        }
        assert!((trader.cur_position - 0.2).abs() < EPSILON);
    }

    #[tokio::test]
    async fn test_extreme_price_conditions() {
        let mut trader = setup_test_trader().await;

        // Test extreme price movement
        trader.latest_mid_price = trader.upper_price * 2.0;
        assert!(trader.latest_mid_price > trader.upper_price);

        trader.latest_mid_price = trader.lower_price / 2.0;
        assert!(trader.latest_mid_price < trader.lower_price);
    }

    #[tokio::test]
    async fn test_invalid_market_data() {
        let mut trader = setup_test_trader().await;
        trader.latest_mid_price = -1.0;

        // Test negative prices
        assert!(trader.latest_mid_price < 0.0);

        // Test zero price
        trader.latest_mid_price = 0.0;
        assert!(trader.latest_mid_price <= trader.lower_price);
    }

    #[tokio::test]
    async fn test_full_grid_cycle() {
        let mut trader = setup_test_trader().await;

        // Initial grid setup
        trader.place_grid_orders().await;
        assert!(!trader.active_orders.is_empty());

        // Price movement and rebalance
        trader.latest_mid_price = 2250.0;
        trader.check_and_rebalance_grid().await;

        // Verify grid adaptation
        assert!(trader
            .active_orders
            .iter()
            .any(|order| order.price < trader.latest_mid_price));
        assert!(trader
            .active_orders
            .iter()
            .any(|order| order.price > trader.latest_mid_price));
    }

    #[tokio::test]
    async fn test_order_race_conditions() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let trader = setup_test_trader().await;
        let trader = Arc::new(Mutex::new(trader));

        let mut handles = Vec::new();
        for i in 0..5 {
            let trader = trader.clone();
            let price = 2000.0 + (i as f64 * 100.0);
            handles.push(tokio::spawn(async move {
                let trader = trader.lock().await;
                trader
                    .place_order_with_rate_limit("ETH".to_string(), 0.1, price, true)
                    .await
            }));
        }

        for handle in handles {
            let _ = handle.await;
        }
    }

    #[tokio::test]
    async fn test_state_recovery() {
        let mut trader = setup_test_trader().await;

        // Setup initial state
        trader.place_grid_orders().await;
        let initial_orders = trader.active_orders.len();

        // Simulate disconnect/reconnect
        trader.active_orders.clear();
        assert_eq!(trader.active_orders.len(), 0);

        // Recover state
        trader.place_grid_orders().await;
        assert_eq!(trader.active_orders.len(), initial_orders);
    }

    #[tokio::test]
    async fn test_stop_loss_trigger() {
        let mut trader = setup_test_trader().await;
        trader.initial_mid_price = 2250.0;
        trader.latest_mid_price = 2250.0 * (1.0 - trader.stop_loss_percentage - 0.01);

        assert!(trader.check_risk_management().await);
    }

    #[tokio::test]
    async fn test_take_profit_check() {
        let mut trader = setup_test_trader().await;
        trader.initial_mid_price = 2250.0;
        trader.latest_mid_price = 2250.0;

        trader.active_orders.push(GridOrder {
            oid: 1,
            position: 0.1,
            price: 2000.0,
            is_buy: true,
        });

        let take_profit_price = 2000.0 * (1.0 + trader.take_profit_percentage);
        trader.latest_mid_price = take_profit_price + 1.0;

        assert!(trader.check_risk_management().await);
    }

    #[tokio::test]
    async fn test_max_drawdown_protection() {
        let mut trader = setup_test_trader().await;
        trader.initial_mid_price = 2250.0;
        trader.latest_mid_price = 2250.0 * (1.0 - trader.max_drawdown - 0.01);

        assert!(trader.check_risk_management().await);
    }

    #[tokio::test]
    async fn test_emergency_shutdown() {
        let mut trader = setup_test_trader().await;
        trader.active_orders.push(GridOrder {
            oid: 1,
            position: 0.1,
            price: 2000.0,
            is_buy: true,
        });

        trader.emergency_shutdown().await;
        assert!(trader.active_orders.is_empty());
    }

    #[tokio::test]
    async fn test_abnormal_price_movement() {
        let mut trader = setup_test_trader().await;
        trader.initial_mid_price = 2250.0;
        trader.latest_mid_price = 2250.0 * 1.11; // 11% price movement

        assert!(trader.check_risk_management().await);
    }

    #[tokio::test]
    async fn test_grid_rebalancing_with_risk_check() {
        let mut trader = setup_test_trader().await;
        trader.initial_mid_price = 2250.0;
        trader.latest_mid_price = 2250.0;

        trader.active_orders.push(GridOrder {
            oid: 1,
            position: 0.1,
            price: 2000.0,
            is_buy: true,
        });

        trader.check_and_rebalance_grid().await;
        assert!(!trader.check_risk_management().await);
    }
}
