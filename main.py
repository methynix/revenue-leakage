import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2026)

n_orders = 1000

regions = [
    'East Africa',
    'South America',
    'Asia',
    'Central Africa',
    'Europe'
]

gateways = [
    'PayPal',
    'Stripe',
    'M-Pesa',
    'Crypto',
    'Yas'
]

statuses = [
    'Completed',
    'Cancelled',
    'Refunded',
    'Failed'
]

raw_data = {
    'order_id': [f'MTX-{1000+i}' for i in range(n_orders)],
    'order_date': pd.date_range(
        start='2026-01-01',
        periods=n_orders,
        freq='h'
    ),
    'region': np.random.choice(
        regions,
        n_orders,
        p=[0.4, 0.05, 0.2, 0.15, 0.2]
    ),
    'payment_gateway': np.random.choice(
        gateways,
        n_orders,
        p=[0.2, 0.1, 0.4, 0.1, 0.2]
    ),
    'unit_price': np.random.uniform(
        50.0,
        150.0,
        size=n_orders
    ),
    'quantity': np.random.randint(
        1,
        11,
        size=n_orders
    ),
    'discount_pct': np.clip(
        np.random.normal(0.15, 0.1, n_orders),
        0,
        0.5
    ),
    'order_status': np.random.choice(
        statuses,
        n_orders,
        p=[0.5, 0.1, 0.2, 0.2]
    ),
    'shipping_cost': np.random.uniform(
        20.0,
        80.0,
        n_orders
    )
}

df = pd.DataFrame(raw_data)

df.loc[
    df.sample(30).index,
    'payment_gateway'
] = np.nan

df.loc[
    df.sample(20).index,
    'unit_price'
] = np.nan

df['unit_price'] = df['unit_price'].astype(str)

df.loc[
    df.sample(10).index,
    'unit_price'
] = 'UNKNOWN'

print("The generated data initially")
print(df)

df['unit_price'] = pd.to_numeric(
    df['unit_price'],
    errors='coerce'
)

df['unit_price'] = df['unit_price'].fillna(
    df.groupby('region')['unit_price'].transform('median')
)

payment_mode = df['payment_gateway'].mode()[0]

df['payment_gateway'] = df['payment_gateway'].fillna(
    payment_mode
)

print('=' * 40)
print("After some cleaning of data")
print(df)

df['gross_revenue'] = (
    df['unit_price'] *
    df['quantity']
)

df['discount_amount'] = (
    df['gross_revenue'] *
    df['discount_pct']
)

df['net_revenue'] = (
    df['gross_revenue'] -
    df['discount_amount']
)

print('=' * 40)
print("After adding other columns")
print(df)

print('=' * 40)

net_revenue_uncompleted = (
    df.query(
        "order_status in ['Cancelled', 'Failed']"
    )['net_revenue']
    .sum()
)

df['risk_flag'] = np.where(
    (df['discount_pct'] >= 0.5) |
    (
        df['order_status'].isin(['Failed', 'Refunded']) &
        (df['net_revenue'] > 500)
    ),
    'CRITICAL',
    'NORMAL'
)

fig, ax = plt.subplots(
    2,
    2,
    figsize=(16, 14)
)

completed_orders = df.loc[
    df['order_status'] == 'Completed'
]

daily_revenue = (
    completed_orders
    .set_index('order_date')['net_revenue']
    .resample('D')
    .sum()
    .cumsum()
)

ax[0, 0].set_title(
    "Cumulative Daily Net Revenue"
)

ax[0, 0].plot(
    daily_revenue.index,
    daily_revenue.values,
    linewidth=1.5,
    linestyle='-',
    marker='o'
)

ax[0, 0].set_xlabel('Date')
ax[0, 0].set_ylabel(
    'Cumulative Net Revenue ($)'
)

r_leakage = (
    df.query(
        "order_status in ['Cancelled', 'Failed', 'Refunded']"
    )
    .groupby('payment_gateway')['net_revenue']
    .sum()
)

ax[0, 1].barh(
    r_leakage.index,
    r_leakage.values
)

ax[0, 1].set_title(
    "Revenue Leakage by Payment Gateway",
    fontweight='bold'
)

ax[0, 1].set_xlabel(
    'Leaked Revenue ($)'
)

ax[0, 1].set_ylabel(
    'Payment Gateway'
)

ax[0, 1].grid(
    True,
    linestyle=':',
    alpha=0.6
)

pvt = df.pivot_table(
    index='region',
    columns='order_status',
    values='order_id',
    aggfunc='count'
)

pvt.plot(
    kind='bar',
    stacked=False,
    ax=ax[1, 0],
    colormap='Set1'
)

ax[1, 0].set_title(
    "Regional Order Status Distribution",
    fontweight='bold'
)

ax[1, 0].set_xlabel(
    'Region'
)

ax[1, 0].set_ylabel(
    'Order Count'
)

ax[1, 0].tick_params(
    axis='x',
    rotation=30
)

ax[1, 0].grid(
    True,
    linestyle=':',
    alpha=0.6
)

critical_df = df[
    df['risk_flag'] == 'CRITICAL'
]

normal_df = df[
    df['risk_flag'] == 'NORMAL'
]

ax[1, 1].scatter(
    normal_df['discount_pct'] * 100,
    normal_df['net_revenue'],
    alpha=0.4,
    label='NORMAL'
)

ax[1, 1].scatter(
    critical_df['discount_pct'] * 100,
    critical_df['net_revenue'],
    alpha=0.7,
    label='CRITICAL'
)

ax[1, 1].set_title(
    "Discount % vs Net Revenue (Risk Tagged)",
    fontweight='bold'
)

ax[1, 1].set_xlabel(
    'Discount Percentage (%)'
)

ax[1, 1].set_ylabel(
    'Net Revenue ($)'
)

ax[1, 1].legend()

ax[1, 1].grid(
    True,
    linestyle=':',
    alpha=0.6
)

fig.suptitle(
    "METHYNIX E-COMMERCE REVENUE LEAKAGE AUDIT (2026)",
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()

fig.savefig(
    "dashboard.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()