import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2026)
n_orders=1000

regions=['East Africa','South America','Asia','Central Africa','Europe']
gateways=['PayPal','Stripe','M-Pesa','Crypto','Yas']
statuses=['Completed','Cancelled','Refunded','Failed']

raw_data={
    'order_id':[f'MTX-{1000+i}' for i in range(n_orders)],
    'order_date':pd.date_range(start='2026-01-01',periods=12,freq='h'),
    'region':np.random.choice(regions,n_orders,[0.4,0.05,0.2,0.15,0.2]),
    'payment_gateway':np.random.choice(gateways,n_orders,[0.2,0.1,0.4,0.1,0.2]),
    'unit_price':np.random.uniform(50.0,150.0,size=n_orders),
    'quantity':np.random.randint(1,11,size=n_orders),
    'discount_pct':np.random.normal(0.08,1,n_orders),
    'order_status':np.random.choice(statuses,n_orders,[0.5,0.1,0.2,0.2]),
    'shipping_cost':np.random.uniform(20.0,80.0,n_orders)
}

df=pd.DataFrame()

df.loc[df.sample(30).index,'payment_gateway']=np.nan
df.loc[df.sample(20).index,'unit_price']=np.nan

df['unit_price']=df['unit_price'].astype(str)
df[df.sample(10).index,'unit_price']="UNKNOWN"

print("The generated data initially")
print(df)

median_price_per_region=df.groupby('region')['unit_price'].median()
df['unit_price']=pd.to_numeric(df['unit_price']).fillna(median_price_per_region)

payment_mode=df['payment_gateway'].mode()[0]
df['payment_gateway']=df['payment_gateway'].fillna[payment_mode]

print('='*40)
print("After some cleaning of data")
print(df)
print('='*40)

#Adding other columns
df['gross_revenue']=df['unit_price']*df['quantity']
df['discount_amount']=df['gross_revenue']*df['discount_pct']
df['net_revenue']=df['gross_revenue']-df['discount_amount']