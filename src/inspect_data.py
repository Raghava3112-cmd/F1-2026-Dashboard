import pandas as pd

cp = pd.read_csv('output/championship_probabilities.csv')
drivers = pd.read_csv('data/raw/drivers.csv')[['driverId','surname']]
cp = cp.merge(drivers, left_on='driver_id', right_on='driverId', how='left')

pivot = cp.pivot_table(index='surname', columns='as_of_round',
                       values='championship_probability', fill_value=0)
contenders = pivot[pivot.max(axis=1) > 0.005].sort_values(by=16, ascending=False)

rounds = sorted(cp['as_of_round'].unique())
header = "Driver         " + "".join([f"  R{r:02d}" for r in rounds])
print("=== Driver Championship Win Probability by Round (%) ===")
print(header)
print("-" * len(header))
for drv, row in contenders.iterrows():
    vals = "".join([f"  {row.get(r, 0)*100:4.1f}" for r in rounds])
    print(f"{drv:<15}{vals}")

print()
print("=== Round-by-round extremes (vs legacy 99.9% at Round 1) ===")
for r in rounds:
    top1 = cp[cp['as_of_round']==r].nlargest(1, 'championship_probability')
    p = float(top1['championship_probability'].values[0]) * 100
    name = str(top1['surname'].values[0])
    print(f"  Round {r:02d}  leader={name:<12}  win_prob={p:5.1f}%")
