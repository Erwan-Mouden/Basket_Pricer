from pricer.data_loader import load_prices, load_rate_curve, load_vol_curves

# Prix
prices = load_prices("data/data_CAC40.xlsx")
print("Nb dates :", len(prices))
print("Tickers  :", list(prices.columns))
print("as_of    :", prices.index[-1].date())

# Taux
rate_curve = load_rate_curve("data/vol_impli.xlsx")
print("\nRate curve OK")
print("DF(1Y) :", round(rate_curve.df(1.0), 4))

# Vols
as_of = prices.index[-1].to_pydatetime()
vol_curves = load_vol_curves("data/vol_impli.xlsx", as_of)
print("\nVol curves OK")
print("Nb tickers :", len(vol_curves))
print("Tickers vol:", list(vol_curves.keys()))