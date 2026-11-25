import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas fila a fila. Aplicar despues de data.clean()."""
    out = df.copy()
    out["limit_to_income"] = out["proposed_credit_limit"] / out["income"].where(out["income"] > 0)
    out["vel_6h_vs_24h"] = out["velocity_6h"] / out["velocity_24h"].where(out["velocity_24h"] > 0)
    out["vel_24h_vs_4w"] = out["velocity_24h"] / out["velocity_4w"].where(out["velocity_4w"] > 0)
    out["balcon_negativo"] = (out["intended_balcon_amount"] < 0).astype(int)
    out["sin_direccion_previa"] = out["prev_address_months_count"].isna().astype(int)
    out["sin_cuenta_previa"] = out["bank_months_count"].isna().astype(int)
    out["income_band"] = out["income"].round(1).astype(str).astype("category")
    return out