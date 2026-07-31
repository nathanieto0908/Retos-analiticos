
from statsmodels.tsa.stattools import ccf, adfuller


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f' Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
        stationary = 1
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    
        stationary = 0
    return stationary


############################################# 

import pandas as pd
import numpy as np
from scipy.stats import (
    pointbiserialr,
    pearsonr,
    spearmanr,
    chi2_contingency,
    f_oneway
)


def correlacion_vs_target(df,
                          target_col,
                          metodo_continuo='pearson'):
    """
    Calcula la asociación entre todas las variables y el target.

    Parámetros
    ----------
    df : DataFrame

    target_col : str
        Nombre del target

    metodo_continuo : {'pearson','spearman'}
        Método para variables numéricas cuando el target es continuo.

    Retorna
    -------
    DataFrame ordenado por fuerza de asociación.
    """

    resultados = []

    # -------------------------------
    # Detectar tipo de target
    # -------------------------------

    target = df[target_col]

    if pd.api.types.is_numeric_dtype(target):

        # Binario si solo tiene dos valores
        if target.nunique(dropna=True) == 2:
            tipo_target = 'binario'
        else:
            tipo_target = 'continuo'

    else:
        tipo_target = 'categorico'

    # -------------------------------
    # Recorrer variables
    # -------------------------------

    for col in df.columns:

        if col == target_col:
            continue

        try:

            es_numerica = pd.api.types.is_numeric_dtype(df[col])

            # =====================================================
            # TARGET BINARIO
            # =====================================================

            if tipo_target == 'binario':

                if es_numerica:

                    tmp = df[[col, target_col]].dropna()

                    if tmp[col].nunique() > 1:

                        corr, p = pointbiserialr(
                            tmp[target_col],
                            tmp[col]
                        )

                        resultados.append({
                            "variable": col,
                            "correlacion": abs(corr),
                            "metodo": "Point-Biserial",
                            "p_value": p
                        })

                else:

                    tabla = pd.crosstab(df[col], df[target_col])

                    if min(tabla.shape) > 1:

                        chi2, p, _, _ = chi2_contingency(tabla)

                        n = tabla.values.sum()

                        cramers = np.sqrt(
                            chi2 / (n * (min(tabla.shape)-1))
                        )

                        resultados.append({
                            "variable": col,
                            "correlacion": cramers,
                            "metodo": "Cramér's V",
                            "p_value": p
                        })

            # =====================================================
            # TARGET CONTINUO
            # =====================================================

            elif tipo_target == 'continuo':

                if es_numerica:

                    tmp = df[[col, target_col]].dropna()

                    if tmp[col].nunique() > 1:

                        if metodo_continuo == "spearman":
                            corr, p = spearmanr(
                                tmp[col],
                                tmp[target_col]
                            )
                            metodo = "Spearman"

                        else:
                            corr, p = pearsonr(
                                tmp[col],
                                tmp[target_col]
                            )
                            metodo = "Pearson"

                        resultados.append({
                            "variable": col,
                            "correlacion": abs(corr),
                            "metodo": metodo,
                            "p_value": p
                        })

                else:

                    tmp = df[[col, target_col]].dropna()

                    grupos = [
                        grupo[target_col].values
                        for _, grupo in tmp.groupby(col)
                    ]

                    if len(grupos) > 1:

                        F, p = f_oneway(*grupos)

                        ss_between = sum(
                            len(g) * (np.mean(g)-tmp[target_col].mean())**2
                            for g in grupos
                        )

                        ss_total = np.sum(
                            (tmp[target_col]-tmp[target_col].mean())**2
                        )

                        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

                        resultados.append({
                            "variable": col,
                            "correlacion": eta2,
                            "metodo": "Eta Squared",
                            "p_value": p
                        })

            # =====================================================
            # TARGET CATEGÓRICO (más de dos clases)
            # =====================================================

            else:

                tabla = pd.crosstab(df[col], df[target_col])

                if min(tabla.shape) > 1:

                    chi2, p, _, _ = chi2_contingency(tabla)

                    n = tabla.values.sum()

                    cramers = np.sqrt(
                        chi2 / (n * (min(tabla.shape)-1))
                    )

                    resultados.append({
                        "variable": col,
                        "correlacion": cramers,
                        "metodo": "Cramér's V",
                        "p_value": p
                    })

        except Exception as e:
            print(f"Error en {col}: {e}")

    resultado = pd.DataFrame(resultados)

    if not resultado.empty:
        resultado = resultado.sort_values(
            "correlacion",
            ascending=False
        ).reset_index(drop=True)

    return resultado