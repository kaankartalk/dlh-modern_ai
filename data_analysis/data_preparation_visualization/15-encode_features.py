#!/usr/bin/env python3
"""Encode features for modeling using scikit-learn."""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode categorical features for modeling.

    Args:
        df: pandas DataFrame.

    Returns:
        A tuple of (encoded DataFrame, fitted LabelEncoder for Churn,
        fitted OrdinalEncoder for the binary columns, fitted
        OrdinalEncoder for TenureGroup).
    """
    df = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df['Churn'] = churn_le.fit_transform(df['Churn'])

    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling',
                   'SeniorCitizen']
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for col in binary_cols:
        df[col] = binary_oe.fit_transform(df[[col]]).astype(int)

    tenure_oe = preprocessing.OrdinalEncoder()
    df['TenureGroup'] = tenure_oe.fit_transform(
        df[['TenureGroup']]).astype(int)

    onehot_cols = ['Contract', 'PaymentMethod']
    dummies = pd.get_dummies(df[onehot_cols], drop_first=True, dtype=int)
    df = df.drop(columns=onehot_cols)
    df = pd.concat([df, dummies], axis=1)

    return df, churn_le, binary_oe, tenure_oe
