import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict

from mlxtend.frequent_patterns import apriori, association_rules


def _default_age_bins() -> Tuple[List[int], List[str]]:
    bins = [0, 30, 40, 50, 60, 70, 200]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    return bins, labels


def df_to_transaction_df(
    df: pd.DataFrame,
    age_col: str = "age",
    categorical_cols: Optional[List[str]] = None,
    numeric_to_quantile: Optional[List[str]] = None,
    q: int = 4,
) -> pd.DataFrame:
    """将心脏数据集转换为适用于 Apriori 的一热编码事务表。

    - 年龄按固定区间分箱 (见 `_default_age_bins`)。
    - `categorical_cols` 中的列按取值编码为 `col=value` 项目。
    - `numeric_to_quantile` 中的数值列按 `q` 分位切分并编码为 `col=Qk` 项目。

    返回值为每列为项目、值为布尔(或 0/1) 的 DataFrame。
    """
    df = df.copy()
    items = []

    # 年龄分箱
    if age_col in df.columns:
        bins, labels = _default_age_bins()
        df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        items.append(pd.get_dummies(df["age_group"], prefix="age"))

    # 分类列按值编码
    if categorical_cols is None:
        categorical_cols = [
            c
            for c in [
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "slope",
                "ca",
                "thal",
                "target",
            ]
            if c in df.columns
        ]
    for c in categorical_cols:
        if c in df.columns:
            items.append(pd.get_dummies(df[c], prefix=c))

    # 数值列按分位数分箱
    if numeric_to_quantile is None:
        numeric_to_quantile = [c for c in ["chol", "trestbps", "thalach"] if c in df.columns]
    for c in numeric_to_quantile:
        try:
            cut = pd.qcut(df[c].rank(method="first"), q, labels=[f"Q{i+1}" for i in range(q)])
            items.append(pd.get_dummies(cut, prefix=c))
        except Exception:
            # 如果 qcut 失败则跳过
            continue

    if not items:
        raise ValueError("没有可用的列来构建事务项，请检查输入 DataFrame")

    trans = pd.concat(items, axis=1)
    # 将列名统一为字符串，值为 0/1
    trans = trans.astype(int)
    return trans


def mine_apriori_rules(
    trans_df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.6,
    use_colnames: bool = True,
) -> Dict[str, pd.DataFrame]:
    """在事务表上运行 Apriori 与 association_rules，返回频繁项集与规则。

    Args:
        trans_df: 事务一热表（0/1 或 bool）。
        min_support: 最小支持度阈值。
        min_confidence: 最小置信度阈值。

    Returns:
        dict 包含 `frequent_itemsets` 和 `rules` 两个 DataFrame。
    """
    frequent_itemsets = apriori(trans_df, min_support=min_support, use_colnames=use_colnames)
    if frequent_itemsets.empty:
        return {"frequent_itemsets": frequent_itemsets, "rules": pd.DataFrame()}

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    # 让输出更可读：把 itemsets 转为字符串
    if not frequent_itemsets.empty:
        frequent_itemsets["itemset_str"] = frequent_itemsets["itemsets"].apply(lambda s: ",".join(sorted(s)))
    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ",".join(sorted(s)))
        rules["consequents_str"] = rules["consequents"].apply(lambda s: ",".join(sorted(s)))

    return {"frequent_itemsets": frequent_itemsets, "rules": rules}


def run_apriori_on_heart(
    csv_path: str = "data/heart.csv",
    min_support: float = 0.05,
    min_confidence: float = 0.6,
    q: int = 4,
) -> Dict[str, pd.DataFrame]:
    """读取 heart.csv，构建事务并挖掘关联规则。

    返回字典：`transactions`, `frequent_itemsets`, `rules`。
    """
    # 优先使用 cleaned_heart.csv（若存在），否则使用传入路径
    import os

    cleaned = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_heart.csv")
    if os.path.exists(cleaned):
        df = pd.read_csv(cleaned)
    else:
        df = pd.read_csv(csv_path)
    trans = df_to_transaction_df(df, q=q)
    mined = mine_apriori_rules(trans, min_support=min_support, min_confidence=min_confidence)
    result = {"transactions": trans}
    result.update(mined)
    return result


def save_rules_to_sqlite(rules_df: pd.DataFrame, table_name: str = "association_rules") -> int:
    """将规则保存到默认的 SQLite 库（由 `utils.db_utils.DB_PATH` 指定）。

    保存字段：`antecedents_str`, `consequents_str`, `support`, `confidence`, `lift`。

    返回插入的行数。
    """
    if rules_df is None or rules_df.empty:
        return 0

    # 选择并重命名列以便于在数据库中存储
    cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    missing = [c for c in cols if c not in rules_df.columns]
    if missing:
        raise ValueError(f"规则数据缺少列: {missing}")

    store_df = rules_df[cols].copy()

    # 延迟导入以避免循环依赖
    from . import db_utils

    conn = db_utils.get_connection()
    try:
        store_df.to_sql(table_name, conn, if_exists="replace", index=False)
        return len(store_df)
    finally:
        conn.close()


__all__ = [
    "df_to_transaction_df",
    "mine_apriori_rules",
    "run_apriori_on_heart",
    "save_rules_to_sqlite",
    "apriori_for_app",
]


def apriori_for_app(min_support: float = 0.05, min_confidence: float = 0.6, q: int = 4):
    """为 Streamlit 应用返回可展示与下载的规则结果。

    Returns:
        rules_df: pandas.DataFrame - 包含 `antecedents_str`, `consequents_str`, `support`, `confidence`, `lift`
        csv_bytes: bytes - 可直接作为下载数据传给 Streamlit 的 `data` 参数
    """
    res = run_apriori_on_heart(min_support=min_support, min_confidence=min_confidence, q=q)
    rules = res.get("rules")
    if rules is None or rules.empty:
        return rules, b""

    out_df = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].copy()
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    return out_df, csv_bytes
