import os
import pandas as pd
from typing import Dict, List, Optional, Any

from . import db_utils


def _default_age_bins() -> (List[int], List[str]):
    bins = [0, 30, 40, 50, 60, 70, 200]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    return bins, labels


def aggregate(df: pd.DataFrame, groupby: List[str], agg_dict: Dict[str, Any]) -> pd.DataFrame:
    """通用分组聚合。

    Args:
        df: 输入的 DataFrame。
        groupby: 用于分组的列名列表。
        agg_dict: 传给 ``DataFrame.groupby(...).agg(...)`` 的聚合字典。

    Returns:
        重置索引后的聚合结果 DataFrame。
    """
    if not isinstance(groupby, list):
        groupby = [groupby]
    res = df.groupby(groupby).agg(agg_dict)
    # 如果出现 MultiIndex 列名，尽量扁平化
    if isinstance(res.columns, pd.MultiIndex):
        res.columns = ["_".join(map(str, c)).strip() for c in res.columns.values]
    return res.reset_index()


def aggregate_by_age_group(
    df: pd.DataFrame,
    age_col: str = "age",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    agg_dict: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """按年龄段聚合并返回结果。

    默认年龄段：`<30, 30-39, 40-49, 50-59, 60-69, 70+`。

    Args:
        df: 输入 DataFrame，须包含年龄列。
        age_col: 年龄列名，默认 `age`。
        bins: 年龄分箱（包含右端点的边界列表），默认使用内置分箱。
        labels: 分箱标签，默认使用内置标签。
        agg_dict: 聚合字典，若为 None，则默认返回每组的样本数和（若存在）`target` 的均值。

    Returns:
        包含 `age_group` 的聚合结果 DataFrame。
    """
    if age_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在年龄列: {age_col}")
    if bins is None or labels is None:
        bins, labels = _default_age_bins()
    df = df.copy()
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    if agg_dict is None:
        agg_dict = {age_col: "count"}
        if "target" in df.columns:
            agg_dict["target"] = "mean"
    return aggregate(df, ["age_group"], agg_dict)


def aggregate_by_gender(
    df: pd.DataFrame, gender_col: str = "sex", agg_dict: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """按性别聚合并返回结果。

    Args:
        df: 输入 DataFrame。
        gender_col: 性别列名，默认 `sex`。
        agg_dict: 聚合字典，若为 None，则返回样本数和（若存在）`target` 的均值。

    Returns:
        聚合结果 DataFrame。
    """
    if gender_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在性别列: {gender_col}")
    if agg_dict is None:
        agg_dict = {gender_col: "count"}
        if "target" in df.columns:
            agg_dict["target"] = "mean"
    return aggregate(df, [gender_col], agg_dict)


def aggregate_by_age_and_gender(
    df: pd.DataFrame,
    age_col: str = "age",
    gender_col: str = "sex",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    agg_dict: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """按年龄段与性别联合分组聚合。

    返回的索引顺序为 `age_group` 然后 `gender_col`。
    """
    if age_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在年龄列: {age_col}")
    if gender_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在性别列: {gender_col}")
    if bins is None or labels is None:
        bins, labels = _default_age_bins()
    df = df.copy()
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    if agg_dict is None:
        agg_dict = {age_col: "count"}
        if "target" in df.columns:
            agg_dict["target"] = "mean"
    return aggregate(df, ["age_group", gender_col], agg_dict)


__all__ = [
    "aggregate",
    "aggregate_by_age_group",
    "aggregate_by_gender",
    "aggregate_by_age_and_gender",
    "aggregate_heart_dataset",
    "aggregate_for_app",
    "save_aggregations_to_sqlite",
]


def aggregate_heart_dataset(
    csv_path: str = "data/heart.csv",
    age_col: str = "age",
    gender_col: str = "sex",
    extra_metrics: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """基于 `data/heart.csv` 的常用聚合汇总。

    返回一个字典，包含按年龄段、按性别、按年龄段+性别的聚合表。每个表默认包含：
      - `count`：样本数
      - `target_mean`：若存在 `target` 列则计算均值
      - `chol_mean`：若存在 `chol` 列则计算均值
      - `trestbps_mean`：若存在 `trestbps` 列则计算均值
      - `thalach_mean`：若存在 `thalach` 列则计算均值

    Args:
        csv_path: heart.csv 文件路径，相对于项目根或绝对路径。
        age_col: 年龄列名。
        gender_col: 性别列名。
        extra_metrics: 要额外计算的列名列表（均值）。

    Returns:
        dict: keys = `by_age_group`, `by_gender`, `by_age_and_gender`，values 为 DataFrame。
    """
    # 优先使用 data/cleaned_heart.csv（如果存在）；兼容相对或绝对路径
    cleaned_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_heart.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        df = pd.read_csv(csv_path)
    metrics = ["target", "chol", "trestbps", "thalach"]
    if extra_metrics:
        metrics = list(dict.fromkeys(metrics + extra_metrics))

    # 构造聚合字典
    agg_dict = {age_col: "count"}
    for col in metrics:
        if col in df.columns:
            agg_dict[f"{col}"] = "mean"

    res_age = aggregate_by_age_group(df, age_col=age_col, agg_dict=agg_dict)
    res_gender = aggregate_by_gender(df, gender_col=gender_col, agg_dict=agg_dict)
    res_age_gender = aggregate_by_age_and_gender(
        df, age_col=age_col, gender_col=gender_col, agg_dict=agg_dict
    )

    # 重命名 count 列为 `count`（age_col 列可能被用作 count 的键）
    def _normalize_count(df_res: pd.DataFrame):
        for c in df_res.columns:
            if c.endswith("count") and c != "count":
                df_res = df_res.rename(columns={c: "count"})
                break
        return df_res

    res_age = _normalize_count(res_age)
    res_gender = _normalize_count(res_gender)
    res_age_gender = _normalize_count(res_age_gender)

    return {"by_age_group": res_age, "by_gender": res_gender, "by_age_and_gender": res_age_gender}


def aggregate_for_app(csv_path: str = "data/heart.csv") -> Dict[str, pd.DataFrame]:
    """为 Streamlit 应用准备的聚合数据。

    返回字典包含：
      - `age`: 每个年龄分组的 `age_group`, `count`, `disease_rate`（0-1）
      - `gender`: 每个性别的 `sex`, `count`, `disease_rate`（0-1）
      - `age_gender`: 年龄段与性别联合汇总（包含 `age_group`, `sex`, `count`, `disease_rate`）
    """
    aggs = aggregate_heart_dataset(csv_path=csv_path)

    def _make_rate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "target" in df.columns:
            df["disease_rate"] = df["target"].astype(float)
        elif "target_mean" in df.columns:
            df["disease_rate"] = df["target_mean"].astype(float)
        else:
            df["disease_rate"] = pd.NA
        # 确保 count 列存在并为 int
        if "count" not in df.columns:
            # 尝试查找以 age 或 sex 命名的 count 列
            for c in df.columns:
                if str(c).endswith("count"):
                    df = df.rename(columns={c: "count"})
                    break
        if "count" in df.columns:
            df["count"] = df["count"].astype(int)
        return df

    age_df = _make_rate(aggs.get("by_age_group", pd.DataFrame()))
    gender_df = _make_rate(aggs.get("by_gender", pd.DataFrame()))
    age_gender_df = _make_rate(aggs.get("by_age_and_gender", pd.DataFrame()))

    # 对 gender_df 添加可读标签
    if "sex" in gender_df.columns:
        gender_df["sex_label"] = gender_df["sex"].map({1: "男", 0: "女"})

    return {"age": age_df, "gender": gender_df, "age_gender": age_gender_df}


def save_aggregations_to_sqlite(aggregations: Dict[str, pd.DataFrame], prefix: str = "agg_") -> int:
    """将聚合结果写入项目 SQLite 数据库（由 `utils.db_utils` 指定）。

    每个 DataFrame 会写入一张表，表名为 `prefix` + key（例如 `agg_by_age_group`）。
    如果表已存在则用 `replace` 模式覆盖。

    返回写入的总行数。
    """
    if not aggregations:
        return 0
    conn = db_utils.get_connection()
    total = 0
    try:
        for name, df in aggregations.items():
            table_name = f"{prefix}{name}"
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            total += len(df)
        conn.commit()
    finally:
        conn.close()
    return total
