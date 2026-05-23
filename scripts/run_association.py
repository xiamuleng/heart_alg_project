from pathlib import Path
from utils.association import run_apriori_on_heart
from utils.association import save_rules_to_sqlite


def main():
    out_dir = Path("outputs/associations")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_apriori_on_heart(csv_path="data/heart.csv", min_support=0.05, min_confidence=0.6, q=4)
    # 保存事务（可选，文件可能很宽）
    trans_fp = out_dir / "transactions.csv"
    results["transactions"].to_csv(trans_fp, index=False)
    print(f"Saved transactions -> {trans_fp}")

    fi = results.get("frequent_itemsets")
    rules = results.get("rules")
    if fi is not None:
        fi_fp = out_dir / "frequent_itemsets.csv"
        fi.to_csv(fi_fp, index=False)
        print(f"Saved frequent itemsets -> {fi_fp}")
    if rules is not None and not rules.empty:
        rules_fp = out_dir / "association_rules.csv"
        rules.to_csv(rules_fp, index=False)
        print(f"Saved rules -> {rules_fp}")
        # 将规则保存入 SQLite（默认表名 association_rules）
        try:
            inserted = save_rules_to_sqlite(rules)
            print(f"Saved {inserted} rules into SQLite table 'association_rules'.")
        except Exception as e:
            print(f"Failed to save rules to DB: {e}")
    else:
        print("No rules generated with the given thresholds.")

    if rules is not None and not rules.empty:
        print("Top 10 rules by confidence:")
        print(rules.sort_values("confidence", ascending=False).head(10)[["antecedents_str","consequents_str","support","confidence","lift"]])


if __name__ == "__main__":
    main()
