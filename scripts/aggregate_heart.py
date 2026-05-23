from pathlib import Path
from utils.aggregation import aggregate_heart_dataset, save_aggregations_to_sqlite


def main():
    out_dir = Path("outputs/aggregations")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = aggregate_heart_dataset(csv_path="data/heart.csv")
    for name, df in results.items():
        fp = out_dir / f"{name}.csv"
        df.to_csv(fp, index=False)
        print(f"Saved {name} -> {fp}")

    # 尝试将聚合写入 SQLite（覆盖同名表）
    try:
        inserted = save_aggregations_to_sqlite(results)
        print(f"Saved {inserted} aggregated rows into SQLite (tables prefixed with 'agg_').")
    except Exception as e:
        print(f"Failed to save aggregations to DB: {e}")

    print("Done. Preview:")
    for name, df in results.items():
        print("---", name, "---")
        print(df.head())


if __name__ == "__main__":
    main()
