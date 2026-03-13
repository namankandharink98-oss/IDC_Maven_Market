"""
=============================================================
Gold Layer — CI/CD Test Cases
=============================================================
Run: In a Databricks notebook or via pytest on cluster
      %pip install pytest
      import pytest; pytest.main(["-v", "/path/to/test_gold.py"])

Tests cover:
  - 05_gold_star_schema
  - Dimensions: dim_customers, dim_products, dim_stores, dim_calendar
  - Facts: fact_sales, fact_returns, fact_kafka_orders, fact_inventory_movements
  - Aggregates: agg_daily_sales, agg_monthly_sales, agg_product_performance,
    agg_store_performance, agg_kafka_throughput_min, agg_inventory_alerts
  - KPI: kpi_executive_summary
  - Audit: gold_pipeline_health_audit
=============================================================
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

CATALOG = "maven_catalog"
GOLD = "gold_schema"
SILVER = "silver_schema"


def g_table(name):
    return spark.read.table(f"{CATALOG}.{GOLD}.{name}")


def s_table(name):
    return spark.read.table(f"{CATALOG}.{SILVER}.{name}")


# ============================================================
# 1. TABLE EXISTENCE
# ============================================================
class TestGoldTableExistence:
    """Verify all 16 Gold tables exist."""

    # Dimensions
    @pytest.mark.parametrize("table_name", [
        "dim_customers",
        "dim_products",
        "dim_stores",
        "dim_calendar",
    ])
    def test_dimension_exists(self, table_name):
        df = g_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    # Facts
    @pytest.mark.parametrize("table_name", [
        "fact_sales",
        "fact_returns",
        "fact_kafka_orders",
        "fact_inventory_movements",
    ])
    def test_fact_exists(self, table_name):
        df = g_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    # Aggregates
    @pytest.mark.parametrize("table_name", [
        "agg_daily_sales",
        "agg_monthly_sales",
        "agg_product_performance",
        "agg_store_performance",
        "agg_kafka_throughput_min",
        "agg_inventory_alerts",
    ])
    def test_aggregate_exists(self, table_name):
        df = g_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    # KPI + Audit
    @pytest.mark.parametrize("table_name", [
        "kpi_executive_summary",
        "gold_pipeline_health_audit",
    ])
    def test_kpi_audit_exists(self, table_name):
        df = g_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"


# ============================================================
# 2. SCHEMA VALIDATION — DIMENSIONS
# ============================================================
class TestGoldDimensionSchema:

    def test_dim_customers_schema(self):
        cols = g_table("dim_customers").columns
        for c in ["customer_id", "full_name", "customer_country",
                   "member_card", "gender", "age"]:
            assert c in cols, f"Missing: {c} in dim_customers"

    def test_dim_products_schema(self):
        cols = g_table("dim_products").columns
        for c in ["product_id", "product_brand", "product_name",
                   "product_retail_price", "product_cost",
                   "profit_margin", "price_tier"]:
            assert c in cols, f"Missing: {c} in dim_products"

    def test_dim_stores_schema(self):
        cols = g_table("dim_stores").columns
        for c in ["store_id", "store_name", "store_country",
                   "sales_district", "sales_region", "total_sqft"]:
            assert c in cols, f"Missing: {c} in dim_stores"

    def test_dim_calendar_schema(self):
        cols = g_table("dim_calendar").columns
        for c in ["date", "year", "month", "quarter",
                   "day_of_week", "is_weekend"]:
            assert c in cols, f"Missing: {c} in dim_calendar"


# ============================================================
# 3. SCHEMA VALIDATION — FACTS
# ============================================================
class TestGoldFactSchema:

    def test_fact_sales_schema(self):
        cols = g_table("fact_sales").columns
        for c in ["transaction_date", "product_id", "customer_id",
                   "store_id", "quantity", "total_revenue",
                   "total_cost", "total_profit"]:
            assert c in cols, f"Missing: {c} in fact_sales"

    def test_fact_returns_schema(self):
        cols = g_table("fact_returns").columns
        for c in ["return_date", "product_id", "store_id", "quantity"]:
            assert c in cols, f"Missing: {c} in fact_returns"

    def test_fact_kafka_orders_schema(self):
        cols = g_table("fact_kafka_orders").columns
        for c in ["order_id", "product_id", "store_id",
                   "quantity", "total_amount", "event_time"]:
            assert c in cols, f"Missing: {c} in fact_kafka_orders"

    def test_fact_inventory_movements_schema(self):
        cols = g_table("fact_inventory_movements").columns
        for c in ["product_id", "store_id", "current_stock",
                   "change_type", "event_time"]:
            assert c in cols, f"Missing: {c} in fact_inventory_movements"


# ============================================================
# 4. FACT TABLE DATA QUALITY
# ============================================================
class TestGoldFactDataQuality:

    def test_fact_sales_revenue_positive(self):
        bad = g_table("fact_sales").filter("total_revenue <= 0").count()
        total = g_table("fact_sales").count()
        pct = bad / total * 100 if total > 0 else 0
        assert pct < 1, f"fact_sales: {pct:.1f}% rows with revenue <= 0"

    def test_fact_sales_profit_calculation(self):
        """profit = revenue - cost (spot check top 100 rows)."""
        df = g_table("fact_sales").limit(100).collect()
        for row in df:
            if row["total_revenue"] and row["total_cost"] and row["total_profit"]:
                expected = round(row["total_revenue"] - row["total_cost"], 2)
                actual = round(row["total_profit"], 2)
                assert abs(expected - actual) < 0.02, (
                    f"Profit mismatch: {actual} != {expected}"
                )

    def test_fact_sales_no_null_keys(self):
        df = g_table("fact_sales")
        for c in ["product_id", "customer_id", "store_id"]:
            nulls = df.filter(f"{c} IS NULL").count()
            assert nulls == 0, f"fact_sales: {nulls} null {c}"

    def test_fact_returns_no_null_keys(self):
        df = g_table("fact_returns")
        for c in ["product_id", "store_id"]:
            nulls = df.filter(f"{c} IS NULL").count()
            assert nulls == 0, f"fact_returns: {nulls} null {c}"


# ============================================================
# 5. DIMENSION UNIQUENESS
# ============================================================
class TestGoldDimensionUniqueness:

    def test_dim_customers_unique(self):
        df = g_table("dim_customers")
        assert df.count() == df.select("customer_id").distinct().count(), \
            "dim_customers has duplicate customer_ids"

    def test_dim_products_unique(self):
        df = g_table("dim_products")
        assert df.count() == df.select("product_id").distinct().count(), \
            "dim_products has duplicate product_ids"

    def test_dim_stores_unique(self):
        df = g_table("dim_stores")
        assert df.count() == df.select("store_id").distinct().count(), \
            "dim_stores has duplicate store_ids"

    def test_dim_calendar_unique(self):
        df = g_table("dim_calendar")
        assert df.count() == df.select("date").distinct().count(), \
            "dim_calendar has duplicate dates"


# ============================================================
# 6. REFERENTIAL INTEGRITY (FK checks)
# ============================================================
class TestGoldReferentialIntegrity:
    """Verify fact table FKs exist in dimension tables."""

    def test_fact_sales_product_fk(self):
        fact_ids = g_table("fact_sales").select("product_id").distinct()
        dim_ids = g_table("dim_products").select("product_id").distinct()
        orphans = fact_ids.subtract(dim_ids).count()
        assert orphans == 0, f"fact_sales: {orphans} product_ids not in dim_products"

    def test_fact_sales_customer_fk(self):
        fact_ids = g_table("fact_sales").select("customer_id").distinct()
        dim_ids = g_table("dim_customers").select("customer_id").distinct()
        orphans = fact_ids.subtract(dim_ids).count()
        assert orphans == 0, f"fact_sales: {orphans} customer_ids not in dim_customers"

    def test_fact_sales_store_fk(self):
        fact_ids = g_table("fact_sales").select("store_id").distinct()
        dim_ids = g_table("dim_stores").select("store_id").distinct()
        orphans = fact_ids.subtract(dim_ids).count()
        assert orphans == 0, f"fact_sales: {orphans} store_ids not in dim_stores"

    def test_fact_returns_product_fk(self):
        fact_ids = g_table("fact_returns").select("product_id").distinct()
        dim_ids = g_table("dim_products").select("product_id").distinct()
        orphans = fact_ids.subtract(dim_ids).count()
        assert orphans == 0, f"fact_returns: {orphans} product_ids not in dim_products"

    def test_fact_returns_store_fk(self):
        fact_ids = g_table("fact_returns").select("store_id").distinct()
        dim_ids = g_table("dim_stores").select("store_id").distinct()
        orphans = fact_ids.subtract(dim_ids).count()
        assert orphans == 0, f"fact_returns: {orphans} store_ids not in dim_stores"


# ============================================================
# 7. AGGREGATE TABLE VALIDATION
# ============================================================
class TestGoldAggregates:

    def test_agg_daily_sales_revenue_matches_fact(self):
        """Total revenue in agg_daily_sales should match fact_sales."""
        agg_rev = (
            g_table("agg_daily_sales")
            .agg({"total_revenue": "sum"}).collect()[0][0]
        )
        fact_rev = (
            g_table("fact_sales")
            .agg({"total_revenue": "sum"}).collect()[0][0]
        )
        if agg_rev and fact_rev:
            diff_pct = abs(agg_rev - fact_rev) / fact_rev * 100
            assert diff_pct < 1, (
                f"Revenue mismatch: agg={agg_rev:.2f}, fact={fact_rev:.2f}, "
                f"diff={diff_pct:.2f}%"
            )

    def test_agg_monthly_sales_not_empty(self):
        cnt = g_table("agg_monthly_sales").count()
        assert cnt > 0, "agg_monthly_sales is empty"

    def test_agg_product_performance_has_sales_region(self):
        cols = g_table("agg_product_performance").columns
        assert "sales_region" in cols, \
            "agg_product_performance missing sales_region column"

    def test_agg_store_performance_revenue_per_sqft(self):
        """revenue_per_sqft should be positive for all stores."""
        bad = g_table("agg_store_performance").filter(
            "revenue_per_sqft <= 0 OR revenue_per_sqft IS NULL"
        ).count()
        assert bad == 0, f"agg_store_performance: {bad} invalid revenue_per_sqft"

    def test_agg_kafka_throughput_not_empty(self):
        cnt = g_table("agg_kafka_throughput_min").count()
        assert cnt > 0, "agg_kafka_throughput_min is empty"


# ============================================================
# 8. KPI EXECUTIVE SUMMARY
# ============================================================
class TestGoldKPISummary:

    def test_kpi_has_yearly_rows(self):
        df = g_table("kpi_executive_summary")
        years = [r["year"] for r in df.select("year").distinct().collect()]
        assert 1997 in years or 1998 in years, \
            f"kpi_executive_summary missing expected years, got {years}"

    def test_kpi_profit_margin_valid(self):
        df = g_table("kpi_executive_summary")
        bad = df.filter("profit_margin_pct < 0 OR profit_margin_pct > 100").count()
        assert bad == 0, f"kpi: {bad} rows with invalid profit_margin_pct"

    def test_kpi_return_rate_valid(self):
        df = g_table("kpi_executive_summary")
        bad = df.filter("return_rate < 0 OR return_rate > 100").count()
        assert bad == 0, f"kpi: {bad} rows with invalid return_rate"


# ============================================================
# 9. PIPELINE HEALTH AUDIT
# ============================================================
class TestGoldAudit:

    def test_audit_table_not_empty(self):
        cnt = g_table("gold_pipeline_health_audit").count()
        assert cnt > 0, "gold_pipeline_health_audit is empty"

    def test_audit_has_expected_columns(self):
        cols = g_table("gold_pipeline_health_audit").columns
        # Should at least have table/layer/count info
        assert len(cols) >= 3, \
            f"Audit table has only {len(cols)} columns, expected >= 3"


# ============================================================
# 10. CROSS-LAYER ROW FLOW
# ============================================================
class TestGoldCrossLayerFlow:
    """Verify data flows correctly: Silver → Gold."""

    def test_fact_sales_count_matches_silver(self):
        """fact_sales should have same rows as slv_transactions (both cleaned)."""
        gold_cnt = g_table("fact_sales").count()
        silver_cnt = s_table("slv_transactions").count()
        diff_pct = abs(gold_cnt - silver_cnt) / silver_cnt * 100 if silver_cnt else 0
        assert diff_pct < 5, (
            f"Row count mismatch: Gold fact_sales={gold_cnt}, "
            f"Silver slv_transactions={silver_cnt}, diff={diff_pct:.1f}%"
        )

    def test_dim_customers_count_reasonable(self):
        """dim_customers (current only) should be <= total SCD2 rows."""
        dim_cnt = g_table("dim_customers").count()
        scd_total = s_table("slv_customers").count()
        assert dim_cnt <= scd_total, (
            f"dim_customers ({dim_cnt}) > total SCD2 rows ({scd_total})"
        )

    def test_dim_products_count_matches_silver(self):
        gold_cnt = g_table("dim_products").count()
        silver_cnt = s_table("slv_products").count()
        assert gold_cnt == silver_cnt, (
            f"dim_products={gold_cnt} != slv_products={silver_cnt}"
        )
