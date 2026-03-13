"""
=============================================================
Silver Layer — CI/CD Test Cases
=============================================================
Run: In a Databricks notebook or via pytest on cluster
      %pip install pytest
      import pytest; pytest.main(["-v", "/path/to/test_silver.py"])

Tests cover:
  - 04_Silver1_transformations_dlt
  - Clean tables: slv_transactions, slv_returns, slv_products,
    slv_customers, slv_stores, slv_regions, slv_calendar,
    slv_kafka_orders, slv_kafka_inventory
  - Quarantine tables: quarantine_transactions, quarantine_returns,
    quarantine_kafka_orders, quarantine_kafka_inventory
  - SCD Type 2: slv_customers
=============================================================
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

CATALOG = "maven_catalog"
SILVER = "silver_schema"
BRONZE = "bronze_schema"


def s_table(name):
    return spark.read.table(f"{CATALOG}.{SILVER}.{name}")


def b_table(name):
    return spark.read.table(f"{CATALOG}.{BRONZE}.{name}")


# ============================================================
# 1. TABLE EXISTENCE
# ============================================================
class TestSilverTableExistence:
    """Verify all Silver tables exist and are non-empty."""

    @pytest.mark.parametrize("table_name", [
        "slv_transactions",
        "slv_returns",
        "slv_stores",
        "slv_regions",
        "slv_calendar",
        "slv_customers",
    ])
    def test_batch_silver_table_exists(self, table_name):
        df = s_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    @pytest.mark.parametrize("table_name", [
        "slv_kafka_orders",
        "slv_kafka_inventory",
    ])
    def test_kafka_silver_table_exists(self, table_name):
        df = s_table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    @pytest.mark.parametrize("table_name", [
        "quarantine_transactions",
        "quarantine_returns",
        "quarantine_kafka_orders",
        "quarantine_kafka_inventory",
    ])
    def test_quarantine_table_exists(self, table_name):
        """Quarantine tables can be empty — just verify they exist."""
        df = s_table(table_name)
        assert df is not None


# ============================================================
# 2. SCHEMA VALIDATION
# ============================================================
class TestSilverSchema:
    """Verify transformed columns exist after Silver processing."""

    def test_slv_transactions_schema(self):
        cols = s_table("slv_transactions").columns
        expected = [
            "transaction_date", "stock_date", "product_id",
            "customer_id", "store_id", "quantity",
            "_silver_timestamp"
        ]
        for c in expected:
            assert c in cols, f"Missing: {c} in slv_transactions"

    def test_slv_returns_schema(self):
        cols = s_table("slv_returns").columns
        for c in ["return_date", "product_id", "store_id", "quantity"]:
            assert c in cols, f"Missing: {c} in slv_returns"

    def test_slv_customers_scd2_columns(self):
        """SCD2 must have tracking columns."""
        cols = s_table("slv_customers").columns
        for c in ["customer_id", "__START_AT", "__END_AT", "__IS_CURRENT"]:
            assert c in cols, f"Missing SCD2 column: {c} in slv_customers"

    def test_slv_stores_has_region_data(self):
        cols = s_table("slv_stores").columns
        for c in ["store_id", "store_name", "sales_district", "sales_region"]:
            assert c in cols, f"Missing: {c} in slv_stores"

    def test_slv_calendar_has_dimensions(self):
        cols = s_table("slv_calendar").columns
        for c in ["date", "year", "month", "month_name", "quarter",
                   "day_of_week", "is_weekend", "fiscal_year"]:
            assert c in cols, f"Missing: {c} in slv_calendar"

    def test_slv_kafka_orders_enriched_columns(self):
        cols = s_table("slv_kafka_orders").columns
        for c in ["order_id", "total_amount", "product_id", "customer_id",
                   "store_id", "quantity", "event_time"]:
            assert c in cols, f"Missing: {c} in slv_kafka_orders"

    def test_slv_kafka_inventory_enriched_columns(self):
        cols = s_table("slv_kafka_inventory").columns
        for c in ["current_stock", "quantity_change", "change_type",
                   "product_id", "store_id", "stock_status", "is_low_stock"]:
            assert c in cols, f"Missing: {c} in slv_kafka_inventory"


# ============================================================
# 3. QUARANTINE VALIDATION
# ============================================================
class TestSilverQuarantine:
    """Verify quarantine tables have proper metadata columns."""

    @pytest.mark.parametrize("q_table", [
        "quarantine_transactions",
        "quarantine_returns",
        "quarantine_kafka_orders",
        "quarantine_kafka_inventory",
    ])
    def test_quarantine_has_reason_column(self, q_table):
        cols = s_table(q_table).columns
        assert "_quarantine_reasons" in cols, \
            f"{q_table} missing _quarantine_reasons"
        assert "_quarantine_timestamp" in cols, \
            f"{q_table} missing _quarantine_timestamp"

    def test_quarantine_reasons_not_empty(self):
        """If quarantine has rows, reasons must be populated."""
        df = s_table("quarantine_transactions")
        if df.count() > 0:
            null_reasons = df.filter("_quarantine_reasons IS NULL").count()
            assert null_reasons == 0, \
                f"quarantine_transactions: {null_reasons} rows with null reasons"

    def test_bronze_silver_quarantine_balance(self):
        """Bronze rows = Silver clean + Quarantine rows (approximately)."""
        brz_count = b_table("brz_transactions").count()
        slv_count = s_table("slv_transactions").count()
        q_count = s_table("quarantine_transactions").count()
        diff = abs(brz_count - (slv_count + q_count))
        pct = diff / brz_count * 100 if brz_count > 0 else 0
        assert pct < 5, (
            f"Row balance off by {pct:.1f}%: "
            f"Bronze={brz_count}, Silver={slv_count}, Quarantine={q_count}"
        )


# ============================================================
# 4. DATA QUALITY — NULL CHECKS
# ============================================================
class TestSilverDataQuality:
    """Verify Silver tables have no nulls in critical columns."""

    def test_slv_transactions_no_null_keys(self):
        df = s_table("slv_transactions")
        for c in ["product_id", "customer_id", "store_id"]:
            nulls = df.filter(f"{c} IS NULL").count()
            assert nulls == 0, f"slv_transactions: {nulls} null {c}"

    def test_slv_transactions_positive_quantity(self):
        bad = s_table("slv_transactions").filter("quantity <= 0").count()
        assert bad == 0, f"slv_transactions: {bad} rows with quantity <= 0"

    def test_slv_returns_no_null_keys(self):
        df = s_table("slv_returns")
        for c in ["product_id", "store_id"]:
            nulls = df.filter(f"{c} IS NULL").count()
            assert nulls == 0, f"slv_returns: {nulls} null {c}"

    def test_slv_stores_no_null_store_id(self):
        nulls = s_table("slv_stores").filter("store_id IS NULL").count()
        assert nulls == 0, "slv_stores has null store_ids"

    def test_slv_regions_no_null_region_id(self):
        nulls = s_table("slv_regions").filter("region_id IS NULL").count()
        assert nulls == 0, "slv_regions has null region_ids"

    def test_slv_kafka_orders_no_null_order_id(self):
        nulls = s_table("slv_kafka_orders").filter("order_id IS NULL").count()
        assert nulls == 0, "slv_kafka_orders has null order_ids"

    def test_slv_kafka_inventory_non_negative_stock(self):
        bad = s_table("slv_kafka_inventory").filter("current_stock < 0").count()
        assert bad == 0, f"slv_kafka_inventory: {bad} rows with negative stock"


# ============================================================
# 5. TYPE CASTING VALIDATION
# ============================================================
class TestSilverTypeCasting:
    """Verify data types were correctly cast during Silver transformation."""

    def test_transactions_date_type(self):
        dtype = dict(s_table("slv_transactions").dtypes).get("transaction_date")
        assert dtype == "date", f"Expected date, got {dtype}"

    def test_transactions_quantity_type(self):
        dtype = dict(s_table("slv_transactions").dtypes).get("quantity")
        assert dtype in ("int", "integer", "bigint"), f"Expected int, got {dtype}"

    def test_kafka_orders_amount_type(self):
        dtype = dict(s_table("slv_kafka_orders").dtypes).get("total_amount")
        assert dtype in ("double", "decimal(18,2)", "decimal(10,0)"), \
            f"Expected numeric, got {dtype}"


# ============================================================
# 6. SCD TYPE 2 — slv_customers
# ============================================================
class TestSilverSCD2:
    """Verify SCD Type 2 on slv_customers."""

    def test_scd2_has_current_records(self):
        current = s_table("slv_customers").filter(
            "__IS_CURRENT = true"
        ).count()
        assert current > 0, "No current records in slv_customers SCD2"

    def test_scd2_current_rows_unique(self):
        """Each customer_id should have exactly 1 current record."""
        df = s_table("slv_customers").filter("__IS_CURRENT = true")
        total = df.count()
        unique = df.select("customer_id").distinct().count()
        assert total == unique, (
            f"SCD2 duplicate current rows: {total} rows, {unique} unique IDs"
        )

    def test_scd2_end_date_null_for_current(self):
        """Current records must have NULL __END_AT."""
        bad = s_table("slv_customers").filter(
            "__IS_CURRENT = true AND __END_AT IS NOT NULL"
        ).count()
        assert bad == 0, f"SCD2: {bad} current rows have non-null END_AT"


# ============================================================
# 7. NO DUPLICATES IN CLEAN TABLES
# ============================================================
class TestSilverNoDuplicates:

    def test_slv_stores_unique(self):
        df = s_table("slv_stores")
        assert df.count() == df.dropDuplicates(["store_id"]).count(), \
            "slv_stores has duplicate store_ids"

    def test_slv_regions_unique(self):
        df = s_table("slv_regions")
        assert df.count() == df.dropDuplicates(["region_id"]).count(), \
            "slv_regions has duplicate region_ids"


# ============================================================
# 8. ENRICHMENT VALIDATION
# ============================================================
class TestSilverEnrichment:
    """Verify derived/enriched columns are populated."""

    def test_stores_has_sales_region(self):
        nulls = s_table("slv_stores").filter("sales_region IS NULL").count()
        total = s_table("slv_stores").count()
        assert nulls == 0, f"slv_stores: {nulls}/{total} null sales_region"

    def test_kafka_inventory_stock_status(self):
        """stock_status should be one of the defined buckets."""
        valid = ["Out of Stock", "Low Stock", "Normal", "Well Stocked"]
        bad = s_table("slv_kafka_inventory").filter(
            ~col("stock_status").isin(valid)
        ).count()
        assert bad == 0, f"slv_kafka_inventory: {bad} invalid stock_status"

    def test_calendar_year_range(self):
        df = s_table("slv_calendar")
        min_yr = df.agg({"year": "min"}).collect()[0][0]
        max_yr = df.agg({"year": "max"}).collect()[0][0]
        assert min_yr >= 1997, f"Calendar min year {min_yr} < 1997"
        assert max_yr <= 1999, f"Calendar max year {max_yr} > 1999"
