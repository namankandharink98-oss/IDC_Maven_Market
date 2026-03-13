"""
=============================================================
Bronze Layer — CI/CD Test Cases
=============================================================
Run: In a Databricks notebook or via pytest on cluster
      %pip install pytest
      import pytest; pytest.main(["-v", "/path/to/test_bronze.py"])

Tests cover:
  - 01_bronze_batch_ingestion (CSV Auto Loader: 5 tables)
  - 02_bronze_mongodb_ingestion (MongoDB via Fivetran: 2 tables)
  - 03_bronze_kafka_streaming (Kafka via Fivetran: 4 tables)
=============================================================
"""

import pytest
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

CATALOG = "maven_catalog"
SCHEMA = "bronze_schema"


def table(name):
    return spark.read.table(f"{CATALOG}.{SCHEMA}.{name}")


# ============================================================
# 1. TABLE EXISTENCE
# ============================================================
class TestBronzeTableExistence:
    """Verify all 11 Bronze tables exist and are readable."""

    # Batch CSV tables
    @pytest.mark.parametrize("table_name", [
        "brz_transactions",
        "brz_returns",
        "brz_stores",
        "brz_regions",
        "brz_calendar",
    ])
    def test_batch_table_exists(self, table_name):
        df = table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    # MongoDB tables
    @pytest.mark.parametrize("table_name", [
        "brz_products_mongo_dlt",
        "brz_customers_mongo_dlt",
    ])
    def test_mongo_table_exists(self, table_name):
        df = table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"

    # Kafka tables
    @pytest.mark.parametrize("table_name", [
        "brz_kafka_orders_raw",
        "brz_kafka_orders",
        "brz_kafka_inventory_raw",
        "brz_kafka_inventory",
    ])
    def test_kafka_table_exists(self, table_name):
        df = table(table_name)
        assert df is not None
        assert df.count() > 0, f"{table_name} is empty"


# ============================================================
# 2. ROW COUNTS — Minimum expected rows
# ============================================================
class TestBronzeRowCounts:
    """Verify Bronze tables have minimum expected row counts."""

    def test_transactions_row_count(self):
        cnt = table("brz_transactions").count()
        assert cnt >= 200_000, f"Expected >=200K transactions, got {cnt}"

    def test_returns_row_count(self):
        cnt = table("brz_returns").count()
        assert cnt >= 7_000, f"Expected >=7K returns, got {cnt}"

    def test_products_row_count(self):
        cnt = table("brz_products_mongo_dlt").count()
        assert cnt >= 1_500, f"Expected >=1500 products, got {cnt}"

    def test_customers_row_count(self):
        cnt = table("brz_customers_mongo_dlt").count()
        assert cnt >= 10_000, f"Expected >=10K customers, got {cnt}"

    def test_stores_row_count(self):
        cnt = table("brz_stores").count()
        assert cnt >= 20, f"Expected >=20 stores, got {cnt}"

    def test_regions_row_count(self):
        cnt = table("brz_regions").count()
        assert cnt >= 100, f"Expected >=100 regions, got {cnt}"

    def test_kafka_orders_row_count(self):
        cnt = table("brz_kafka_orders").count()
        assert cnt >= 1, f"Kafka orders has 0 rows"

    def test_kafka_inventory_row_count(self):
        cnt = table("brz_kafka_inventory").count()
        assert cnt >= 1, f"Kafka inventory has 0 rows"


# ============================================================
# 3. SCHEMA VALIDATION
# ============================================================
class TestBronzeSchema:
    """Verify critical columns exist in Bronze tables."""

    def test_transactions_schema(self):
        cols = table("brz_transactions").columns
        expected = [
            "transaction_date", "stock_date", "product_id",
            "customer_id", "store_id", "quantity",
            "_ingestion_timestamp", "_source_system"
        ]
        for c in expected:
            assert c in cols, f"Missing column: {c} in brz_transactions"

    def test_returns_schema(self):
        cols = table("brz_returns").columns
        for c in ["return_date", "product_id", "store_id", "quantity"]:
            assert c in cols, f"Missing column: {c} in brz_returns"

    def test_products_mongo_schema(self):
        cols = table("brz_products_mongo_dlt").columns
        for c in ["product_id", "product_name", "product_retail_price",
                   "product_cost", "product_brand"]:
            assert c in cols, f"Missing column: {c} in brz_products_mongo_dlt"

    def test_customers_mongo_schema(self):
        cols = table("brz_customers_mongo_dlt").columns
        for c in ["customer_id", "first_name", "last_name",
                   "customer_country", "gender"]:
            assert c in cols, f"Missing column: {c} in brz_customers_mongo_dlt"

    def test_stores_schema(self):
        cols = table("brz_stores").columns
        for c in ["store_id", "region_id", "store_name", "store_country"]:
            assert c in cols, f"Missing column: {c} in brz_stores"

    def test_kafka_orders_schema(self):
        cols = table("brz_kafka_orders").columns
        for c in ["order_id", "product_id", "customer_id", "store_id",
                   "quantity", "total_amount", "event_time"]:
            assert c in cols, f"Missing column: {c} in brz_kafka_orders"

    def test_kafka_inventory_schema(self):
        cols = table("brz_kafka_inventory").columns
        for c in ["current_stock", "quantity_change", "change_type",
                   "product_id", "store_id", "event_time"]:
            assert c in cols, f"Missing column: {c} in brz_kafka_inventory"


# ============================================================
# 4. METADATA COLUMNS
# ============================================================
class TestBronzeMetadata:
    """Verify _ingestion_timestamp and _source_system exist and are populated."""

    @pytest.mark.parametrize("table_name", [
        "brz_transactions", "brz_returns", "brz_stores",
        "brz_regions", "brz_calendar",
        "brz_products_mongo_dlt", "brz_customers_mongo_dlt",
        "brz_kafka_orders", "brz_kafka_inventory",
    ])
    def test_metadata_columns_exist(self, table_name):
        cols = table(table_name).columns
        assert "_ingestion_timestamp" in cols, \
            f"{table_name} missing _ingestion_timestamp"
        assert "_source_system" in cols, \
            f"{table_name} missing _source_system"

    @pytest.mark.parametrize("table_name", [
        "brz_transactions", "brz_returns",
        "brz_products_mongo_dlt", "brz_kafka_orders",
    ])
    def test_metadata_not_null(self, table_name):
        df = table(table_name)
        null_ts = df.filter("_ingestion_timestamp IS NULL").count()
        null_src = df.filter("_source_system IS NULL").count()
        assert null_ts == 0, f"{table_name}: {null_ts} rows with null timestamp"
        assert null_src == 0, f"{table_name}: {null_src} rows with null source"


# ============================================================
# 5. DATA QUALITY SPOT CHECKS
# ============================================================
class TestBronzeDataQuality:
    """Basic data quality: no full-table nulls, valid value ranges."""

    def test_transactions_no_null_product_id(self):
        nulls = table("brz_transactions").filter("product_id IS NULL").count()
        total = table("brz_transactions").count()
        pct = nulls / total * 100 if total > 0 else 0
        assert pct < 5, f"brz_transactions: {pct:.1f}% null product_ids"

    def test_products_prices_positive(self):
        bad = table("brz_products_mongo_dlt").filter(
            "product_retail_price <= 0 OR product_retail_price IS NULL"
        ).count()
        assert bad == 0, f"brz_products: {bad} rows with invalid price"

    def test_customers_valid_gender(self):
        bad = table("brz_customers_mongo_dlt").filter(
            "gender NOT IN ('M', 'F') AND gender IS NOT NULL"
        ).count()
        assert bad == 0, f"brz_customers: {bad} rows with invalid gender"

    def test_kafka_orders_positive_quantity(self):
        bad = table("brz_kafka_orders").filter(
            "quantity <= 0 OR quantity IS NULL"
        ).count()
        total = table("brz_kafka_orders").count()
        pct = bad / total * 100 if total > 0 else 0
        assert pct < 5, f"brz_kafka_orders: {pct:.1f}% bad quantities"

    def test_kafka_inventory_non_negative_stock(self):
        bad = table("brz_kafka_inventory").filter(
            "current_stock < 0"
        ).count()
        assert bad == 0, f"brz_kafka_inventory: {bad} rows with negative stock"


# ============================================================
# 6. NO DUPLICATE RAW TABLES
# ============================================================
class TestBronzeNoDuplicates:
    """Verify no full-row duplicates in key Bronze tables."""

    def test_stores_no_duplicates(self):
        df = table("brz_stores")
        assert df.count() == df.dropDuplicates(["store_id"]).count(), \
            "brz_stores has duplicate store_ids"

    def test_regions_no_duplicates(self):
        df = table("brz_regions")
        assert df.count() == df.dropDuplicates(["region_id"]).count(), \
            "brz_regions has duplicate region_ids"

    def test_products_no_duplicates(self):
        df = table("brz_products_mongo_dlt")
        assert df.count() == df.dropDuplicates(["product_id"]).count(), \
            "brz_products has duplicate product_ids"
