import pandas as pd
from olist.data import Olist


class Order:

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self):

        orders = self.data["orders"].copy()

        orders = orders[
            orders["order_status"] == "delivered"
        ].copy()

        date_columns = [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]

        for col in date_columns:
            orders[col] = pd.to_datetime(orders[col])

        orders["wait_time"] = (
            orders["order_delivered_customer_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

        orders["expected_wait_time"] = (
            orders["order_estimated_delivery_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

        orders["delay_vs_expected"] = (
            orders["wait_time"]
            - orders["expected_wait_time"]
        ).clip(lower=0)

        columns = [
            "order_id",
            "wait_time",
            "expected_wait_time",
            "delay_vs_expected",
            "order_status"
        ]

        return orders[columns].copy()