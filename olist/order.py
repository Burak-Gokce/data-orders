
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

    def get_review_score(self):
        """
        Returns a DataFrame containing order review scores
        and binary indicators for five-star and one-star reviews.
        """
        reviews = self.data["order_reviews"].copy()

        reviews["dim_is_five_star"] = (
            reviews["review_score"] == 5
        ).astype(int)

        reviews["dim_is_one_star"] = (
            reviews["review_score"] == 1
        ).astype(int)

        columns = [
            "order_id",
            "dim_is_five_star",
            "dim_is_one_star",
            "review_score"
        ]

        return reviews[columns].copy()
    
    def get_number_items(self):
        """
        Returns a DataFrame with the number of items per order.
        """
        order_items = self.data["order_items"].copy()

        number_items = (
            order_items
            .groupby("order_id")
            .size()
            .reset_index(name="number_of_items")
        )

        return number_items

    def get_number_sellers(self):
        """
        Returns a DataFrame with the number of unique sellers per order.
        """
        order_items = self.data["order_items"].copy()

        number_sellers = (
            order_items
            .groupby("order_id")["seller_id"]
            .nunique()
            .reset_index(name="number_of_sellers")
        )

        return number_sellers
