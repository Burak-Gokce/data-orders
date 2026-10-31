
import pandas as pd
import numpy as np
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

    def get_price_and_freight(self):
        """
        Returns a DataFrame with the total price
        and freight value for each order.
        """
        order_items = self.data["order_items"].copy()

        price_and_freight = (
            order_items
            .groupby("order_id")[["price", "freight_value"]]
            .sum()
            .reset_index()
        )

        return price_and_freight

    def get_training_data(self, with_distance_seller_customer=False):
        """
        Returns a DataFrame combining all order features.
        """
        wait_time = self.get_wait_time()
        review_score = self.get_review_score()
        number_items = self.get_number_items()
        number_sellers = self.get_number_sellers()
        price_and_freight = self.get_price_and_freight()

        training_data = (
            wait_time
            .merge(review_score, on="order_id", how="inner")
            .merge(number_items, on="order_id", how="inner")
            .merge(number_sellers, on="order_id", how="inner")
            .merge(price_and_freight, on="order_id", how="inner")
        )

        if with_distance_seller_customer:
            distances = self.get_distance_seller_customer()

            training_data = training_data.merge(
                distances,
                on="order_id",
                how="left",
                validate="many_to_one"
            )

        training_data = training_data.dropna(
            subset=[
                "wait_time",
                "expected_wait_time",
                "delay_vs_expected",
                "review_score",
                "number_of_items",
                "number_of_sellers",
                "price",
                "freight_value"
            ]
        )

        return training_data

    def get_distance_seller_customer(self):
        """
        Calculate the average seller-customer distance
        in kilometers for each order.
        """
        geo = self.data["geolocation"].copy()

        geo_unique = (
            geo.groupby("geolocation_zip_code_prefix")[
                ["geolocation_lat", "geolocation_lng"]
            ]
            .mean()
            .reset_index()
        )

        customers_geo = self.data["customers"].merge(
            geo_unique,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left"
        ).rename(columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng"
        })

        sellers_geo = self.data["sellers"].merge(
            geo_unique,
            left_on="seller_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left"
        ).rename(columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng"
        })

        order_locations = (
            self.data["orders"][["order_id", "customer_id"]]
            .merge(
                self.data["order_items"][["order_id", "seller_id"]],
                on="order_id",
                how="inner"
            )
            .merge(
                customers_geo[
                    ["customer_id", "customer_lat", "customer_lng"]
                ],
                on="customer_id",
                how="left"
            )
            .merge(
                sellers_geo[
                    ["seller_id", "seller_lat", "seller_lng"]
                ],
                on="seller_id",
                how="left"
            )
        )

        lat1 = np.radians(order_locations["customer_lat"])
        lon1 = np.radians(order_locations["customer_lng"])
        lat2 = np.radians(order_locations["seller_lat"])
        lon2 = np.radians(order_locations["seller_lng"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        order_locations["distance_seller_customer"] = 6371 * c

        return (
            order_locations
            .groupby("order_id", as_index=False)
            .agg(
                distance_seller_customer=(
                    "distance_seller_customer", "mean"
                )
            )
        )
