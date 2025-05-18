import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, bike_data_path, weather_data_path, poi_data_path, road_data_path, taxi_data_path):
        """
        Initialize data loader with file paths for all data sources.
        """
        self.bike_data_path = bike_data_path
        self.weather_data_path = weather_data_path
        self.poi_data_path = poi_data_path
        self.road_data_path = road_data_path
        self.taxi_data_path = taxi_data_path

    def load_bike_data(self):
        """
        Load bike-sharing data, extracting basic features such as timestamps.
        """
        bike_data = pd.read_csv(self.bike_data_path)
        bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'])
        bike_data['hour'] = bike_data['timestamp'].dt.hour
        bike_data['day_of_week'] = bike_data['timestamp'].dt.dayofweek
        bike_data['is_weekend'] = bike_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        bike_data.fillna(0, inplace=True)
        return bike_data

    def load_weather_data(self):
        """
        Load weather data with basic normalization.
        """
        weather_data = pd.read_csv(self.weather_data_path)
        weather_data.fillna(0, inplace=True)
        return weather_data

    def load_poi_data(self):
        """
        Load POI data for stations.
        """
        poi_data = pd.read_csv(self.poi_data_path)
        return poi_data

    def load_road_data(self):
        """
        Load road network data.
        """
        road_data = pd.read_csv(self.road_data_path)
        return road_data

    def load_taxi_data(self):
        """
        Load taxi trip data.
        """
        taxi_data = pd.read_csv(self.taxi_data_path)
        return taxi_data

    def load_all_data(self):
        """
        Load all datasets and return a dictionary for downstream feature extraction.
        """
        bike_data = self.load_bike_data()
        weather_data = self.load_weather_data()
        poi_data = self.load_poi_data()
        road_data = self.load_road_data()
        taxi_data = self.load_taxi_data()

        return {
            "bike_data": bike_data,
            "weather_data": weather_data,
            "poi_data": poi_data,
            "road_data": road_data,
            "taxi_data": taxi_data,
        }
