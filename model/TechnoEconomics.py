import json
from typing import List, Dict
from pydantic import BaseModel, Field

class FileWithKPIs(BaseModel):
    # KPIs
    total_throughput_gbps: float = Field(default=0.0, description="Total throughput in Gbps")
    daily_avg_throughput_gbps: float = Field(default=0.0, description="Average daily throughput in Gbps")
    
    # Power consumption
    total_power_consumption_kWh: float = Field(default=0.0, description="Total power consumption in kWh")
    daily_avg_power_consumption_kWh: float = Field(default=0.0, description="Average daily power consumption in kWh")
    yearly_power_estimate_kWh: float = Field(default=0.0, description="Yearly power estimate in kWh")
    
    # Availability metrics
    availability_percentage: float = Field(default=0.0, description="Percentage of served users")
    
    # Blocked traffic
    blocked_traffic_gbps: float = Field(default=0.0, description="Amount of blocked traffic in Gbps")
    
    # Time series data
    throughput_time_series_gbps: List[float] = Field(default_factory=list, description="Throughput values over time in Gbps")
    power_time_series_kWh: List[float] = Field(default_factory=list, description="Power consumption values over time in kWh")

    @classmethod
    def from_file(cls, filename: str) -> 'FileWithKPIs':
        """Load KPIs and time series data from a JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, filename: str) -> None:
        """Save all KPIs and time series data to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)