__author__ = "Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Enrique Fernandez Sanchez"]
__version__ = "1.2"
__maintainer__ = "Enrique Fernandez Sanchez"
__email__ = "efernandez@e-lighthouse.com"
__status__ = "Validated"

from enum import Enum

class Weather(Enum):
    SUNNY = 1.0   # No reduction in irradiance for sunny day
    CLOUDY = 0.5  # 50% reduction in irradiance for cloudy day [20-50% less]
    RAINY = 0.1   # 90% reduction in irradiance for rainy day [80-90% less]

class SolarPanel:
    def __init__(self, power_rating, voltage_charging, efficiency, area):
        self.power_rating = power_rating            # Panel's power rating in Watts (10W) 
        self.voltage_charging = voltage_charging    # Panel's Voltage charging to battery
        self.efficiency = efficiency                # Panel's efficiency as decimal (0.15 == 15%)
        self.area = area                            # Panel's area in square meters (0.10 == 0.1m2)

    # > Go to this website: https://globalsolaratlas.info/
    # > Select a place, and save the value of GHI *per Day*
    # -> Example: Cartagena, GHI per Day: 4.900 kWh/m2
    irradiance_city = {
        "Cartagena": 0.980,   # kWh/m2 -> mean of two peak hours, summer, sunny day
    }
    
    def calculate_power_generated(self, solar_irradiance, timeStep: int, weather_condition: Weather = Weather.SUNNY) -> float:
        """
        Calculate the power generated by the solar panel based on solar irradiance and weather condition.
        :param solar_irradiance: Solar irradiance in kWh/m² per day
        :param timeStep: simulation steps in seconds (s)
        :param weather_condition: Enum value from Weather (SUNNY, CLOUDY, RAINY)
        :return: Power generated in Watts (W)
        """
        solar_irradiance = solar_irradiance * 1000  # Convert kWh to Wh
        adjusted_irradiance = solar_irradiance * weather_condition.value
        power_generated = adjusted_irradiance * self.area * self.efficiency
        power_generated_timeStep = (power_generated / 3600) * timeStep
        return power_generated_timeStep  # In Watts (W)
    
    def calculate_current(self, solar_irradiance, timeStep: int, weather_condition : Weather = Weather.SUNNY) -> float:
        """
        Calculate the current generated by the solar panel based on solar irradiance and weather condition.
        :param solar_irradiance: Solar irradiance in hW/m² per day
        :param timeStep: simulation steps in seconds (s)
        :param weather_condition: Enum value from Weather (SUNNY, CLOUDY, RAINY)
        :return: Current in amperes (A)
        """
        solar_irradiance = solar_irradiance * 1000  # Convert kWh to Wh
        current = self.calculate_power_generated(solar_irradiance, timeStep=timeStep, weather_condition=weather_condition) / self.power_rating
        return current  # In Amperes (A)
    
    def calculate_Ah_in_timeStep(self, solar_irradiance, timeStep, weather_condition : Weather = Weather.SUNNY):
        """
        Calculate the Amperes Hours of a solara panel, based on the irradiance and the weather conditions
        :param solar_irradiance: Solar irradiance in khW/m² per day
        :param timeStep: simulation steps in seconds (s)
        :param weather_condition: Enum value from Weather (SUNNY, CLOUDY, RAINY)
        :return: Current in amperes hour (Ah)
        """
        charging_power = self.calculate_power_generated(solar_irradiance=solar_irradiance, timeStep=timeStep, weather_condition=weather_condition)
        return charging_power / self.voltage_charging   # In Amperes Hour (Ah)

if __name__ == "__main__":
    # Example of usage
    # 1. Create a solar panel with the specific caracteristics:
    # Reference Solar Panel:
    # SeedStudio Panel: https://www.seeedstudio.com/Solar-Panel-PV-12W-with-mounting-bracket-p-5003.html
    # https://www.mouser.es/new/seeed-studio/seeed-studio-pv-12w-solar-panel/
    solar_panel = SolarPanel(power_rating=12, voltage_charging=14, efficiency=0.2, area=(0.35 * 0.25))
    print(f'Using irradiance of Cartagena: {solar_panel.irradiance_city["Cartagena"]}')
    print(f"Using Solar Panel: { solar_panel.__dict__}")

    # 2. Calculate the current for the selected timeStep
    # > We can select the weather of the actual step
    current_sunny = solar_panel.calculate_current(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.SUNNY)
    current_cloudy = solar_panel.calculate_current(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.CLOUDY)
    current_rainy = solar_panel.calculate_current(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.RAINY)

    print(f"Current for the time step under Sunny conditions: {current_sunny} A")
    print(f"Current for the time step under Cloudy conditions: {current_cloudy} A")
    print(f"Current for the time step under Rainy conditions: {current_rainy} A")
    
    print("-----")
    # 3. Calculate the power for the selected timeStep
    # > We can select the weather of the actual step
    power_sunny = solar_panel.calculate_power_generated(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.SUNNY)
    power_cloudy = solar_panel.calculate_power_generated(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.CLOUDY)
    power_rainy = solar_panel.calculate_power_generated(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.RAINY)

    print(f"Power for the time step under Sunny conditions: {power_sunny} W")
    print(f"Power for the time step under Cloudy conditions: {power_cloudy} W")
    print(f"Power for the time step under Rainy conditions: {power_rainy} W")
    
    print("-----")
    # 4. Get the Ah to charge battery
    amperes_hour_sunny = solar_panel.calculate_Ah_in_timeStep(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.SUNNY)
    amperes_hour_cloudy = solar_panel.calculate_Ah_in_timeStep(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.CLOUDY)
    amperes_hour_rainy = solar_panel.calculate_Ah_in_timeStep(solar_irradiance=solar_panel.irradiance_city["Cartagena"], timeStep=0.5, weather_condition=Weather.RAINY)
    print(f"Amperes hour used to charge the battery in one timeStep, sunny: {amperes_hour_sunny} Ah")
    print(f"Amperes hour used to charge the battery in one timeStep, cloudy: {amperes_hour_cloudy} Ah")
    print(f"Amperes hour used to charge the battery in one timeStep, rainy: {amperes_hour_rainy} Ah")
