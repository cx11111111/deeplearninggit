import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("C:\\Users\\22279\\Desktop\\数据集\\风机功率数据\\wtbdata_245days.csv")

# Load spatial location data
location_data = pd.read_csv("C:\\Users\\22279\\Desktop\\数据集\\风机功率数据\\kdd_loc.CSV")

# Choose a specific day for analysis (replace with your desired day)
selected_day_data = data[data['Day'] == 1]

# Plot wind direction histogram
plt.hist(selected_day_data['Wdir'], bins=360, edgecolor='black')
plt.xlabel('Wind Direction (degrees)')
plt.ylabel('Frequency')
plt.title('Wind Direction Distribution')
plt.show()



# Merge power and location data
merged_data = pd.merge(data, location_data, on='TurbID')

# Plot scatter plot
plt.scatter(merged_data['x'], merged_data['y'], c=merged_data['Patv'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Average Active Power (kW)')
plt.xlabel('Horizontal Coordinate (x)')
plt.ylabel('Vertical Coordinate (y)')
plt.title('Spatial Distribution and Power Generation')
plt.show()

