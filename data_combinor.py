import json
import csv
from math import radians, sin, cos, sqrt, atan2


# Load the plate boundary data from the JSON file
with open("C:/EPSCI10/Seismograph-Earthquake-Detection/tectonicplates-master/GeoJSON/PB2002_boundaries.json") as f:
    data = json.load(f)

# Extract the latitude and longitude coordinates for each boundary point
boundaries = []
for feature in data["features"]:
    for lineString in feature["geometry"]["coordinates"]:
        # print(lineString)
        lon, lat = lineString
        boundaries.append((lat, lon))

file = open("file1.txt", "w+")
content = str(boundaries)
print(boundaries)
file.write(content)
file.close()

file = open("file1.txt", "r")
content = file.read()

print("\nContent in file1.txt:\n", content)
file.close()

# # Load the earthquake data from the USGS CSV file
# with open("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/usgs_main2.csv") as f:
#     reader = csv.DictReader(f)
#     earthquake_data = list(reader)

# # Combine the earthquake data with the plate boundary data
# for earthquake in earthquake_data:
#     # Calculate the distance between the earthquake and the nearest plate boundary
#     min_distance = float("inf")
#     for boundary in boundaries:
#         lat1, lon1 = radians(float(earthquake["latitude"])), radians(float(earthquake["longitude"]))
#         lat2, lon2 = radians(boundary[0]), radians(boundary[1])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#         c = 2 * atan2(sqrt(a), sqrt(1 - a))
#         distance = 6371 * c  # Approximate radius of Earth in km
#         if distance < min_distance:
#             min_distance = distance

#     # Add the plate boundary distance as a new feature to the earthquake data
#     earthquake["plate_boundary_distance"] = min_distance

# # Save the combined earthquake and plate boundary data to a new CSV file
# fieldnames = list(earthquake_data[0].keys())
# with open("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/combined_data4.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     for earthquake in earthquake_data:
#         writer.writerow(earthquake)

# print("Combined earthquake and plate boundary data saved to 'combined_data4.csv'.")
