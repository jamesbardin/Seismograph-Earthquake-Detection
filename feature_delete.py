import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/combined_data_good.csv")

# bad_features = ["magType","nst","gap","dmin","rms","net","id","updated","type","horizontalError","depthError","magError","magNst","status","locationSource","magSource", "place", "plate_boundary_distance"]
# # Delete the feature (column) from the header and the DataFrame

# for feature in bad_features:
#     if feature in df.columns:
#         del df[feature]

df = df[df['plate_boundary_distance'] <= 500]

# Save the updated DataFrame to the CSV file
df.to_csv("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/usgs_main_no_far.csv", index=False)



