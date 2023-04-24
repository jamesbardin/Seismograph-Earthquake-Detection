import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/usgs_main2.csv")

bad_features = ["magType","nst","gap","dmin","rms","net","id","updated","type","horizontalError","depthError","magError","magNst","status","locationSource","magSource", "place"]
# Delete the feature (column) from the header and the DataFrame

for feature in bad_features:
    if feature in df.columns:
        del df[feature]

# Save the updated DataFrame to the CSV file
df.to_csv("C:/EPSCI10/Seismograph-Earthquake-Detection/csvs/usgs_main2.csv", index=False)
