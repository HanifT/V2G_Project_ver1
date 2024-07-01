# %%
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
with open("merged_dict.json", "r") as json_file:
     merged_dict = json.load(json_file)

full_data_dataframes = []
for vehicle, times in merged_dict.items():
    for hour, stats in times.items():
        full_data_dataframes.append({"vehicle": vehicle, "hour": hour, **stats})

full_data_dataframes = pd.DataFrame(full_data_dataframes)
full_data_dataframes_charging = full_data_dataframes[~full_data_dataframes["charge_type"].isna()]
full_data_dataframes_charging = full_data_dataframes_charging[full_data_dataframes_charging["charge_type"] != "None"]
full_data_dataframes_charging = full_data_dataframes_charging.groupby(["vehicle", "location", "model", "end_time", "charge_type"]).agg({
    "hour": "first",
    "bat_cap": "first",
    "distance": "sum",
    "soc_init": "first",
    "soc_end": "first",
    "soc_need": "first",
}).reset_index(drop=False)

full_data_dataframes_charging.loc[full_data_dataframes_charging["soc_init"].isna(), "soc_init"] = full_data_dataframes_charging.loc[full_data_dataframes_charging["soc_init"].isna(), "soc_end"]
full_data_dataframes_charging["hour"] = full_data_dataframes_charging["hour"].astype(int)

shours_in_day = [hour % 24 for hour in full_data_dataframes_charging["hour"]]
ehours_in_day = [hour % 24 for hour in full_data_dataframes_charging["end_time"]]
full_data_dataframes_charging["start_time_charging"] = shours_in_day
full_data_dataframes_charging["end_time_charging"] = ehours_in_day


price_dataframe = []
for hour, price in combined_price_PGE_average.items():
        price_dataframe.append({"hour": hour, "price": price/1000})

price_dataframe = pd.DataFrame(price_dataframe)
price_dataframe["hour_day"] = price_dataframe["hour"].astype(int) % 24
price_dataframe_average = price_dataframe.groupby("hour_day")["price"].mean().reset_index(drop=False)


price_dataframe_TOU = []
for hour, price in tou_prices.items():
        price_dataframe_TOU.append({"hour": hour, "price": price/1000})

price_dataframe_TOU = pd.DataFrame(price_dataframe_TOU)
price_dataframe_TOU["hour_day"] = price_dataframe_TOU["hour"].astype(int) % 24
price_dataframe_TOU_average = price_dataframe_TOU.groupby("hour_day")["price"].mean().reset_index(drop=False)
# %%
df = full_data_dataframes_charging.copy()
df["power"] = ((df["soc_end"] - df["soc_init"])/100) * df["bat_cap"]
# Define the charging rates
charging_rates = {
    'LEVEL_1': 1.5,
    'LEVEL_2': 6.6,
    'DC_FAST_Tesla': 150,
    'DC_FAST_Chevy': 50
}

# Assuming df is your existing DataFrame
num_rows = len(df)
hour_labels = [f'{i}' for i in range(0, 24)]

# Create an empty DataFrame with the same number of rows and 24 columns, initialized with zeros
hourly_soc_df = pd.DataFrame(0.0, index=range(num_rows), columns=hour_labels)


def cahrging_profile(df1):
    # Loop through each row in the dataframe
    for index, row in df1.iterrows():
        start_hour = row["start_time_charging"]
        end_hour = row["end_time_charging"] + 1
        power = row['power']

        if 'DC_FAST' in row['charge_type']:
            rate = charging_rates['DC_FAST_Tesla']*0.95 if row['model'] == 'Tesla' else charging_rates['DC_FAST_Chevy']*0.95
        else:
            rate = charging_rates[row['charge_type']]*0.95

        hour = start_hour
        while power > 0:

            charging_rate = min(power, rate)  # Adjust last increment if power is less than the charging rate

            # Add the charging rate to the appropriate hour in the hourly_soc_df
            hourly_soc_df.loc[index, str(hour)] += charging_rate

            # Decrease the remaining power by the charging rate
            power -= charging_rate

            # Move to the next hour, wrapping around if necessary
            hour = (hour + 1) % 24  # Wrap around after 24 hours

    return hourly_soc_df


charging_profile_df = cahrging_profile(df)
charging_profile_df_sum = charging_profile_df.sum()
charging_profile_df_sum.sum()
df["power"].sum()

result_df = pd.DataFrame({
    "total_power": df["power"],
    "total_charging_profile": charging_profile_df.sum(axis=1)
})
result_df["dif"] = result_df["total_power"] - result_df["total_charging_profile"]
# %%
hourly_data_RTH_6_smart = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 6.6) & (hourly_data_RTH["Charging Type"] == "smart") & (hourly_data_RTH["GHG Cost"] == 0.05)]
hourly_data_RTH_12_smart = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 12) & (hourly_data_RTH["Charging Type"] == "smart") & (hourly_data_RTH["GHG Cost"] == 0.05)]
hourly_data_RTH_19_smart = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 19) & (hourly_data_RTH["Charging Type"] == "smart") & (hourly_data_RTH["GHG Cost"] == 0.05)]

hourly_data_RTHW_6_smart = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 6.6) & (hourly_data_RTHW["Charging Type"] == "smart") & (hourly_data_RTHW["GHG Cost"] == 0.05)]
hourly_data_RTHW_12_smart = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 12) & (hourly_data_RTHW["Charging Type"] == "smart") & (hourly_data_RTHW["GHG Cost"] == 0.05)]
hourly_data_RTHW_19_smart = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 19) & (hourly_data_RTHW["Charging Type"] == "smart") & (hourly_data_RTHW["GHG Cost"] == 0.05)]

hourly_data_TOUH_6_smart = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 6.6) & (hourly_data_TOUH["Charging Type"] == "smart") & (hourly_data_TOUH["GHG Cost"] == 0.05)]
hourly_data_TOUH_12_smart = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 12) & (hourly_data_TOUH["Charging Type"] == "smart") & (hourly_data_TOUH["GHG Cost"] == 0.05)]
hourly_data_TOUH_19_smart = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 19) & (hourly_data_TOUH["Charging Type"] == "smart") & (hourly_data_TOUH["GHG Cost"] == 0.05)]

hourly_data_TOUHW_6_smart = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 6.6) & (hourly_data_TOUHW["Charging Type"] == "smart") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]
hourly_data_TOUHW_12_smart = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 12) & (hourly_data_TOUHW["Charging Type"] == "smart") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]
hourly_data_TOUHW_19_smart = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 19) & (hourly_data_TOUHW["Charging Type"] == "smart") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]

hourly_data_RTH_6_v2g = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 6.6) & (hourly_data_RTH["Charging Type"] == "v2g") & (hourly_data_RTH["GHG Cost"] == 0.05)]
hourly_data_RTH_12_v2g = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 12) & (hourly_data_RTH["Charging Type"] == "v2g") & (hourly_data_RTH["GHG Cost"] == 0.05)]
hourly_data_RTH_19_v2g = hourly_data_RTH[(hourly_data_RTH["Charging Speed"] == 19) & (hourly_data_RTH["Charging Type"] == "v2g") & (hourly_data_RTH["GHG Cost"] == 0.05)]

hourly_data_RTHW_6_v2g = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 6.6) & (hourly_data_RTHW["Charging Type"] == "v2g") & (hourly_data_RTHW["GHG Cost"] == 0.05)]
hourly_data_RTHW_12_v2g = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 12) & (hourly_data_RTHW["Charging Type"] == "v2g") & (hourly_data_RTHW["GHG Cost"] == 0.05)]
hourly_data_RTHW_19_v2g = hourly_data_RTHW[(hourly_data_RTHW["Charging Speed"] == 19) & (hourly_data_RTHW["Charging Type"] == "v2g") & (hourly_data_RTHW["GHG Cost"] == 0.05)]

hourly_data_TOUH_6_v2g = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 6.6) & (hourly_data_TOUH["Charging Type"] == "v2g") & (hourly_data_TOUH["GHG Cost"] == 0.05)]
hourly_data_TOUH_12_v2g = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 12) & (hourly_data_TOUH["Charging Type"] == "v2g") & (hourly_data_TOUH["GHG Cost"] == 0.05)]
hourly_data_TOUH_19_v2g = hourly_data_TOUH[(hourly_data_TOUH["Charging Speed"] == 19) & (hourly_data_TOUH["Charging Type"] == "v2g") & (hourly_data_TOUH["GHG Cost"] == 0.05)]

hourly_data_TOUHW_6_v2g = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 6.6) & (hourly_data_TOUHW["Charging Type"] == "v2g") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]
hourly_data_TOUHW_12_v2g = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 12) & (hourly_data_TOUHW["Charging Type"] == "v2g") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]
hourly_data_TOUHW_19_v2g = hourly_data_TOUHW[(hourly_data_TOUHW["Charging Speed"] == 19) & (hourly_data_TOUHW["Charging Type"] == "v2g") & (hourly_data_TOUHW["GHG Cost"] == 0.05)]

hourly_data_NR_6 = hourly_data_charging_NR[(hourly_data_charging_NR["Charging Speed"] == 6.6) & (hourly_data_charging_NR["Charging Type"] == "smart") & (hourly_data_charging_NR["GHG Cost"] == 0.05)]


# List of all dataframes
dataframes_smart = [
    hourly_data_RTH_6_smart, hourly_data_RTH_12_smart, hourly_data_RTH_19_smart,
    hourly_data_RTHW_6_smart, hourly_data_RTHW_12_smart, hourly_data_RTHW_19_smart,
    hourly_data_TOUH_6_smart, hourly_data_TOUH_12_smart, hourly_data_TOUH_19_smart,
    hourly_data_TOUHW_6_smart, hourly_data_TOUHW_12_smart, hourly_data_TOUHW_19_smart,
    hourly_data_NR_6
]

# Names for the columns
dataframe_names_smart = [
    "RTH_6_smart", "RTH_12_smart", "RTH_19_smart",
    "RTHW_6_smart", "RTHW_12_smart", "RTHW_19_smart",
    "TOUH_6_smart", "TOUH_12_smart", "TOUH_19_smart",
    "TOUHW_6_smart", "TOUHW_12_smart", "TOUHW_19_smart",
    "NR_6"
]

dataframes_v2g = [
    hourly_data_RTH_6_v2g, hourly_data_RTH_12_v2g, hourly_data_RTH_19_v2g,
    hourly_data_RTHW_6_v2g, hourly_data_RTHW_12_v2g, hourly_data_RTHW_19_v2g,
    hourly_data_TOUH_6_v2g, hourly_data_TOUH_12_v2g, hourly_data_TOUH_19_v2g,
    hourly_data_TOUHW_6_v2g, hourly_data_TOUHW_12_v2g, hourly_data_TOUHW_19_v2g,
    hourly_data_NR_6
]

# Names for the columns
dataframe_names_v2g = [
    "RTH_6_v2g", "RTH_12_v2g", "RTH_19_v2g",
    "RTHW_6_v2g", "RTHW_12_v2g", "RTHW_19_v2g",
    "TOUH_6_v2g", "TOUH_12_v2g", "TOUH_19_v2g",
    "TOUHW_6_v2g", "TOUHW_12_v2g", "TOUHW_19_v2g",
    "NR_6"
]


# %%
def final_charging_profiles(dataframes, dataframe_names, price_dataframe, price_dataframe1,title):
    # Add Hour_day column to each dataframe
    for df in dataframes:
        df.loc[:, "Hour_day"] = df["Hour"] % 24

    # Function to group by Hour_day and sum X_CHR
    def group_and_sum(df):
        return df.groupby(['Hour_day'])['X_CHR'].sum().reset_index()

    # Apply the function to each dataframe
    grouped_dataframes = [group_and_sum(df) for df in dataframes]

    # Create a new DataFrame to hold the results
    result_df = pd.DataFrame()

    # Populate the result DataFrame with the grouped data
    for df_name, grouped_df in zip(dataframe_names, grouped_dataframes):
        result_df[df_name] = grouped_df['X_CHR']

    # Add the Hour_day column
    result_df['Hour_day'] = grouped_dataframes[0]['Hour_day']

    # Set Hour_day as the index
    result_df.set_index('Hour_day', inplace=True)

    # Plotting
    plt.figure(figsize=(14, 8))  # Adjust the figure size as needed

    fig, ax1 = plt.subplots(figsize=(14, 8))  # Adjust the figure size as needed

    # Plot charging profiles
    for column in result_df.columns:
        ax1.plot(result_df.index, result_df[column] / 365, label=column)

    ax1.set_xlabel('Hour of the Day', fontsize=18)
    ax1.set_ylabel('Average Charging / Discharging (kW) per Day', fontsize=18)
    ax1.set_title(title, fontsize=18)
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))  # Adjust legend location as needed
    ax1.grid(True)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    # Create a second y-axis for the electricity price
    ax2 = ax1.twinx()
    ax2.plot(price_dataframe['hour_day'], price_dataframe['price'], color='black', label='Electricity Price Real Time', linestyle='--')
    ax2.plot(price_dataframe1['hour_day'], price_dataframe1['price'], color='r', label='Electricity Price TOU', linestyle='-.')

    ax2.set_ylabel('Electricity Price ($/kWh)', fontsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=14, colors='black')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.9))
    ax2.set_ylim([0.2, 0.7])
    plt.tight_layout()

    # Save the plot if needed
    # plt.savefig(f'{title}.png', dpi=300)

    # Show the plot
    plt.show()

    return result_df
# %%


gr_df_smart = final_charging_profiles(dataframes_smart, dataframe_names_smart, price_dataframe_average, price_dataframe_TOU_average, 'Smart Charging Profiles Over 24 Hours')

gr_df_v2g = final_charging_profiles(dataframes_v2g, dataframe_names_v2g,price_dataframe_average, price_dataframe_TOU_average, 'V2G Charging Profiles Over 24 Hours')


# %%
