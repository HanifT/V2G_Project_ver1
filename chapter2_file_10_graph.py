# %%
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %% Reading Files

with open("combined_price_PGE_average.json", "r") as json_file:
    combined_price_PGE_average = json.load(json_file)

with open("tou_prices.json", "r") as json_file:
    tou_prices = json.load(json_file)

with open("merged_dict.json", "r") as json_file:
    merged_dict = json.load(json_file)

# Flatten the dictionary
flattened_data = []
for vehicle, hours in merged_dict.items():
    for hour, values in hours.items():
        entry = {'Vehicle': vehicle, 'Hour': int(hour)}
        entry.update(values)
        flattened_data.append(entry)

# Create DataFrame
flatten_veh_data = pd.DataFrame(flattened_data)

# %%


def xlsx_read(dic):

    # List of Excel file names
    excel_files = [f for f in os.listdir(dic) if f.endswith('.xlsx')]

    # Dictionary to store dataframes
    all_dataframes = {}

    # Iterate over each Excel file
    for excel_file_name in excel_files:
        excel_file_path = os.path.join(dic, excel_file_name)
        print(f"Reading Excel file '{excel_file_path}'...")

        # Read each sheet into a separate dataframe
        with pd.ExcelFile(excel_file_path) as xls:
            sheet_names = xls.sheet_names  # Get the names of all sheets in the Excel file

            # Read each sheet into a dataframe and store it in the dictionary
            for sheet_name in sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                new_df_name = f"{excel_file_name[:-5]}_{sheet_name}"  # Add sheet name to the file name
                all_dataframes[new_df_name] = df

    # Create a new dataframe to store total cost data
    total_costs_df = pd.DataFrame()

    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes.items():
        if "Total Costs" in df_name:
            # Extract charging type and speed from the dataframe name
            charging_type = "smart" if "smart" in df_name else "v2g"
            charging_speed = df_name.split("_")[2][:-1]
            ghg_cost = df_name.split("_")[3][:-2]

            # Add a column indicating the charging type (smart or v2g)
            df['Charging Type'] = charging_type
            # Add columns indicating charging speed and GHG cost
            df['Charging Speed'] = charging_speed
            df['GHG Cost'] = ghg_cost

            # Concatenate this dataframe with the total_costs_df
            total_costs_df = pd.concat([total_costs_df, df])

    print("Total cost data has been extracted.")
    total_costs_df = total_costs_df.reset_index(drop=True)
    # Display the new dataframe
    print(total_costs_df)

    # Create a new dataframe to store total cost data
    individual_cost_df = pd.DataFrame()

    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes.items():
        if "Individual Cost" in df_name:
            # Extract charging type and speed from the dataframe name
            charging_type = "smart" if "smart" in df_name else "v2g"
            charging_speed = df_name.split("_")[2][:-1]
            ghg_cost = df_name.split("_")[3][:-2]

            # Add a column indicating the charging type (smart or v2g)
            df['Charging Type'] = charging_type
            # Add columns indicating charging speed and GHG cost
            df['Charging Speed'] = charging_speed
            df['GHG Cost'] = ghg_cost

            # Concatenate this dataframe with the total_costs_df
            individual_cost_df = pd.concat([individual_cost_df, df])

    individual_cost_df = individual_cost_df.reset_index(drop=True)
    # Display the new dataframe
    return total_costs_df


# %%

def plotting(df, num_vehicles):
    df1 = df.copy()
    # Create a new column called `stacked_index` by concatenating the columns `Charging Type`, `Charging Speed`, and `GHG Cost` with hyphens as separators.
    df1['stacked_index'] = df1['Charging Type'] + ' - ' + df1['Charging Speed'] + ' - ' + df1['GHG Cost'].astype(str) + df1.apply(lambda row: '' if row['Charging Type'] == 'smart' else ' - ' + row['V2G Location'], axis=1)

    # Define custom order for charging speed and ghg cost
    charging_speed_order = ['6.6', '12', '19']
    ghg_cost_order = ['0.05', '0.191']

    # Create categorical columns with the specified orders
    df1['Charging Speed'] = pd.Categorical(df1['Charging Speed'], categories=charging_speed_order, ordered=True)
    df1['GHG Cost'] = pd.Categorical(df1['GHG Cost'], categories=ghg_cost_order, ordered=True)

    # Sort by charging type first, then by the ordered categorical columns
    df1 = df1.sort_values(by=['V2G Location', 'Charging Type', 'Charging Speed', 'GHG Cost'])
    df1[["Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR"]] = df1[["Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR"]].div(num_vehicles, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    components = ['Electricity_Cost', 'Degradation_Cost', 'GHG_Cost']
    colors = ["#008EAA", "#C10230", "#8A532F"]
    bottom_pos = [0] * len(df1)
    bottom_neg = [0] * len(df1)

    # Keep track of plotted components to avoid duplicate legends
    plotted_components = set()

    # Calculate total cost per stack
    total_costs = df1[components].sum(axis=1)

    for i, component in enumerate(components):
        if component not in plotted_components:
            ax1.bar(df1['stacked_index'], df1[component].clip(lower=0), color=colors[i], bottom=bottom_pos, label=component)
            ax1.bar(df1['stacked_index'], df1[component].clip(upper=0), color=colors[i], bottom=bottom_neg)
            plotted_components.add(component)

        # Update bottom values for stacking (using stacked_index)
        bottom_pos = [bottom_pos[j] + df1[component].clip(lower=0).iloc[j] for j in range(len(df1))]
        bottom_neg = [bottom_neg[j] + df1[component].clip(upper=0).iloc[j] for j in range(len(df1))]

    # Plot total cost line (on ax1, the primary axis)
    ax1.plot(df1['stacked_index'], total_costs, color='black', marker='o', linestyle='-', label='Total Cost')

    # Rotate x-axis labels (optional)
    plt.xticks(rotation=45, ha='right')

    # Set other labels, title, legends, grid, and layout (with adjustments for new legend)
    ax1.set_ylabel('Cost/Revenue ($)')
    ax1.set_xlabel('Charging Scenarios')
    ax1.set_ylim(-3500, 3500)
    ax2 = ax1.twinx()
    ax2.plot(df1['stacked_index'], df1['X_CHR'], color='blue', label='Charging Demand')
    ax2.set_ylim(0, 5000)  # Adjust the secondary y-axis limits if needed
    ax2.set_ylabel('Charging Demand (kWh)', color='blue')

    # Set the tick parameters for the second y-axis to be blue
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['right'].set_color('blue')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    ax1.set_title(f'Average Cost Breakdown per Vehicle (for {num_vehicles} BEVs) Under Different Scenarios')

    ax1.grid(True)
    # Remove the grid from the secondary axis (ax2)
    ax2.grid(False)
    # Add V2G location annotations

    plt.subplots_adjust(bottom=0.35)
    # plt.savefig('plot_output.png')
    plt.show()

# %%


def draw_RT(df):

    # Create a sample array representing electricity price for 8760 hours
    electricity_price = [] # Random values between 0 and 1
    for key, value in df.items():
        electricity_price.append(value)
    electricity_price = electricity_price[:8760]
    # Create time labels for the x-axis (assuming hourly data)
    hours = np.arange(0, 8760, 1)  # Array of hours from 0 to 8759

    # Plot the electricity price line
    plt.figure(figsize=(12, 6))
    plt.plot(hours, electricity_price, label='Electricity Price ($ / MWh)')

    # Set labels and title
    plt.xlabel('Hour')
    plt.ylabel('Electricity Price ($ / MWh)')
    plt.title('Electricity Price for year 2021 - PG&E territory ')

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# %%


def json_file(dic, flatten_veh):

    # List of JSON file names
    json_files = [f for f in os.listdir(dic) if f.endswith('.json')]
    # Dictionary to store dataframes
    all_dataframes1 = {}

    # Iterate over each JSON file
    for json_file_name in json_files:
        json_file_path = os.path.join(dic, json_file_name)
        print(f"Reading JSON file '{json_file_path}'...")

        # Read each JSON file into a dataframe
        df = pd.read_json(json_file_path)
        # Add the dataframe to the dictionary with the file name as key
        all_dataframes1[json_file_name[:-5]] = df

    # Create a new dataframe to store total cost data
    hourly_data = pd.DataFrame()
    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes1.items():
        # Extract charging type and speed from the dataframe name
        charging_type = "smart" if "smart" in df_name else "v2g"
        charging_speed = df_name.split("_")[2][:-1]
        ghg_cost = df_name.split("_")[3][:-2]

        # Add a column indicating the charging type (smart or v2g)
        df['Charging Type'] = charging_type
        # Add columns indicating charging speed and GHG cost
        df['Charging Speed'] = charging_speed
        df['GHG Cost'] = ghg_cost

        # Concatenate this dataframe with the hourly_data dataframe
        hourly_data = pd.concat([hourly_data, df], ignore_index=True)

    hourly_data = hourly_data.reset_index(drop=True)
    hourly_data = pd.merge(hourly_data, flatten_veh[["Vehicle", "Hour", "charging_indicator", "location"]], how="left", on=["Vehicle", "Hour"])

    hourly_data["Charging Speed"] = hourly_data["Charging Speed"].astype(float)
    hourly_data["charging_indicator"] = hourly_data["charging_indicator"].fillna(0).astype(int)
    hourly_data["GHG Cost"] = hourly_data["GHG Cost"].astype(float)
    hourly_data_discharging = hourly_data[(hourly_data["X_CHR"] <= 0) & (hourly_data["charging_indicator"] == 1)]
    hourly_data_charging = hourly_data[(hourly_data["X_CHR"] > 0)]
    hourly_data_charging_smart = hourly_data_charging[(hourly_data_charging["Charging Type"] == "smart") & (hourly_data_charging["Charging Speed"] == 6.6)]
    hourly_data_charging_v2g = hourly_data_charging[(hourly_data_charging["Charging Type"] == "v2g") & (hourly_data_charging["Charging Speed"] == 6.6)]
    # Group the DataFrame by 'hour'
    hourly_data_discharging = hourly_data_discharging[hourly_data_discharging["Charging Type"] == "v2g"]
    grouped_df = hourly_data_discharging.groupby(['Hour', "Charging Speed", "GHG Cost", "Batt_cap", "location"])
    # Calculate sum and size (count) for each group
    result = grouped_df.agg({'X_CHR': ['sum', 'count']})
    result = result.reset_index(drop=False)

    # Create two new columns from the MultiIndex
    result['X_CHR_Sum'] = result[('X_CHR', 'sum')]
    result['X_CHR_Count'] = result[('X_CHR', 'count')]

    # Drop the original MultiIndex column
    result = result.drop(columns=[('X_CHR', 'sum'), ('X_CHR', 'count')])

    # Calculate Total_power
    result["Total_power"] = result["Charging Speed"] * result["X_CHR_Count"]
    result["Utilization Rate"] = abs(result["X_CHR_Sum"] / result["Total_power"])*100

    # Convert the hour values to modulo 24 to represent a 24-hour clock
    result['Hour_of_day'] = result['Hour'] % 24

    # Identify the peak hours between 4 PM and 9 PM (16:00 to 21:00)
    result['Peak'] = (result['Hour_of_day'] >= 16) & (result['Hour_of_day'] <= 21)

    # Convert boolean values to 'Peak' and 'Non-Peak' strings
    result['Peak'] = result['Peak'].map({True: 'Peak', False: 'Non-Peak'})

    return result, hourly_data, hourly_data_charging, hourly_data_discharging


# %%

def draw_util(df):
    # Assuming you have a DataFrame named 'df' with columns 'Charging_Speed', 'GHG_Cost', and 'Peak'
    # Filter data for peak and non-peak hours
    peak_data = df[df['Peak'] == 'Peak']
    non_peak_data = df[df['Peak'] == 'Non-Peak']

    # Create the box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Box plot for peak hours
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=peak_data, ax=axes[0], whis=[10, 90])
    axes[0].set_title('Peak Hours')
    axes[0].set_xlabel('Discharging Speed')
    axes[0].set_ylabel('%')

    # Box plot for non-peak hours
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=non_peak_data, ax=axes[1], whis=[10, 90])
    axes[1].set_title('Non-Peak Hours')
    axes[1].set_xlabel('Discharging Speed')
    axes[1].set_ylabel('%')
    # Add annotation explaining whiskers
    annotation_text = "Whiskers extend to the 5th and 95th percentiles of the data"
    plt.annotate(annotation_text, xy=(0.5, -0.15), xytext=(0, -50), ha='center', fontsize=12,
                 xycoords='axes fraction', textcoords='offset points', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# %%


def draw_util_rt(df):
    # Assuming you have a DataFrame named 'df' with columns 'Charging_Speed', 'GHG_Cost', 'Peak', and 'Utilization Rate'

    # Create a new column to distinguish between peak and non-peak hours
    df['Hour Type'] = df['Peak'].map({'Peak': 'Peak Hours', 'Non-Peak': 'Non-Peak Hours'})

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=df, whis=[5, 95])
    plt.title('Utilization Rate by Charging Speed')
    plt.xlabel('Charging Speed')
    plt.ylabel('Utilization Rate (%)')

    # Add annotation explaining whiskers
    annotation_text = "Whiskers extend to the 5th and 95th percentiles of the data"
    plt.annotate(annotation_text, xy=(0.5, -0.15), xytext=(0, -50), ha='center', fontsize=12,
                 xycoords='axes fraction', textcoords='offset points', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# %%


def draw_profile(charging_cost, hourly_data):

    combined_price_PGE_average_df = pd.DataFrame([charging_cost])
    combined_price_PGE_average_df = combined_price_PGE_average_df.T
    combined_price_PGE_average_df = combined_price_PGE_average_df.copy()
    combined_price_PGE_average_df = combined_price_PGE_average_df.rename(columns={0: 'Price'}).reset_index(drop=False)
    combined_price_PGE_average_df = combined_price_PGE_average_df.rename(columns={'index': 'Hour'})
    combined_price_PGE_average_df["Hour"] = combined_price_PGE_average_df["Hour"].astype(int)
    hourly_data_1087 = hourly_data[(hourly_data["Vehicle"] == "P_1087") & (hourly_data["Charging Type"] == "v2g") & (hourly_data["Charging Speed"] == 6.6)]
    # Data Preparation

    optimal_result = pd.merge(hourly_data_1087, combined_price_PGE_average_df, on='Hour', how='left').copy()
    optimal_result['Hour'] = pd.to_numeric(optimal_result['Hour']) + 1

    # Specify the year
    year = 2021

    # Convert the hour index to timedelta
    optimal_result['Hour_timedelta'] = pd.to_timedelta(optimal_result['Hour'], unit='h')

    # Add the timedelta to the start of the year to get datetime values
    optimal_result['Datetime'] = pd.Timestamp(year=year, month=1, day=1) + optimal_result['Hour_timedelta']
    optimal_result = optimal_result[8025:8700].reset_index(drop=True)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Plot each vehicle as a separate line on the primary y-axis
    for vehicle in optimal_result['Vehicle'].unique():
        vehicle_data = optimal_result[optimal_result['Vehicle'] == vehicle]
        ax1.plot(vehicle_data['Datetime'], vehicle_data['X_CHR'], label=vehicle, marker='o')

    ax1.set_xlabel('Datetime', fontsize=20)
    ax1.set_ylabel('Charging (kW)', fontsize=20)
    ax1.set_title('Vehicle Charging vs Electricity Price', fontsize=20)
    ax1.legend(loc="upper left", fontsize=20)
    ax1.set_ylim([0, 150])
    # Increase the size of the tick labels
    ax1.tick_params(axis='both', which='major', labelsize=12)
    # Create a secondary y-axis for electricity price
    ax2 = ax1.twinx()

    # Plot the electricity price on the secondary y-axis
    ax2.plot(optimal_result['Datetime'], optimal_result['Price']/1000, label='Electricity Price', color='red', linestyle='dashed', marker='x')
    ax2.set_ylabel('Price ($/kWh)', fontsize=20)
    # Increase the size of the tick labels on the secondary y-axis
    ax2.tick_params(axis='both', which='major', labelsize=18)
    # Combine the legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + [lines2[0]], labels + [labels2[0]], loc='upper center', fontsize=18)
    ax2.set_ylim([0.2, 0.6])
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.show()

# %%


def demand_response(df, exclude_locations):
    # Ensure exclude_locations is a list
    if not isinstance(exclude_locations, list):
        exclude_locations = [exclude_locations]

    # Filter the DataFrame to exclude specified locations
    df = df[df["location"].isin(exclude_locations)].copy()
    df["%1increase"] = df["Total_power"] * 1.01 - df["Total_power"]
    df["%10increase"] = df["Total_power"] * 1.10 - df["Total_power"]
    df["%20increase"] = df["Total_power"] * 1.20 - df["Total_power"]

    df["%1response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%1increase"]
    df["%10response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%10increase"]
    df["%20response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%20increase"]

    # Group by 'hour' and calculate percentages
    grouped1 = df.groupby(['Hour_of_day', "Batt_cap"])['%1response'].value_counts(normalize=True).unstack().fillna(0) * 100
    grouped10 = df.groupby(['Hour_of_day', "Batt_cap"])['%10response'].value_counts(normalize=True).unstack().fillna(0) * 100
    grouped20 = df.groupby(['Hour_of_day', "Batt_cap"])['%20response'].value_counts(normalize=True).unstack().fillna(0) * 100

    # Rename columns for clarity
    grouped1.columns = ['Percentage_False', 'Percentage_True']
    grouped10.columns = ['Percentage_False', 'Percentage_True']
    grouped20.columns = ['Percentage_False', 'Percentage_True']

    # Reset index to make 'hour' a column again
    grouped1 = grouped1.reset_index()
    grouped10 = grouped10.reset_index()
    grouped20 = grouped20.reset_index()

    return grouped1, grouped10, grouped20

# %%
charging_cost_NR = tou_prices
directory_norm = "/Users/haniftayarani/V2G_Project/Result_normal/V2G_locations/Home_Work"

total_costs_df_norm = xlsx_read(directory_norm)
total_costs_df_norm["V2G Location"] = " "
total_costs_df_norm["Charging Type"] = "Normal"
plotting(total_costs_df_norm, 50)
draw_RT(combined_price_PGE_average)
# draw_RT(combined_price_PGE_gen)
result_NR, hourly_data_NR, hourly_data_charging_NR, hourly_data_discharging_NR = json_file(directory_norm, flatten_veh_data)
draw_util_rt(result_NR)
result_RTH_DR1, result_RTH_DR10, result_RTH_DR20 = demand_response(result_NR, ["Home"])
draw_profile(charging_cost_NR, hourly_data_NR)

# %%
charging_cost_RT = combined_price_PGE_average
directory_RTH = "/Users/haniftayarani/V2G_Project/Result_RT/V2G_locations/Home"

total_costs_df_RTH = xlsx_read(directory_RTH)
total_costs_df_RTH["V2G Location"] = "Home"
plotting(total_costs_df_RTH, 50)
draw_RT(combined_price_PGE_average)
# draw_RT(combined_price_PGE_gen)
result_RTH, hourly_data_RTH, hourly_data_charging_RTH, hourly_data_discharging_RTH = json_file(directory_RTH, flatten_veh_data)
draw_util_rt(result_RTH)
result_RTH_DR1, result_RTH_DR10, result_RTH_DR20 = demand_response(result_RTH, ["Home"])
draw_profile(charging_cost_RT, hourly_data_RTH)

directory_RTHW = "/Users/haniftayarani/V2G_Project/Result_RT/V2G_locations/Home_Work"

total_costs_df_RTHW = xlsx_read(directory_RTHW)
total_costs_df_RTHW["V2G Location"] = "Home & Work"
plotting(total_costs_df_RTHW, 50)
draw_RT(combined_price_PGE_average)
result_RTHW, hourly_data_RTHW, hourly_data_charging_RTHW, hourly_data_discharging_RTHW = json_file(directory_RTHW, flatten_veh_data)
draw_util_rt(result_RTHW)
result_RTHW_DR1, result_RTHW_DR10, result_RTHW_DR20 = demand_response(result_RTHW, ["Home", "Work"])
draw_profile(charging_cost_RT, hourly_data_RTHW)

total_costs_df_RTH_HW = pd.concat([total_costs_df_RTH, total_costs_df_RTHW[total_costs_df_RTHW["Charging Type"] != "smart"]], axis=0).reset_index(drop=True)
total_costs_df_RTH_HW.loc[total_costs_df_RTH_HW["V2G Location"] == "Home", "group"] = 1
total_costs_df_RTH_HW.loc[total_costs_df_RTH_HW["V2G Location"] != "Home", "group"] = 2
plotting(pd.concat([total_costs_df_RTH_HW, total_costs_df_norm]), 50)

# %%
charging_cost_TOU = tou_prices
directory_TOUH = "/Users/haniftayarani/V2G_Project/Result_TOU/V2G_locations/Home1"

total_costs_df_TOUH = xlsx_read(directory_TOUH)
total_costs_df_TOUH["V2G Location"] = "Home"
plotting(total_costs_df_TOUH, 50)
draw_RT(combined_price_PGE_average)
result_TOUH, hourly_data_TOUH, hourly_data_charging_TOUH, hourly_data_discharging_TOUH = json_file(directory_TOUH, flatten_veh_data)
draw_util(result_TOUH)
result_TOUH_DR1, result_TOUH_DR10, result_TOUH_DR20 = demand_response(result_TOUH, ["Home"])
draw_profile(charging_cost_TOU, hourly_data_TOUH)

directory_TOUHW = "/Users/haniftayarani/V2G_Project/Result_TOU/V2G_locations/Home_Work"

total_costs_df_TOUHW = xlsx_read(directory_TOUHW)
total_costs_df_TOUHW["V2G Location"] = "Home & Work"
plotting(total_costs_df_TOUHW, 50)
draw_RT(combined_price_PGE_average)
result_TOUHW, hourly_data_TOUHW, hourly_data_charging_TOUHW, hourly_data_discharging_TOUHW = json_file(directory_TOUHW, flatten_veh_data)
draw_util(result_TOUHW)
result_TOUHW_DR1, result_TOUHW_DR10, result_TOUHW_DR20 = demand_response(result_TOUHW, ["Home", "Work"])
draw_profile(charging_cost_TOU, hourly_data_TOUHW)


total_costs_df_TOUH_HW = pd.concat([total_costs_df_TOUH, total_costs_df_TOUHW[total_costs_df_TOUHW["Charging Type"] != "smart"]], axis=0).reset_index(drop=True)
total_costs_df_TOUH_HW.loc[total_costs_df_TOUH_HW["V2G Location"] == "Home", "group"] = 1
total_costs_df_TOUH_HW.loc[total_costs_df_TOUH_HW["V2G Location"] != "Home", "group"] = 2
total_costs_df_TOUH_HW = total_costs_df_TOUH_HW.sort_values(by=['V2G Location', 'Charging Type', 'Charging Speed', 'GHG Cost'])

plotting(pd.concat([total_costs_df_TOUH_HW, total_costs_df_norm]), 50)

# %%
import matplotlib.pyplot as plt

# Calculate the total energy injected back to the grid for each pricing method
hourly_data_discharging_RT_df = hourly_data_discharging_RTH[hourly_data_discharging_RTH["Charging Speed"] == 12]
hourly_data_discharging_TOU_df = hourly_data_discharging_TOUH[hourly_data_discharging_TOUH["Charging Speed"] == 12]


total_energy_RT = hourly_data_discharging_RT_df["X_CHR"].sum()
total_energy_TOU = hourly_data_discharging_TOU_df["X_CHR"].sum()

# Data for plotting
categories = ['Real-Time Pricing', 'TOU Pricing']
totals = [abs(total_energy_RT), abs(total_energy_TOU)]

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, totals, color=['blue', 'green'])

# Add labels and title
ax.set_xlabel('Pricing Method', fontsize=14)
ax.set_ylabel('Total Energy Injected (kWh)', fontsize=14)
ax.set_title('Total Energy Injected Back to the Grid via V2G', fontsize=16)

# Add values on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)

# Increase the size of tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim([0, 2e06])
plt.tight_layout()
plt.show()


# %%

def dr_plot(data1, data2):
    df1 = data1.copy()
    df2 = data2.copy()

    battery_capacities = df1['Batt_cap'].unique()
    n_battery_capacities = len(battery_capacities)

    fig, axes = plt.subplots(2, n_battery_capacities, figsize=(15, 10), sharey=True)
    plt.subplots_adjust(hspace=0.4)  # Adjust space between rows

    bar_width = 1  # Set bar width to reduce distance between bars

    # Define the colors
    blue_color = sns.color_palette("Blues", 10)[5]  # Ensure we get the 6th shade
    red_color = sns.color_palette("Reds", 10)[6]    # Ensure we get the 7th shade

    for i, batt_cap in enumerate(battery_capacities):
        # Filter data for the specific battery capacity
        data1_filtered = df1[df1['Batt_cap'] == batt_cap]
        data2_filtered = df2[df2['Batt_cap'] == batt_cap]

        # Plot for first dataframe
        axes[0, i].bar(data1_filtered['Hour_of_day'], data1_filtered['Percentage_False'] / 100, color=red_color, width=bar_width, label='False', hatch='\\')
        axes[0, i].bar(data1_filtered['Hour_of_day'], data1_filtered['Percentage_True'] / 100, bottom=data1_filtered['Percentage_False'] / 100, color=blue_color, width=bar_width, label='True', hatch='//')
        axes[0, i].set_title(f'Battery Capacity {batt_cap} kWh')
        axes[0, i].set_xticks(range(0, 24, 2))  # Show every other hour for readability
        axes[0, i].set_xlabel('Hour of Day')
        axes[0, i].set_ylabel('Response Probability to Demand Response Signal \n Home only')
        if i == 0:
            axes[0, i].legend()

        # Plot for second dataframe
        axes[1, i].bar(data2_filtered['Hour_of_day'], data2_filtered['Percentage_False'] / 100, color=red_color, width=bar_width, label='False', hatch='\\')
        axes[1, i].bar(data2_filtered['Hour_of_day'], data2_filtered['Percentage_True'] / 100, bottom=data2_filtered['Percentage_False'] / 100, color=blue_color, width=bar_width, label='True', hatch='//')
        axes[1, i].set_title(f'Battery Capacity {batt_cap} kWh')
        axes[1, i].set_xticks(range(0, 24, 2))  # Show every other hour for readability
        axes[1, i].set_xlabel('Hour of Day')
        axes[1, i].set_ylabel('Response Probability to Demand Response Signal \n Home and Work')
        if i == 0:
            axes[1, i].legend()

    sns.despine()
    plt.tight_layout()
    plt.show()


# %%
# Plot the grid of stacked bar charts
dr_plot(result_RTH_DR1, result_RTHW_DR1)
dr_plot(result_RTH_DR10, result_RTHW_DR10)
dr_plot(result_RTH_DR20, result_RTHW_DR20)

dr_plot(result_TOUH_DR1, result_TOUHW_DR1)
dr_plot(result_TOUH_DR10, result_TOUHW_DR10)
dr_plot(result_TOUH_DR20, result_TOUHW_DR20)
