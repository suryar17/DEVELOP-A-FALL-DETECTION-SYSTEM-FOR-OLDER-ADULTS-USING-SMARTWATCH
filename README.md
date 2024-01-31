# DEVELOP-A-FALL-DETECTION-SYSTEM-FOR-OLDER-ADULTS-USING-SMARTWATCH

This paper introduces a smartwatch-based system using movement data to detect falls in older adults, potentially reducing injuries. The system, combining machine learning and signal processing, accurately distinguishes falls from daily activities.
# Features
In this project we take the live datasets from smartwatch by using the smartwatch mobile app and somecase we can extract the live dataset from mobile. The app is Physics Toolbox Sensor Suite

The fall detection system utilizes a smartwatch's gyroscope sensor to capture movement data, employing machine learning to accurately identify falls. Its key features include continuous monitoring, real-time fall detection, and the potential to improve safety for older adults by preventing injuries
# Wrok Flow
![image](https://github.com/suryar17/DEVELOP-A-FALL-DETECTION-SYSTEM-FOR-OLDER-ADULTS-USING-SMARTWATCH/assets/75236145/b0390699-12fa-4093-8da5-48c043ba8e22)
# Requirements
* Python
* Smartwatch
# Input Form
![Screenshot 2023-11-26 124353](https://github.com/suryar17/DEVELOP-A-FALL-DETECTION-SYSTEM-FOR-OLDER-ADULTS-USING-SMARTWATCH/assets/75236145/aefd3201-0df4-449f-9122-a050dca35bf2)
# program
```python
import pandas as pd
# Read data from ac.csv
ac_data = pd.read_csv('ac.csv')

# Keep only 'times' and 'az' columns
ac_data_filtered = ac_data[['times', 'az (m/s^2)']]

# Convert 'times' column to minutes
ac_data_filtered['times'] = (ac_data_filtered['times'] // 60) + 1

# Define the range for normal variations
lower_limit = -10
upper_limit = 10

# Initialize variables to track the start and end times of a fall event
fall_start_time = None
fall_end_time = None
fall_detected = False

# Check for "fall" events and find the time range
for index, row in data.iterrows():
    z_value = row['z']
    if z_value < lower_limit or z_value > upper_limit:
        if fall_start_time is None:
            fall_start_time = row['time']
        fall_end_time = row['time']
        fall_detected = True
    else:
        if fall_detected:
            print(f"Fall detected from {fall_start_time} to {fall_end_time}")
            fall_start_time = None
            fall_end_time = None
            fall_detected = False
    
# Read data from heart.csv
heart_data = pd.read_csv('heart.csv')

# Merge DataFrames based on the 'times' column
merged_data = pd.merge(ac_data_filtered, heart_data, left_on='times', right_on='min', how='left')

# Fill NaN values in the 'heartrate' column using forward-fill
merged_data['heartrate'].fillna(method='ffill', inplace=True)

# Create a new 'beat' column
merged_data['beat'] = merged_data['heartrate']

# Drop unnecessary columns
merged_data.drop(['min', 'heartrate'], axis=1, inplace=True)

# Add a column to check if conditions are met
merged_data['fall_detected'] = (
    ((merged_data['beat'] < 70) | (merged_data['beat'] > 100)) & 
    ((merged_data['az (m/s^2)'].diff().abs() > 10) | (merged_data['az (m/s^2)'].diff(-1).abs() > 10))
)

# Get unique times where falls are detected
unique_fall_times = merged_data.loc[merged_data['fall_detected'], 'times'].unique()


# Create a message for fall detection
if len(unique_fall_times) > 0:
    fall_message = "Fall detected at minute" if len(unique_fall_times) == 1 else "Fall detected at minutes"
    fall_message += " " + ", ".join(map(str, unique_fall_times))
    print(fall_message)
else:
    print("No falls detected.")

# Alert Notification to Whatsapp 
if(l==1):
    account_sid = 'ACf2128dace8c40d62e0d335326ed2c271'
    auth_token = 'ad3ae12753e6585637c67b21b94ccb11'
    client = Client(account_sid, auth_token)
    message = client.messages.create(from_='whatsapp:+14155238886',body='Fall Detected! Please check once',to='whatsapp:+919344857514')
    print(message.sid)
```
# Output

![image](https://github.com/suryar17/DEVELOP-A-FALL-DETECTION-SYSTEM-FOR-OLDER-ADULTS-USING-SMARTWATCH/assets/75236145/7844e9a7-d265-431b-9d88-889035473da5)

![image](https://github.com/suryar17/DEVELOP-A-FALL-DETECTION-SYSTEM-FOR-OLDER-ADULTS-USING-SMARTWATCH/assets/75236145/a8b2bfea-e0a5-45e3-a90f-97f1b1762db3)

# Result
In conclusion, Fall detection is a critical technology for the elderly and people with disabilities, as it can help to prevent serious injuries. Smartwatches are a promising platform for fall detection, as they are worn on the wrist, which is a good location for detecting falls, and they contain a variety of sensors that can be used to collect data on the user's movement and heart rate.
The proposed project will develop a new fall detection algorithm that is more accurate, energy-efficient, and robust to environmental factors than existing algorithms. The algorithm will be implemented on a smartwatch and evaluated in a real-world setting.



