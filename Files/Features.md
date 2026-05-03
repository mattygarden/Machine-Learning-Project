Creating the target variable Y:
    Y = 1 if Date Closed - Date Created < 24 hrs
    Y = 0 if Date Closed - Date Created > 24 hrs

There are some rows where Date Closed is NaN (blank)
I assume the date in which the dataset is "published" is the latest among the dates --> 2026-04-19 01:35:00 = max_date
Then for each row with Date Closed == NaN we check:
    max_date - Date Created < 24 hrs --> this means we can't properly evaluate the ticket because it did not have the chance yet to reach the 24 hrs cutoff --> WE DROP THE ROW
    max_date - Date Created > 24 hrs --> this means the ticket has been open for more than 24 hrs for sure and still was not closed --> Gets Y = 0

We observe no dropped rows by this logic, meaning the dataset actually includes only rows which are evaluated at least 24 hours before the last date included


The dataset has a Y proportion of 61% (1) vs 39% (0)
This must be taken into account when we create a validation set on the train set (ideally 80-20 split) --> simply done in the function train_test_split(..., stratify = True)


REMEMBER WE CAN ONLY INPUT IN THE MODEL AS A FEATURE THE DATE CREATED
Why? --> Because if we also input the date closed, we would basically be incurring into data leakage, the model could see exactly when the difference is less than 24 hours!

So, to input information about the date created, we convert the date format into multiple columns:
Created_Hour --> hour of creation of the ticket (0 - 23)
Created_DayOfWeek --> day of the week (monday = 0 - 6 = sunday)
Created_Month --> month (1 - 12)
Is_weekend --> 1 if sunday or saturday, 0 else





FEATURES TO DROP
Unique Key
Created Date --> Because we already include this through the other columns (Created_Hour...)
Closed Date
Agency Name --> We already have the acronyms in Agency
Problem (formerly Complaint Type) --> substituted by Problem_Grouped
Problem Detail (formerly Descriptor) --> too fine grained
Additional Details 
Location Type
Incident Address
Street Name
Cross Street 1
Cross Street 2
Intersection Street 1
Intersection Street 2
Address Type --> 97k/110k are category ADDRESS and I think this is already captured by Location_Grouped
City --> Drop it because we already keep Incident Zip and Borough
Landmark
Facility Type --> Only 300 values, not relevant
Community Board --> it is just Borough with a number in front
Council District
BBL
Park Facility Name --> 110851 Unspecified
Park Borough --> exaclty the same as Borough
Vehicle Type
Taxi Company Borough
Taxi Pickup Location
Bridge Highway Name
Bridge Highway Direction
Road Ramp
Bridge Highway Segment
X Coordinate (State Plane)
Y Coordinate (State Plane)
Latitude
Longitude
Location



FEATURES MODIFIED
Problem (formerly Complaint Type) --> Problem_Grouped
    turned this from 163 categories to a 9 category variable, by keyword grouping, with the category "Other" being 11.6% of the data

Location Type --> Location_Grouped
    encoded this from 106 to 6 categories including "Other/Mixed" which is 17%

Landmark --> Is_Landmark
    Binary variable to indicate whether the request is identified as a landmark (no detail included)
    1 if Landmark
    0 if NaN
    Tries to answer the question: does identifying the request with a landmark help in closing the ticket earlier?

Vehicle Type --> Is_Taxi
    from a 5k column, turn this into a binary indicator to only assess whether the request comes from a Taxi Vehicle
    No need to carry the Vehicle information because they are too few non-null values.



