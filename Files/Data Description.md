

* AGENCY column: --> 15 categories


* AGENCY NAME column --> same exact thing but not acronyms (i think it can be dropped)


* Problem (formerly Complaint Type) column --> 163 categories
Many have few categories --> look into it to understand better


* Problem Detail (formerly Descriptor) column --> 700 categories


* Additional Details column --> 577 categories


* Location Type --> 106 categories


* Incident Zip --> 222 categories


* Incident Address --> all different values i think


* Street Name (Incident Address stripped of the house number at the beginning) --> 6210 categories


* Cross Street 1 --> 7199 categories


* Cross Street 2 --> 7077


* Intersection Street 1 --> 5119 categories


* Intersection Street 2 --> 5148 categories


I think we can drop / unify all these columns on the location --> I would keep some easy and high-level information on the location, like Incident Zip or City or Borough


* Address Type --> 5 categories (can be useful IMO)


* City --> 54 categories
some are doubles, because written in caps lock and not in caps, so we must unite them and maybe this column could be useful


* Landmark --> 4636 categories and like 65k non-null
Idea: we could simply put a binary indicator if it has been marked as a landmark or not


* Facility Type --> only 363 non-null values, all of which are DSNY Garage, so maybe we could turn this into a binary variable to indicate it, or just drop it


* Community Board --> the column_description csv says there are 59 of them, but in the ds there are 77


* Borough --> 6 categories (5 boroughs + 117 unspecified observations)
these is just Community Board stripped of the first number so i would keep this only


* Council District (numerical values but categorical variable) --> 51 categories


* Police Precinct --> 78 categories


* BBL --> 43k categories


* Have to decide what to do with X and Y coordinates


* Open Data Channel Type --> 4 categories


* Park Facility Name --> 110k observations are of type Unspecified, other 7 are specified --> I would drop this


* Park Borough --> exact same as Borough (differ in 0 rows) --> drop this


* Vehicle Type --> 6 categories but only 5.3k non-null values (decide what to do)


* Taxi Company Borough --> only 70 non-null values (drop this)


* Bridge Highway Name --> useless


* Bridge Highway Direction --> even more useless


* Must decide what to do with Latitude and Longitude