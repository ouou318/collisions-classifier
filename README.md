In 2015, Seattle government launched Vision Zero and planned to end traffic deaths and serious injuries on city streets by 2030. They published collision data in Seattle from 2004 to 2020 online. I am curious if we could learn what are the driving factors that cause deadly accidents from the data. My project applied supervised machine learning models to classify the collision data. The goal is to identify key features to differentiate fatal accidents from other accidents. Given fatal accidents are rare, my model has to well take care of the imbalanced dataset: my baseline model shows terrible recall and precision scores for the minority class. You will learn what are the useful metrics to measure improvement for imbalanced dataset. You will learn how I improved AUC from 0.74 to 0.93 through resampling, feature engineering, adding new features, and model tuning. You will also learn how I identified important features using feature permutation and dropping columns.

## Sources
Seattle Traffic Collisions Dataset This dataset maintained by the city of seattle contains traffic collision report details from 2004 to present. This will be my main datasource for collision data. https://data-seattlecitygis.opendata.arcgis.com/datasets/collisions
There are other supporting datasets:

- Seattle street data: https://www.seattle.gov/Documents/Departments/SDOT/GIS/Seattle_Streets_OD.pdf
- Seattle street sign: https://www.seattle.gov/Documents/Departments/SDOT/GIS/Street_Signs_OD.pdf
- Seattle historical weather data: https://www.meteoblue.com/en/weather/archive/export/seattle_united-states-of-america_5809844

## Key Points
- Imbalanced data
- Feature importance 