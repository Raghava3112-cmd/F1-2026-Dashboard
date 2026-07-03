# 🏎️ Formula 1 2026 Season Analytics Dashboard

![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![DAX](https://img.shields.io/badge/DAX-Advanced-blue?style=for-the-badge)
![Data Modeling](https://img.shields.io/badge/Data_Modeling-Star_Schema-success?style=for-the-badge)

## 📌 Project Overview
This project is an end-to-end data analytics dashboard built in **Power BI**, designed to provide broadcast-quality insights into the 2026 Formula 1 season. It transforms raw race telemetry, results, and pit stop data into actionable strategic insights, analyzing both driver performance and constructor efficiency. 

The dashboard features a dark-themed, premium UI (inspired by the F1 TV broadcast) and utilizes advanced DAX engineering to calculate complex metrics such as True Points Lost to DNFs, Lap 1 Aggression, and Driver Consistency.

*(Include a screenshot of Page 1 here)*

## 📊 Dashboard Pages & Features

### 1. The Championship Battle
A high-level executive view of the title fight.
* **Cumulative Points Race:** A dynamic line chart tracking driver points round-by-round.
* **Constructor Power Share:** Donut chart visualizing constructor dominance.
* **Season Scorecard:** A dense, color-coded matrix showing every driver's points across all races.

### 2. Driver Deep Dive
An analytical breakdown of individual driver performance.
* **Positions Fought:** Analyzes overtaking aggression vs. pure pace.
* **Adjusted vs. Actual Points:** Isolates the "reliability cost" by showing what the championship would look like without mechanical DNFs.
* **Teammate Extraction %:** Compares a driver's performance directly against the capability of their car.

### 3. Head-to-Head Arena
A dynamic comparison tool allowing users to pit any two drivers against each other.
* **The Consistency Rating:** Uses statistical standard deviation to measure lap-by-lap and race-by-race consistency.
* **Pit Box Athlete:** Analyzes the median pit stop duration to evaluate driver precision in the pit lane.
* **The Sniper:** Tracks the rawest pace metric—total Fastest Laps secured.

### 4. The Constructor War Room
A strategy page built for Team Principals to analyze team operations.
* **The Development Race:** Area chart tracking cumulative team points to visualize mid-season upgrades.
* **Mechanical Nightmares:** Tracks total DNFs per team to evaluate car reliability.
* **The Carry Factor:** A 100% stacked bar chart revealing if a team is perfectly balanced or overly reliant on a single driver.

## 🛠️ Technical Stack & Skills Demonstrated
* **Tool:** Power BI Desktop
* **Data Processing:** Power Query (Data cleaning, type casting, merging historical datasets)
* **Data Modeling:** Star Schema (1-to-Many relationships, handling inactive relationships)
* **Languages:** DAX (Data Analysis Expressions)
* **Design:** Custom UI/UX, navigation menus, conditional formatting, dark mode aesthetics

## 🧠 Advanced DAX Highlights
This dashboard relies on advanced DAX rather than simple aggregations. Key examples include:

**1. True Points Lost to DNFs (Handling Context Transition):**
```dax
Points Lost to DNF = 
SUMX(
    FILTER('results', NOT('results'[statusId] IN {1, 11, 12, 13})),
    SWITCH(TRUE(),
        'results'[grid] = 1, 25,
        'results'[grid] = 2, 18,
        'results'[grid] = 3, 15,
        ...
        0
    )
) + 0
```

**2. The Consistency Rating (Statistical Variance):**
```dax
Finish Consistency (Std Dev) = 
IFERROR(STDEV.P('results'[positionOrder]), 0)
```

**3. Operational Pit Lane Precision (Removing Outliers):**
```dax
Driver Pit Stop Time (s) = 
DIVIDE(MEDIAN('pit_stops'[milliseconds]), 1000, 0)
```

## 📂 Dataset
The data originates from the [Ergast Developer API / Kaggle F1 Datasets](http://ergast.com/mrd/), filtered specifically to isolate the 2026 grid and constructor mappings.

## 🤝 Let's Connect!
If you're a recruiter or data enthusiast, I'd love to chat about this project!
* **LinkedIn:** [Raghavendra Yecham](https://www.linkedin.com/in/raghavendra-yecham-b0318b2b6/)
