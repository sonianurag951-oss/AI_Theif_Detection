CREATE TABLE gridguard_data (
    customer_id VARCHAR(50),
    meter_id VARCHAR(50) PRIMARY KEY,
    day_1 FLOAT,
    day_2 FLOAT,
    day_3 FLOAT,
    day_4 FLOAT,
    day_5 FLOAT,
    day_6 FLOAT,
    day_7 FLOAT,
    day_8 FLOAT,
    day_9 FLOAT,
    day_10 FLOAT,
    day_11 FLOAT,
    day_12 FLOAT,
    day_13 FLOAT,
    day_14 FLOAT,
    day_15 FLOAT,
    day_16 FLOAT,
    day_17 FLOAT,
    day_18 FLOAT,
    day_19 FLOAT,
    day_20 FLOAT,
    day_21 FLOAT,
    day_22 FLOAT,
    day_23 FLOAT,
    day_24 FLOAT,
    day_25 FLOAT,
    day_26 FLOAT,
    FLAG INT,
    meter_type VARCHAR(30),
    region VARCHAR(100),
    tariff_category VARCHAR(50),
    installation_date DATE
);

drop Table gridguard_data 


select * from gridguard_data 
limit 10

-- Compare normal vs theft consumption
select FLAG,count(*) as total_customers,
round(AVG(
        (day_1 + day_2 + day_3 + day_4 + day_5 +
         day_6 + day_7 + day_8 + day_9 + day_10 +
         day_11 + day_12 + day_13 + day_14 + day_15 +
         day_16 + day_17 + day_18 + day_19 + day_20 +
         day_21 + day_22 + day_23 + day_24 + day_25 +
         day_26)::numeric
    ), 2) AS average_consumption
	from gridguard_data
	group by FLAG


 -- Identify high-risk customers

SELECT
    customer_id,
    meter_id,
    region,
    tariff_category,
    FLAG,
    ROUND(
        (day_1 + day_2 + day_3 + day_4 + day_5 +
         day_6 + day_7 + day_8 + day_9 + day_10 +
         day_11 + day_12 + day_13 + day_14 + day_15 +
         day_16 + day_17 + day_18 + day_19 + day_20 +
         day_21 + day_22 + day_23 + day_24 + day_25 +
         day_26)::numeric, 2
    ) AS total_consumption
FROM gridguard_data
WHERE FLAG = 1
ORDER BY total_consumption ASC
LIMIT 20;

-- Theft detection by region



--  Electricity consumption analysis
select tariff_category,count(*) as total_consumption,
 ROUND(AVG(day_1 + day_2 + day_3 + day_4 + day_5 +
              day_6 + day_7 + day_8 + day_9 + day_10 +
              day_11 + day_12 + day_13 + day_14 + day_15 +
              day_16 + day_17 + day_18 + day_19 + day_20 +
              day_21 + day_22 + day_23 + day_24 + day_25 +
              day_26)::numeric, 2) AS avg_26_day_consumption
			  from gridguard_data
			  group by tariff_category
			  order by  total_consumption asc

-- Find suspicious consumption patterns

select customer_id, meter_id,
    region,
    tariff_category,
	FLAG,
 ROUND((
        day_1 + day_2 + day_3 + day_4 + day_5 +
        day_6 + day_7 + day_8 + day_9 + day_10 +
        day_11 + day_12 + day_13 + day_14 + day_15 +
        day_16 + day_17 + day_18 + day_19 + day_20 +
        day_21 + day_22 + day_23 + day_24 + day_25 +
        day_26
    )::numeric, 2) AS total_consumption
	from gridguard_data
	where day_26<day_1*0.5
	order by total_consumption desc 

-- data quality
SELECT
    COUNT(*) FILTER (WHERE customer_id IS NULL) AS missing_customer_ids,
    COUNT(*) FILTER (WHERE meter_id IS NULL) AS missing_meter_ids,
    COUNT(*) FILTER (WHERE FLAG NOT IN (0, 1)) AS invalid_labels,
    COUNT(*) FILTER (WHERE meter_type IS NULL) AS missing_meter_types,
    COUNT(*) FILTER (WHERE region IS NULL) AS missing_regions
FROM gridguard_data;


-- summary

CREATE OR REPLACE VIEW gridguard_customer_summary AS
select customer_id,
    meter_id,
    region,
    meter_type,
    tariff_category,
    FLAG,
    ROUND((
        day_1 + day_2 + day_3 + day_4 + day_5 +
        day_6 + day_7 + day_8 + day_9 + day_10 +
        day_11 + day_12 + day_13 + day_14 + day_15 +
        day_16 + day_17 + day_18 + day_19 + day_20 +
        day_21 + day_22 + day_23 + day_24 + day_25 +
        day_26
    )::numeric, 2) AS total_consumption,
	 ROUND((
        day_1 + day_2 + day_3 + day_4 + day_5 +
        day_6 + day_7 + day_8 + day_9 + day_10 +
        day_11 + day_12 + day_13)::numeric,2) as first_13days, 
 ROUND((
		 day_14 + day_15 +
        day_16 + day_17 + day_18 + day_19 + day_20 +
        day_21 + day_22 + day_23 + day_24 + day_25 +
        day_26)::numeric,2) as last_13days 
		from gridguard_data

	select * from gridguard_customer_summary



-- Daily average consumption by tariff category
select tariff_category,
round(avg(day_1)::numeric,2) as avg_day1,
round(avg(day_2)::numeric,2) as avg_day2,
round(avg(day_3)::numeric,2) as avg_day3,
round(avg(day_4)::numeric,2) as avg_day4,
round(avg(day_5)::numeric,2) as avg_day5
from gridguard_data
group by tariff_category

-- Find customers with unusually low consumption
with customer_consumption as (
select tariff_category,
customer_id,
meter_id,
FLAG,round(
(day_1 + day_2 + day_3 + day_4 + day_5 +
         day_6 + day_7 + day_8 + day_9 + day_10 +
         day_11 + day_12 + day_13 + day_14 + day_15 +
         day_16 + day_17 + day_18 + day_19 + day_20 +
         day_21 + day_22 + day_23 + day_24 + day_25 +
         day_26)::numeric ,2) AS total_consumption
		 
		 from gridguard_data
	)
	select * from customer_consumption
	where total_consumption< (
	select avg(total_consumption)
	from customer_consumption
)
order by total_consumption

-- rank over customer
SELECT
    customer_id,
    meter_id,
    tariff_category,
    FLAG,
    ROUND((
        day_1 + day_2 + day_3 + day_4 + day_5 +
        day_6 + day_7 + day_8 + day_9 + day_10 +
        day_11 + day_12 + day_13 + day_14 + day_15 +
        day_16 + day_17 + day_18 + day_19 + day_20 +
        day_21 + day_22 + day_23 + day_24 + day_25 + day_26
    )::numeric, 2) AS total_consumption,
    RANK() OVER (
        ORDER BY (
            day_1 + day_2 + day_3 + day_4 + day_5 +
            day_6 + day_7 + day_8 + day_9 + day_10 +
            day_11 + day_12 + day_13 + day_14 + day_15 +
            day_16 + day_17 + day_18 + day_19 + day_20 +
            day_21 + day_22 + day_23 + day_24 + day_25 + day_26
        ) DESC
    ) AS consumption_rank
FROM gridguard_data;
