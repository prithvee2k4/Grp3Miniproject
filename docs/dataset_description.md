# Step 2 - Dataset description

**1. Data source**
India Residential Energy Survey (IRES) 2020, run by the Council on Energy, Environment and Water
(CEEW). Published on Harvard Dataverse (doi:10.7910/DVN/AHFINM). File used: `CEEW20IRES_20Data.csv`.

**2. Why we chose it** *(edit this in your own words)*
It is a real, public, nationally representative household survey. The raw file is messy in realistic
ways (517 columns, mixed date formats, "don't know" codes, extreme outliers), so it needs a proper
ETL pipeline before it can be used for Data Science.

**3. What information it contains**
One row per household: 14,851 households in 21 Indian states (about 150 districts), surveyed
between November 2019 and March 2020. Columns cover household members, house type, income and
expenditure category, electricity access and bills, appliances (fans, ACs, fridges, TVs), cooking
fuel and satisfaction with electricity supply.

**4. Structured, semi-structured or unstructured?**
Structured: a flat table of rows and columns. (A few columns hold free text such as "other, please
specify" answers.)

**5. Format**
CSV (comma-separated values), about 13 MB.

**6. Update frequency**
It is a one-time survey snapshot (fieldwork Dec 2019 - Feb 2020), not a continuously updated feed.
CEEW may publish later survey rounds, but this file does not change.
