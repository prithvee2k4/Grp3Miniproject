# Step 6 - Transformations and why each one is needed

| # | Transformation | Member | Why it was necessary |
|---|----------------|--------|----------------------|
| 1 | Rename columns | 2 | Raw names are survey codes (`q234_month_exp`); readable names make analysis easier |
| 2 | Remove duplicate records | 2 | Guarantees one row per household so averages are not inflated (none found in this file, but the check protects future data) |
| 3 | Remove unwanted columns | 2 | 190 of 517 columns are more than 90% empty and carry no usable information |
| 4 | Standardise text | 2 | Extra spaces and mixed case would split the same state/district into different groups |
| 5 | Convert date formats | 2 | Dates arrive as both `12/18/2019` and `12-10-2019` text; real dates allow monthly reports |
| 6 | Change data types | 2 | Category and small integer types use less memory and prevent wrong calculations |
| 7 | Replace "don't know" codes | 3 | Codes 88/99 are answers, not numbers; left in, they distort averages |
| 8 | Validation checks / remove invalid values | 3 | Age 18-100, household size at least 1, positive land area and spending, age groups add up to household size |
| 9 | Cap outliers | 3 | Some land areas are over 100 million sq ft; extremes are capped at the 1st/99th percentile |
| 10 | Handle missing values | 3 | Key money/area columns filled with the state median (with a `_was_missing` flag); unanswered "not applicable" questions are left empty on purpose |
| 11 | Create calculated columns | 4 | Per-member bill and spending, bill share of spending, people per bedroom make households comparable |
| 12 | Create categories | 4 | Age, household size, spending and bill bands are easier to group and report on |
| 13 | Ranking | 4 | Ranks states and households by electricity bill |
| 14 | Sort records | 4 | Predictable order: state, district, household id |
| 15 | Generate summary statistics / grouping / aggregation | 5 | Feeds the reports |
