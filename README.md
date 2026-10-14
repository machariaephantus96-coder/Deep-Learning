# AI Resume Screening: Predicting Hiring Decisions

A machine learning project that predicts whether a job candidate is **Selected** or **Rejected** from their resume profile, using a 100,000-record resume screening and ATS (applicant tracking system) dataset and a neural network built with TensorFlow/Keras.

> **Status:** the data analysis in this project is complete. The model results are being re-run after a data leakage bug was found in the training pipeline. See [Known issue](#known-issue-data-leakage-in-the-first-model-run) below.

## Problem

Recruiters receive far more applications than they can read. This project explores what in a candidate's profile is linked to being selected, and whether a model can learn to make that decision.

- **Task:** binary classification (`Selected` vs `Rejected`)
- **Target:** `selected`
- **Balance:** perfectly balanced, 50,000 selected and 50,000 rejected

## Dataset

- **Source:** "Resume Screening and ATS Hiring Dataset (100k records)" from Kaggle
- **Size:** 100,000 candidates and 44 columns, with no missing values and no duplicate rows
- **Note:** the data appears to be synthetic (generated names, emails, phone numbers and profile links). Findings describe this dataset and should not be read as statements about real hiring.

| Group | Columns |
|---|---|
| Identity and contact | `candidate_id`, `full_name`, `email`, `phone`, `github_profile`, `linkedin_profile` |
| Demographics | `age`, `gender`, `country`, `city` |
| Education | `highest_education`, `university`, `field_of_study`, `cgpa`, `graduation_year` |
| Experience | `experience_years`, `previous_companies`, `current_job_title`, `internship_experience`, `leadership_experience`, `projects_completed`, `publications` |
| Skills | `technical_skills`, `programming_languages`, `frameworks`, `databases`, `cloud_platform`, `certifications` |
| Assessment scores | `communication_score`, `problem_solving_score`, `technical_test_score`, `interview_score`, `aptitude_score` |
| Resume and ATS | `resume_quality_score`, `resume_length`, `keyword_match_percentage`, `ats_score` |
| Job preferences | `job_role`, `expected_salary`, `preferred_location`, `employment_type`, `remote_preference`, `availability` |

## Workflow

### 1. Data checks
Checked duplicates, shape, missing values, data types, and summary statistics for the numerical and categorical columns. Several columns use the text `Not Applicable` (for example `previous_companies`, `programming_languages` and `certifications`) to mean "none", which is a real category and not a missing value.

### 2. Exploratory analysis
- Target distribution (balanced).
- Age against years of experience (scatter plot).
- Application outcome against years of experience.
- Selection rate by education level. The rates are almost identical across levels: PhD 51.3%, Diploma 50.7%, High School 50.5%, Bachelor 49.9%, Master 49.5%. Education level on its own tells us very little about who is selected.
- Correlation heatmap across all numerical variables.

### 3. Hypothesis testing
Chi-square tests of independence between each categorical column and `selected`.

| Significant association (p ≤ 0.05) | No significant association |
|---|---|
| `previous_companies`, `current_job_title`, `internship_experience`, `leadership_experience`, `programming_languages`, `databases`, `cloud_platform` | `gender`, `country`, `city`, `highest_education`, `university`, `field_of_study`, `technical_skills`, `frameworks`, `certifications`, `job_role`, `preferred_location`, `employment_type`, `remote_preference`, `availability` |

Gender and location showing no association with selection is a reassuring result from a fairness point of view. The chi-square test is unreliable on columns with tens of thousands of unique values (names, emails, profile links), so those results should not be over-read.

### 4. Feature selection
Dropped identifiers, contact details, free-text profile links and high-cardinality or non-significant columns (`candidate_id`, `full_name`, `email`, `phone`, `country`, `city`, `gender`, `highest_education`, `university`, `field_of_study`, `technical_skills`, `frameworks`, `github_profile`, `linkedin_profile`, `certifications`, `job_role`, `preferred_location`, `employment_type`, `remote_preference`, `availability`). Kept 16 numerical features and 6 categorical ones.

### 5. Preprocessing
- Stratified split: **70% train / 15% validation / 15% test** (70,000 / 15,000 / 15,000).
- `ColumnTransformer` with a scikit-learn `Pipeline`: `StandardScaler` for numerical columns and `OneHotEncoder(handle_unknown="ignore")` for categorical columns. Fitted on the training set only.
- Target encoded with `LabelEncoder`.
- The one-hot encoding produces **26,104 input features**, almost all of them from `previous_companies`, which has 34,104 distinct values.

### 6. Model
A feed-forward neural network in TensorFlow/Keras:

| Layer | Units | Activation |
|---|---|---|
| Dense | 64 | ReLU |
| Dense | 32 | ReLU |
| Dense | 32 | ReLU |
| Output | 3 | Softmax |

- **Optimizer:** Adam
- **Loss:** sparse categorical cross-entropy
- **Training:** batch size 32, up to 200 epochs, with early stopping on validation loss (patience 5, best weights restored)
- **Hardware:** Kaggle notebook with a Tesla T4 GPU

## Known issue: data leakage in the first model run

The first training run reported **100% accuracy** on the validation and test sets (precision, recall and F1 all 1.00 on 15,000 test candidates). That result is not real. It comes from **target leakage**:

- The list of columns dropped when building `X` did not include `selected`.
- `selected` is an object column, so it was picked up as a categorical feature and one-hot encoded.
- The network was therefore given the answer as an input, and it learned to copy it. The loss falling to zero within a few epochs is the sign of this.

**The fix** is to drop the target from the features before splitting:

```python
X = data.drop(columns=["selected", "candidate_id", "full_name", ...])
y = data["selected"]
```

Other changes to make on the re-run:

- Use a **2-unit softmax** (or a 1-unit sigmoid with binary cross-entropy) for a binary target, instead of a 3-unit output.
- Reduce or encode `previous_companies` (for example, a count of previous companies or a "none" flag) instead of 34,104 one-hot columns.
- Compare against a simple baseline such as logistic regression, so the neural network's result has something to be measured against.
- Report precision, recall, F1 and a confusion matrix on the held-out test set only.

Updated results will be added here after the re-run.

## What the analysis shows so far

- The dataset is balanced, so accuracy is a fair metric here.
- Selection rates barely differ by education level, and gender, country and city show no significant link to selection.
- Experience-related features (previous companies, job title, internships, leadership) and technical-stack features (programming languages, databases, cloud platform) are the categorical columns linked to the decision.
- Age, years of experience and graduation year move together, so they carry largely the same information. Keep only one of them, or build a combined feature, when modelling.

## Project structure

```
├── notebook.ipynb          # full analysis and model training
├── README.md
└── data/                   # not included; download from Kaggle
```

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy scipy scikit-learn seaborn matplotlib tensorflow jupyter
jupyter notebook notebook.ipynb
```

Download the dataset from Kaggle and update the file path at the top of the notebook.

## Tech stack

Python, pandas, NumPy, SciPy (chi-square tests), scikit-learn (pipelines, preprocessing, metrics), TensorFlow/Keras, seaborn, matplotlib

## Author

**Ephantus Macharia**, aspiring data analyst, Nairobi, Kenya
[GitHub](https://github.com/machariaephantus96-coder)
