# Airbnb Recommendation Engine

An end-to-end recommendation system for Airbnb listings built with **PySpark** and **Spark ML**. The pipeline ingests raw NYC listing data, generates synthetic user booking histories, engineers rich features from user behaviour and listing metadata, and trains a **Factorization Machine** classifier to produce personalised listing recommendations for every user.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Steps](#pipeline-steps)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Key Concepts](#key-concepts)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture Overview

```
┌───────────────────┐     ┌──────────────┐
│ listing_similarity│     │   metadata   │
│    (Step 1)       │     │   (Step 2)   │
└────────┬──────────┘     └──────┬───────┘
         │                       │
         ▼                       │
┌───────────────────┐            │
│  simulate_txns    │            │
│    (Step 3)       │            │
└────────┬──────────┘            │
         │                       │
         ▼                       │
┌───────────────────┐            │
│  ones_n_zeros     │            │
│    (Step 4)       │            │
└────────┬──────────┘            │
         │                       │
         ▼                       │
┌───────────────────┐            │
│    cohorts        │            │
│    (Step 5)       │            │
└────────┬──────────┘            │
         │                       │
         ▼                       ▼
┌───────────────────────────────────┐
│           affinities              │
│            (Step 6)               │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│       feature_stitching           │
│            (Step 7)               │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│            libsvm                 │
│            (Step 8)               │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│              FM                   │
│            (Step 9)               │
│   Train → Evaluate → Recommend   │
└───────────────────────────────────┘
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `jobs/listing_similarity.py` | Computes pairwise similarity between all listings using a weighted combination of superhost status, neighbourhood, accommodates, price, and review rating. Retains the **top-100 most similar listings** per listing. |
| 2 | `jobs/metadata.py` | Transforms raw listing attributes into a normalised long-format metadata table. Continuous fields (price, rating) are discretised into bins. |
| 3 | `jobs/simulate_txns.py` | Generates **10,000 synthetic users** with exponentially-distributed booking frequencies (1–30 bookings). Sequential bookings are correlated via listing similarity. |
| 4 | `jobs/ones_n_zeros.py` | Creates labelled training data: positive examples (actual bookings) and negative examples (random non-booked listings at 2× ratio). Each example is enriched with **RFM** (Recency, Frequency, Monetary) features. Also builds the scoring candidate set. |
| 5 | `jobs/cohorts.py` | Segments users into **4 RFM cohorts** using K-Means clustering on weighted, scaled RFM dimensions. |
| 6 | `jobs/affinities.py` | Computes per-user affinity scores for each listing metadata facet (e.g. neighbourhood, price bin), normalised within each RFM cohort. |
| 7 | `jobs/feature_stitching.py` | Assembles three feature families — **user RFM bins**, **user affinities**, and **listing metadata** — into a unified sparse feature table. Generates a feature dictionary mapping feature names to integer IDs. |
| 8 | `jobs/libsvm.py` | Converts the assembled feature table into **LIBSVM-formatted** text files consumable by Spark ML. |
| 9 | `jobs/FM.py` | Trains a **Factorization Machine** classifier (factorSize=10, regParam=0.05), evaluates via PR-AUC on a 10% held-out test set, and scores candidates to produce **top-10 personalised recommendations** per user. |

## Project Structure

```
├── data/
│   ├── nyc_listings.csv                    # Raw NYC Airbnb listings
│   ├── nyc_listing_data_clean.csv          # Cleaned listing data
│   ├── nyc_listing_data_5pcnt_sample.csv   # 5% sample used by pipeline
│   ├── listing_metadata_parquet/           # Step 2 output
│   ├── top_100_sim_listings_parquet/       # Step 1 output
│   ├── users_parquet/                      # Step 3 output (synthetic users)
│   ├── transactions_parquet/               # Step 3 output (bookings)
│   ├── training/                           # Training artifacts (steps 4-8)
│   │   ├── ones_and_zeros_rfm_parquet/
│   │   ├── cohorts_parquet/
│   │   ├── affinities_parquet/
│   │   ├── features_values_label_parquet/
│   │   ├── feature_dictionary_csv/
│   │   ├── row_identifier_parquet/
│   │   └── libsvm/
│   └── scoring/                            # Scoring artifacts (steps 4-8)
│       ├── ones_and_zeros_rfm_parquet/
│       ├── cohorts_parquet/
│       ├── affinities_parquet/
│       ├── features_values_label_parquet/
│       ├── row_identifier_parquet/
│       └── libsvm/
├── jobs/
│   ├── run_pipeline.py              # Master pipeline orchestrator
│   ├── listing_similarity.py        # Step 1
│   ├── metadata.py                  # Step 2
│   ├── simulate_txns.py             # Step 3
│   ├── ones_n_zeros.py              # Step 4
│   ├── cohorts.py                   # Step 5
│   ├── affinities.py                # Step 6
│   ├── feature_stitching.py         # Step 7
│   ├── libsvm.py                    # Step 8
│   └── FM.py                        # Step 9
├── models/
│   └── factors_10_reg_0.05_model/   # Trained FM model
├── requirements.txt
├── LICENSE
└── ReadMe.md
```

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Apache Spark 3.5+** (PySpark)
- **Java 8 or 11** (required by Spark)

### Installation

```bash
# Clone the repository
git clone https://github.com/mpabon10/Airbnb_Recommendation_Engine.git
cd Airbnb_Recommendation_Engine

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

The easiest way to run everything end-to-end is with the pipeline orchestrator:

```bash
python jobs/run_pipeline.py
```

This executes all 9 steps in the correct order. Each step runs in its own subprocess with an isolated Spark session.

### Resume from a Specific Step

If a step fails, fix the issue and resume without re-running earlier steps:

```bash
python jobs/run_pipeline.py --start 4    # resume from step 4 onward
```

### Run a Single Step

```bash
python jobs/run_pipeline.py --only 5     # run only step 5 (cohorts)
```

### Run Individual Scripts

Each job can also be run independently (ensure dependencies have already been produced):

```bash
python jobs/listing_similarity.py
python jobs/metadata.py
python jobs/simulate_txns.py
# ... etc.
```

## Key Concepts

### Listing Similarity
A weighted composite score combining five attributes (superhost match, neighbourhood match, accommodates distance, price distance, review rating distance). Weights are configurable:  price (0.35), neighbourhood (0.25), superhost (0.20), accommodates (0.10), rating (0.10).

### RFM Framework
Each user's booking behaviour is characterised by three dimensions:
- **Recency** — days since last booking
- **Frequency** — total number of bookings
- **Monetary** — running average nightly price

### RFM Cohorts
Users are clustered into 4 segments via K-Means on weighted RFM features. Cohorts allow affinity scores to be normalised against similar users rather than the global population.

### User Affinities
For each metadata facet (e.g. "neighbourhood = Manhattan"), a user's engagement intensity is compared to their cohort peers and min-max normalised to produce a score in [0, 1].

### Factorization Machine
A Spark ML `FMClassifier` that models second-order feature interactions via latent factor vectors. This captures complex relationships (e.g. "users who prefer Manhattan also favour superhosts in the $200–300 price range") without explicit feature crossing.


## License

Creative Commons Legal Code