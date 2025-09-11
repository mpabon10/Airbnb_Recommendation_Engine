# Airbnb Recommendation Engine

This project aims to build a recommendation engine for Airbnb listings. The goal is to develop a system that can analyze user behavior and provide personalized suggestions based on their preferences.

## Project Structure

The project consists of several directories:

* `data`: contains CSV files with listing data
* `libsvm`: contains code for converting listings data into LIBSVM format
* `listing_similarity`: contains code for calculating similarities between listings
* `models`: contains trained models for making predictions
* `utils`: contains utility functions for data processing and model evaluation

## Major Python Files

### `libsvm.py`

This file contains code for converting listings data into LIBSVM format. It uses Spark to read in feature dictionaries, join them with Parquet data, group by user ID, collect features, sort features, and convert them to LIBSVM format.

### `listing_similarity.py`

This file contains code for calculating similarities between listings based on multiple factors such as host type, neighborhood, accommodates count, price, and rating. It uses Spark to process the data and calculate weighted average similarity scores.

### `models.py`

This file contains trained models for making predictions. The specific models used are not publicly available due to confidentiality agreements with Airbnb.

## Usage Examples

To run the project, follow these steps:

1. Install necessary dependencies using `pip install -r requirements.txt`
2. Set up a Spark session using `spark-shell --master local[4]`
3. Run `libsvm.py` to convert listings data into LIBSVM format
4. Run `listing_similarity.py` to calculate similarities between listings

Note: The usage examples above are simplified and may not reflect the actual steps required to run the project.

## Contributing

Contributions to this project are welcome! Please follow these guidelines:

1. Fork the repository on GitHub
2. Create a new branch for your feature or bug fix
3. Commit changes with descriptive commit messages
4. Push changes to your forked repository
5. Submit a pull request to merge your changes into the main repository

## License

Creative Commons Legal Code