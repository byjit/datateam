import os
import logging
from textwrap import dedent
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
from google.adk.tools import FunctionTool


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_kaggle_datasets(query: str, max_results: int = 10):
    """
    Searches for datasets on Kaggle and returns a list of results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of datasets, where each dataset is a dictionary of metadata.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        sort_by = 'votes' # 'hottest', 'votes', 'updated', 'active', 'published'
        filetype = 'all'
        page = 1
        datasets = api.dataset_list(sort_by=sort_by, file_type=filetype, search=query, page=page, min_size=1024)
        # Limit results and convert to dict for serialization
        return [d.to_dict() for d in datasets[:max_results]]
    except Exception as e:
        logging.error(f"An error occurred while searching for datasets: {e}")
        return []

def download_kaggle_dataset(source_url: str):
    """
    Downloads a dataset from Kaggle.

    Args:
        source_url (str): The Kaggle dataset URL

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        api = KaggleApi()
        download_path: str = os.path.join(os.getcwd(), "data")
        api.authenticate()
        dataset_ref = source_url.replace('https://www.kaggle.com/datasets/', '') # Extract dataset reference from URL

        logging.info(f"Downloading dataset '{dataset_ref}' to '{download_path}'...")
        api.dataset_download_files(dataset_ref, path=download_path, unzip=True)
        logging.info(f"Dataset '{dataset_ref}' downloaded and unzipped successfully.")
        return True
    except Exception as e:
        logging.error(f"An error occurred while downloading the dataset: {e}")
        return False



def search_sources_using_sonar(query: str):
    """
    Queries Sonar API to find the best sites for obtaining datasets related to the query.

    Args:
        query (str): The search query for datasets.

    Returns:
        dict: The API response containing recommended sites for datasets, or None if error occurs.
    """
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": "Bearer "+ os.getenv("PERPLEXITY_API_KEY"),
            "Content-Type": "application/json"
        }
        data = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": dedent(
                        """
                        You are a helpful assistant whose role is to help find sources from which either datasets can be downloaded or data can be scraped from to create a dataset. 
                        Be precise and concise. Do whichever is the best. If possible, also tell how to download the dataset or scrape the data.
                        If you cannot find any sources, say so.
                        """
                    )
                },
                {
                    "role": "user",
                    "content": f"Where can I find data for: {query}?"
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"An error occurred while querying Sonar API: {e}")
        return None


def save_to_local_file(data: dict, filename: str, type: str = 'txt'):
    """
    Saves the provided data to a local file in the specified format.

    Args:
        data (dict): The data to save.
        filename (str): The name of the file to save the data to.
        type (str): The format to save the data in ('txt' or 'json').
    """
    try:
        import json
        # Ensure the 'data' directory exists in the current working directory
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'w') as f:
            if type == 'json':
                json.dump(data, f, indent=4)
            else:
                f.write(str(data))
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving data to file: {e}")


search_kaggle_datasets_tool = FunctionTool(func=search_kaggle_datasets)
search_sources_using_sonar_tool = FunctionTool(func=search_sources_using_sonar)
download_kaggle_dataset_tool = FunctionTool(func=download_kaggle_dataset)
save_to_local_file_tool = FunctionTool(func=save_to_local_file)