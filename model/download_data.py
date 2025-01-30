from .tabularLLM.preprocessing.preprocess_kaggle_dataset import *
import argparse
import logging
from .tabularLLM.preprocessing.preprocess_kaggle_dataset import preprocess_metadata
import json

DEFAULT_DATASET_PATH = './dataset/'

def get_link_from_json(info_path):
    """
    Reads the "link" value from a JSON file.

    Args:
        info_path (str): The file path to the JSON file.

    Returns:
        str: The value of the "link" key in the JSON file.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: If the "link" key is not found in the JSON file.
    """
    try:
        with open(info_path, 'r') as file:
            data = json.load(file)
            if "link" in data:
                return data["link"]
            else:
                raise KeyError("The key 'link' was not found in the JSON file.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at path {info_path} was not found.") from e

def extract_kaggle_path(url: str) -> str:
    """
    Extract the portion of the Kaggle dataset URL after "https://www.kaggle.com/datasets/".

    Args:
        url (str): The Kaggle dataset URL.

    Returns:
        str: The extracted portion of the URL.
    """
    # Define the base prefix to identify the target portion
    base_prefix = "https://www.kaggle.com/datasets/"

    if url.startswith(base_prefix):
        # Extract the part after the base prefix
        return url[len(base_prefix):]
    else:
        raise ValueError("The provided URL is not a valid Kaggle dataset URL.")

def save_dataset_data(dataset_url, dataname):
    download_path = DEFAULT_DATASET_PATH + dataname

    try:
        kaggle.api.dataset_metadata(
            dataset_url,
            path=download_path
        )
    except Exception as e:
            print('failed')
            print(e)

def preprocess_all_metadata(dataname):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        count = 0
        item_path = DEFAULT_DATASET_PATH + dataname
        try:
            preprocess_metadata(item_path)
            count += 1
        except openai.error.RateLimitError:
            # retry until no ratelimit error
            result = None
            retry_count = 0  # 初始化重试次数
            max_retries = 5  # 设置最大重试次数
            while result is None and retry_count < max_retries:
                retry_count += 1
                logging.info(f'Retrying {retry_count}/{max_retries} for {item_path}')
                time.sleep(10)  # 等待 10 秒后重试
                try:
                    preprocess_metadata(item_path)
                    result = 'worked'
                    logging.info(f'Successfully processed {item_path} after {retry_count} retries')
                except openai.error.RateLimitError:
                    logging.warning(f'RateLimitError: retry {retry_count}/{max_retries} for {item_path}')
                except Exception as e:
                    logging.error(f'Error during retry: {e}')
                    break  # 如果出现其他异常，退出循环
            if retry_count == max_retries:
                logging.error(f'Max retries reached for {item_path}, skipping this item.')
        except Exception as e:
            print(f'failed: {e}')
            logging.error(f'Failed to process {item_path}: {e}')
    
def preprocess_all_data(dataname):
    DataObject(dataname)


def download_metadata(dataset_name):
    info_path = DEFAULT_DATASET_PATH + dataset_name + '/info.json'
    url = get_link_from_json(info_path)
    data_url = extract_kaggle_path(url)
    save_dataset_data(dataset_url=data_url,dataname=dataset_name)
    preprocess_all_metadata(dataname=dataset_name)
    preprocess_all_data(dataname=dataset_name)



