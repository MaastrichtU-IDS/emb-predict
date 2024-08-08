# ruff: noqa: S603
import logging
import ast
import subprocess
import requests


def set_logging(log_level=logging.INFO):
    log = logging.getLogger(__name__)
    log.propagate = False
    log.setLevel(log_level)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: [%(module)s:%(funcName)s] %(message)s"
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


log = set_logging()


def download_remote_directory(remote_file: str, download_path: str):
    wget_command = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=8",
        "-P",
        download_path,
        "-e",
        "robots=off",
        "-R",
        "index.html*",
        remote_file,
    ]

    result = subprocess.run(wget_command, capture_output=True, text=True)

    if result.returncode == 0:
        log.info(f"Download {remote_file} successful!")
        return True
    else:
        log.error(f"Download {remote_file} failed! {result.stderr}")
    return False


def download_file_to_string(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    return None


def parse_string_to_dict(input_string):
    result_dict = {}

    # Split the input string into lines
    lines = input_string.strip().split("\n")

    for line in lines:
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Remove surrounding quotes from strings, and convert to appropriate types
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value == "false":
            value = False
        elif value == "true":
            value = True
        elif value.startswith("[") and value.endswith("]"):
            value = ast.literal_eval(value)

        result_dict[key] = value

    return result_dict
