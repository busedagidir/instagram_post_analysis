import re
import os
import sys
import csv
import json
import time
import yaml
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import logging
import coloredlogs

# Logger
logger = logging.getLogger("ColorLogger")
coloredlogs.install(level='DEBUG', logger=logger)

# Training - Inference Phase
# This scripts aims to send requests to LLaVa model and test them

csv.field_size_limit(sys.maxsize)

class LLavaRequest:
    def __init__(self, config_path):
        # Dynamically determine the project root
        self.project_root = Path(__file__).resolve().parent.parent  # Adjust as needed
        self.config_path = self.project_root / config_path
        #print(f"Project root: {self.project_root}")
        #print(f"Config path: {self.config_path}")

        # Load the configuration
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.csv_file_path = self.project_root / "annotated_data" / self.config["csv_file_path"]
        node_name = self.config["node_name"]
        self.api_url = f"http://{node_name}.medien.uni-weimar.de:11434/api/generate" #buradaki port adresi degisebilir buna dikkat et

    # create dynamic prompt
    def create_prompt(self, caption):
        prompt = self.config["model"]["prompt"]
        prompt += f"Caption: {caption}"
        return prompt

    
    def create_payload_json(self, caption, base64_str: List[str]) -> Dict[str, Any]:
        payload = {
            "model": self.config["model"]["name"],  # "llava",
            "prompt": self.create_prompt(caption),
            "images": base64_str if isinstance(base64_str, list) else [base64_str],
            "format": "json",
            "options": {
                "num_ctx": self.config["model"]["options"]["num_ctx"],  # 32768,
                "temperature": self.config["model"]["options"]["temperature"]  # 0.1
            },
            "stream": self.config["model"]["stream"]
        }
        return payload


    def save_json(self, all_responses):
        dir_name = "llava_responses"
        llava_responses_path = os.path.join(self.project_root, dir_name)
        os.makedirs(llava_responses_path, exist_ok=True)

        with open(os.path.join(llava_responses_path, "llava_all_responses.json"), "w", encoding="utf-8") as outfile:
            json.dump(all_responses, outfile, ensure_ascii=False, indent=4)
        print("All responses saved to llava_all_responses.json.")

    def send_request(self):
        all_responses = []
        timeout_limit = 10
        with open(self.csv_file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                caption = row["body"]  # caption column
                #base64_str = row["filename"].split(',')[1]  # filename (base64 string)
                if "," in row["filename"]:
                    base64_str = row["filename"].split(",")[1]
                else:
                    print(f"Warning: Unexpected filename format in row: {row['filename']}")
                    base64_str = ""

                annotation_id = row.get('annotation_id') or row.get('id')
                logger.info(f"Processing row annotation_id = {annotation_id}")

                payload = self.create_payload_json(caption, base64_str)
                
                # write paylaod to file
                payload_file = "payload.json"
                with open(payload_file, "w", encoding="utf-8") as jsonfile:
                    json.dump(payload, jsonfile, ensure_ascii=False, indent=4)

                # Measure request duration
                start_time = time.time()

                curl_command = [
                    "curl",
                    self.api_url,
                    "-H",
                    "Content-Type: application/json",
                    "--data-binary",
                    f"@{payload_file}"
                ]

                try:
                    # Run curl command with timeout control
                    process = subprocess.run(
                        curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_limit
                    )
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # Log execution time
                    logger.info(
                        f"Request processing time for annotation_id = {annotation_id}, Took {execution_time:.2f} seconds")

                    #print(f"Request for caption: {caption}")
                    if process.returncode == 0:
                        response = process.stdout.decode().strip()
                        #print(f"response: {response}")
                        try:
                            # save as JSON
                            response_data = json.loads(response)
                            all_responses.append({
                                "response": response_data,
                                "base64_str": base64_str,
                                "annotation_id": annotation_id
                            })
                            print(f"Response for post annotation_id = {annotation_id} added to all_responses list.")
                        except json.JSONDecodeError:
                            print(f"Invalid JSON response for caption: {caption}")
                    else:
                        print(f"Error occurred: {process.stderr.decode()}")
                except subprocess.CalledProcessError as e:
                    print(f"Subprocess failed: {e.stderr.decode()}")
                    continue
                except subprocess.TimeoutExpired:
                    # Log and skip processing if timeout occurs
                    logger.warning(
                        f"Request for annotation_id = {row['annotation_id']} exceeded timeout limit of {timeout_limit} seconds. Skipping.")
                    print(f"Timeout exceeded for annotation_id = {row['annotation_id']}. Skipping.")

                # Add delay between requests
                time.sleep(1)

                # (optional)
                os.remove(payload_file)

        # save response json file
        self.save_json(all_responses)
        print(f"Len of responses: {len(all_responses)}")
        """
        
        dir_name = "llava_responses"
        llava_responses_path = os.path.join(self.project_root, dir_name)
        os.makedirs(llava_responses_path, exist_ok=True)

        with open(os.path.join(llava_responses_path, "llava_all_responses.json"), "w", encoding="utf-8") as outfile:
            json.dump(all_responses, outfile, ensure_ascii=False, indent=4)
        print("All responses saved to llava_all_responses.json.")
        """


if __name__ == "__main__":
    llava_request = LLavaRequest("config/config.yml")
    llava_request.send_request()