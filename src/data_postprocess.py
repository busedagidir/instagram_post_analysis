import os
import csv
import base64
import json
import logging
import coloredlogs
import numpy as np
import matplotlib.pyplot as plt

from requests_class import LLavaRequest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score


# Logger
logger = logging.getLogger("ColorLogger")
coloredlogs.install(level='DEBUG', logger=logger)

class PostProcess(LLavaRequest):
    def __init__(self, config_path):
        super().__init__(config_path)

    def trim_llava_labels(self, json_path):
        with (open(json_path, 'r') as f):
            data = json.load(f)
            logger.info(f"Read file from: {json_path}")
            print(f"Len json: {len(data)}")
            trimmed_lava_labels = []
            llava_ids = []
            
            for resp in data:
                llava_response = resp.get('response', {}).get('response', "")
                #llava_response = resp['response']['response']
                annotation_id = resp.get('annotation_id')
                try:
                    # check if the response in json format
                    if isinstance(llava_response, str) and llava_response.strip():
                        llava_response = json.loads(llava_response)
                        print(f"Lava response: {llava_response}")
                    sentiment_label = llava_response.get("overall_sentiment")
                    if not sentiment_label:
                        sentiment_label = llava_response.get("caption_sentiment") or \
                        llava_response.get("image_sentiment") or \
                        llava_response.get("sentiment")

                    if not sentiment_label or sentiment_label.lower().strip() not in ["positive", "negative",
                                                                                      "neutral"]:
                        logger.warning(f"Model couldn't label properly for id={annotation_id}: {llava_response}.\n"
                                       f"Labelling as 'neutral'")
                        sentiment_label = "neutral"

                        #llava_ids.append(resp.get("annotation_id", ""))
                    sentiment_label = sentiment_label.lower().strip()
                    trimmed_lava_labels.append(sentiment_label)
                    llava_ids.append(annotation_id)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response: {llava_response} for id = {annotation_id}")
                    trimmed_lava_labels.append(sentiment_label)
                    llava_ids.append(annotation_id) # added invalid ids also
            print(trimmed_lava_labels)
            print(len(trimmed_lava_labels))
            print(f"LLaVa response length: {(len(trimmed_lava_labels))}")
            return trimmed_lava_labels, llava_ids

                
    def get_label_studio_annotations(self):
        annotations = []
        base64_strings = []
        lstudio_ids = []
        with open(self.csv_file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                #image_label = row["image_label"]
                #image_label = image_label.lower()
                image_label = row.get("label_post", "").strip().lower()
                #base64_str = row["filename"]
                base64_str = row.get("filename", "").strip()
                annotation_id = row.get("annotation_id", "").strip()

                annotations.append(image_label)
                base64_strings.append(base64_str)
                lstudio_ids.append(annotation_id)
                
        print(f"Annotations: {annotations}")
        print(f"Label Studio annotations length: {len(annotations)}")
        return annotations, base64_strings, lstudio_ids

    def evaluate_labels(self, llava_list, annotation_list, llava_ids, lstudio_ids, base64_strings):
        # Convert lists into dictionaries for fast lookup
        llava_dict = {id_: label for id_, label in zip(llava_ids, llava_list)}
        lstudio_dict = {id_: label for id_, label in zip(lstudio_ids, annotation_list)}
        base64_dict = {id_: img_str for id_, img_str in zip(lstudio_ids, base64_strings)}

        #print(f"Llava dict: {llava_dict}")
        #print(f"Lstudio dict: {lstudio_dict}")

        # Get intersection of IDs to compare only matching ones
        common_ids = set(llava_dict.keys()) & set(lstudio_dict.keys())

        if not common_ids:
            print("No matching IDs found between LLaVA and Label Studio.")
            return [], []

        print(f"Total common IDs: {len(common_ids)}")

        mismatch_indices = []
        mismatch_images = []
        pn_opposite_count = 0
        np_opposite_count = 0
        nn_opposite_count = 0
        positive_match_count = 0
        negative_match_count = 0
        notr_match_count = 0

        for annotation_id in common_ids:
            llava_label = llava_dict[annotation_id]
            lstudio_label = lstudio_dict[annotation_id]

            if llava_label != lstudio_label:
                mismatch_indices.append(annotation_id)  # Save ID for later
                mismatch_images.append(base64_dict.get(annotation_id, None))  # Save base64 image string
                # Print mismatch details
                print(f"Mismatch: ID {annotation_id} - LLaVA = {llava_label}, Label Studio = {lstudio_label}")

            # Count specific mismatches
            if (llava_label == "positive" and lstudio_label == "negative") or \
                    (llava_label == "negative" and lstudio_label == "positive"):
                pn_opposite_count += 1

            if (llava_label == "neutral" and lstudio_label == "positive") or \
                    (llava_label == "positive" and lstudio_label == "neutral"):
                np_opposite_count += 1

            if (llava_label == "neutral" and lstudio_label == "negative") or \
                    (llava_label == "negative" and lstudio_label == "neutral"):
                nn_opposite_count += 1

            # Count exact matches
            if llava_label == "positive" and lstudio_label == "positive":
                positive_match_count += 1

            if llava_label == "negative" and lstudio_label == "negative":
                negative_match_count += 1

            if llava_label == "neutral" and lstudio_label == "neutral":
                notr_match_count += 1


            # Save mismatched images
            #self.save_mismatch_images(mismatch_indices, mismatch_images)
            
        print(f"Total mismatches: {len(mismatch_indices)}")
        print(f"Opposite count [Positive-Negative]: {pn_opposite_count}")
        print(f"Opposite count [Positive-Neutral]: {np_opposite_count}")
        print(f"Opposite count [Neutral-Negative]: {nn_opposite_count}")
        print(f"Match count [Positive-Positive]: {positive_match_count}")
        print(f"Match count [Negative-Negative]: {negative_match_count}")
        print(f"Match count [Neutral-Neutral]: {notr_match_count}")

        # Confusion matrix
        self.calculate_cm_recall_precision_f1(pn_opposite_count, np_opposite_count, nn_opposite_count, positive_match_count, negative_match_count, notr_match_count)

        return mismatch_indices, mismatch_images

    def calculate_cm_recall_precision_f1(self, pn_opposite_count, np_opposite_count, nn_opposite_count, positive_match_count, negative_match_count, notr_match_count):
        labels = ["positive", "negative", "neutral"]

        conf_matrix = np.zeros((3, 3), dtype=int)

        conf_matrix[0][0] = positive_match_count
        conf_matrix[0][1] = pn_opposite_count
        conf_matrix[0][2] = np_opposite_count

        conf_matrix[1][0] = pn_opposite_count
        conf_matrix[1][1] = negative_match_count
        conf_matrix[1][2] = nn_opposite_count

        conf_matrix[2][0] = np_opposite_count
        conf_matrix[2][1] = nn_opposite_count
        conf_matrix[2][2] = notr_match_count

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix)

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Add labels to axes
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, conf_matrix[i,j], ha="center", va="center", color="black")

        plt.xlabel("Predicted Labels (LLaVA)")
        plt.ylabel("True Labels (Label Studio)")
        plt.tight_layout()

        # Save the confusion matrix as an image
        output_path = '/mnt/ceph/storage/data-tmp/2024/gani7218/instagram-post-analysis/src/confusion_matrix.png'
        plt.savefig(output_path, dpi=300)  # Save at high resolution
        print(f"Confusion matrix saved to {output_path}")
        plt.close()  # Close the figure to avoid memory issues

        
        # Compute Precision, Recall, and F1-score per class
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for i in range(len(labels)):
            TP = conf_matrix[i][i]  # True Positives for class i
            FP = sum(conf_matrix[:, i]) - TP  # False Positives (Column Sum - TP)
            FN = sum(conf_matrix[i, :]) - TP  # False Negatives (Row Sum - TP)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1_score)

        # Compute Overall Accuracy
        total_samples = np.sum(conf_matrix)
        correct_predictions = sum(conf_matrix[i][i] for i in range(len(labels)))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # Print Metrics
        print("\n--- Classification Metrics ---")
        print(f"Overall Accuracy: {accuracy:.4f}")

        print("\nPer-Class Metrics:")
        for i, label in enumerate(labels):
            print(
                f"{label.capitalize()} -> Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1-Score: {f1_per_class[i]:.4f}")

        # Compute Macro-Average (Simple Mean of Each Class)
        #macro_precision = np.mean(precision_per_class)
        #macro_recall = np.mean(recall_per_class)
        #macro_f1 = np.mean(f1_per_class)
        #print("\nMacro-Averaged Metrics:")
        #print(
        #    f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1-Score: {macro_f1:.4f}")

        # Compute Weighted-Average (Each Class Weighted by Total Occurrences)
        #total_per_class = np.sum(conf_matrix, axis=1)
        #weighted_precision = np.sum(np.array(precision_per_class) * total_per_class) / np.sum(total_per_class)
        #weighted_recall = np.sum(np.array(recall_per_class) * total_per_class) / np.sum(total_per_class)
        #weighted_f1 = np.sum(np.array(f1_per_class) * total_per_class) / np.sum(total_per_class)

        #print("\nWeighted-Averaged Metrics:")
        #print(
        #    f"Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

    def save_mismatch_images(self, mismatch_ids, base64_strings):
        dir_name = "mismatch_imgs"
        mismatch_path = os.path.join(self.project_root, dir_name)
        print(f"Mismatch path: {mismatch_path}")
        os.makedirs(mismatch_path, exist_ok=True)

        for annotation_id, base64_str in zip(mismatch_ids, base64_strings):
            if not base64_str:
                print(f"Skipping ID {annotation_id}: No base64 image found.")
                continue  # Skip missing images
            try:
                base64_str = base64_str.split(",")[1] if "," in base64_str else base64_str
                image_data = base64.b64decode(base64_str)
                file_path = os.path.join(mismatch_path, f"mismatch_image_{annotation_id}.png")

                with open(file_path, "wb") as image_file:
                    image_file.write(image_data)
                    print(f"Saved mismatch image at: {file_path}")

            except Exception as e:
                print(f"Error saving image for ID {annotation_id}: {str(e)}")
        



if __name__ == '__main__':
    process = PostProcess("config/config.yml")
    llava_labels, llava_indices = process.trim_llava_labels("/mnt/ceph/storage/data-tmp/2024/gani7218/instagram-post-analysis/llava_responses/llava_all_responses.json")
    annotations, base64_strings, label_studio_indices = process.get_label_studio_annotations()
    mismatch_indices, _ = process.evaluate_labels(llava_labels, annotations, llava_indices, label_studio_indices, base64_strings)
    process.save_mismatch_images(mismatch_indices, base64_strings)
        
