import difflib
import json
import logging
import math
import os
import subprocess
import sys
import time

import cv2
import pandas as pd
import pytesseract
import torch
#from bbox_retrieval import OCRBoundingBox
#from data_ingestion import BaseDataIngestion

from data_ingestion.pdftoimage import PDF
#from pdftoimage import PDF

from nltk.tokenize import word_tokenize
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
#from sqlalchemy import update

#from utils.databases.sql_orm import sql_orm
#from utils.databases.table_schema import file_content
from data_ingestion.utils.helper import get_pytessaract_custom_config, get_stopwords, text_filteration
#from utils.helper import get_pytessaract_custom_config, get_stopwords, text_filteration

sys.path.append("unilm")
sys.path.append("detectron2")


from data_ingestion.unilm.dit.object_detection.ditod.config import add_vit_config
#from unilm.dit.object_detection.ditod.config import add_vit_config

# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultPredictor

#bbox = OCRBoundingBox()

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def generate_config():
    # Create a base configuration
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file("./data_ingestion/config/cascade_dit_base.yml")

    cfg.MODEL.WEIGHTS = "./local_volume/publaynet_dit-b_cascade.pth"
    #subprocess.call(["scripts/download_layoutlmv3.sh"])
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # You can directly use the cfg object as is for a default configuration
    return cfg

config = generate_config()
predictor = DefaultPredictor(config)
stop_words = get_stopwords()
logger = logging.getLogger(__name__)

MAPPING_PATH = "statics/es_mapping.json"

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
    def __str__(self):
        return f"Document(\n  page_content= '{self.page_content}', metadata= {self.metadata}\n)"

class ImageExtractor():
    def __init__(self, pdf_instance):  # Pass a PDF instance to ImageExtractor
        self.pdf = pdf_instance

    def model_results(self, image):
        # Set the metadata
        md = MetadataCatalog.get(config.DATASETS.TEST[0])
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

        # Predict the output of the image
        output = predictor(image)["instances"]

        return output

    def cosine_sim(self, arr_1, arr_2):
        similarity_array = cosine_similarity(arr_1, arr_2)
        return similarity_array

    def merge_boxes(self, box1, box2):
        # Merge two bounding boxes
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]

    def cal_header_footer(self, sorted_files, footer_threshold=0.99):
        """
        To calculate the header and footer:
        For each row of pixels in the image compare the first page
        with the rest of the pages (N-1 comparisons) using cosine similarity.
        If the cosine similarity is below a threshold, that row will be
        assigned as headr or footer.

        """
        # Calculate Header
        # Read the first image
        image1 = cv2.imread(os.path.join(self.pdf.folder_path, sorted_files[0]))
        flag = 0
        header = -20  # padding
        for row in range(self.pdf.height):
            for idx in range(1, len(sorted_files)):
                image2 = cv2.imread(
                    os.path.join(self.pdf.folder_path, sorted_files[idx])
                )
                arr1 = image1[row, :]
                arr2 = image2[row, :]
                # Calculate cosine similiarity and compare against a threshold
                if self.cosine_sim([arr1.flatten()], [arr2.flatten()])[0][0] < 0.9:
                    flag = 1
                    break
            if flag:
                break
        header += row

        flag = 0
        footer = 20  # padding
        for row in range(self.pdf.height - 1, 0, -1):
            for idx in range(1, len(sorted_files)):
                img2 = cv2.imread(os.path.join(self.pdf.folder_path, sorted_files[idx]))
                r1 = image1[row, :]
                r2 = img2[row, :]
                sm = difflib.SequenceMatcher(None, r1.flatten(), r2.flatten())
                if sm.ratio() < 0.35:
                    flag = 1
                    break
            if flag:
                break
        footer += row
        return header, footer

    def check_layout(self, boxes):
        """
        Sum the vertical span of bounding boxes in each layout (row or columnar).
        For column layout, majority of bounding boxes will have top left and bottom right x coordinate
        either on left or right side from the center of the page.

        """

        # Set counters
        row_span = 0.1
        col_span = 0.1

        for box in boxes:
            # Check if both sets of coordinates are on either side from the center of the page, if yes add vertical span of the box to the count
            if (box[0][0] < self.pdf.width / 2 and box[0][2] <= self.pdf.width / 2) or (
                box[0][0] > self.pdf.width / 2 and box[0][2] < self.pdf.width
            ):
                col_span += box[0][3] - box[0][1]

            else:
                row_span += box[0][3] - box[0][1]

        return row_span, col_span

    def overlap_remove(self, original_boxes, overlap_boxes, threshold=0):
        """
        Iterate over overlapping bounding boxes and calculate IOU with each bounding box in
        original boxes. If the IOU between any two bounding boxes is more than a threshold,
        remove the box from overlap_boxes. Return the updated overlap_boxes

        """

        i = 0
        while i < len(overlap_boxes):
            box1 = overlap_boxes[i]
            flag = 0
            for box2 in original_boxes:
                # Calculate the IOU
                iou = self.calculate_iou(box1[0], box2[0])
                if iou > threshold:
                    flag = 1
                    break

            if flag == 1:
                # Remove the overlapping bounding box
                overlap_boxes = [
                    x for x in overlap_boxes if not (x == overlap_boxes[i]).all()
                ]
            else:
                i += 1

        return overlap_boxes

    def list_append(self, main_list, append_list):
        # Append the append_list to the main_list:
        for lst in append_list:
            main_list.append(lst)

        return main_list

    def sort_divide(
        self, title_boxes, text_boxes, list_boxes, figure_boxes, table_boxes
    ):
        """
        Seperate the bounding boxes based on their position in the page.
        This will retain the reading flow of the pdf and each page.

        """

        # Convert numpy array to list
        text_list = [box[0] for box in text_boxes]
        title_list = [box[0] for box in title_boxes]
        list_list = [box[0] for box in list_boxes]
        figure_list = [box[0] for box in figure_boxes]
        table_list = [box[0] for box in table_boxes]

        # Append non overlapping list,figure and table bounding boxes to text boxes
        text_list, text_list, text_list = (
            self.list_append(text_list, list_list),
            self.list_append(text_list, figure_list),
            self.list_append(text_list, table_list),
        )

        # Sort on top left y coordinate
        text_boxes = sorted(text_list, key=lambda x: [x[1], x[0]])
        title_boxes = sorted(title_list, key=lambda x: [x[1], x[0]])

        # Divide the boxes w.r.t their position from the middle of the page
        # Check if the top left x coordinate is on the left or right side from the center
        text_boxes_left = [box for box in text_boxes if box[0] < self.pdf.width / 2]
        text_boxes_right = [
            box for box in text_boxes if not box[0] < self.pdf.width / 2
        ]
        title_boxes_left = [box for box in title_boxes if box[0] < self.pdf.width / 2]
        title_boxes_right = [
            box for box in title_boxes if not box[0] < self.pdf.width / 2
        ]

        return title_boxes_left, title_boxes_right, text_boxes_left, text_boxes_right

    def merge_title(self, boxes):
        """
        Merge nearby title bounding boxes.
        If the horizontal or vertical distance between two boxes is less than
        a threshold. Merge the two boxes to encompass both titles.
        """

        i = 0
        while i < len(boxes) - 1:
            if (abs(boxes[i][1] - boxes[i + 1][1]) < 5) or (
                abs(boxes[i][3] - boxes[i + 1][1]) < 5
            ):
                boxes[i] = self.merge_boxes(boxes[i], boxes[i + 1])
                # Remove the extra bounding box
                boxes = [x for x in boxes if not (x == boxes[i + 1]).all()]
            else:
                i += 1

        return boxes

    def region_recog(
        self,
        row_span,
        col_span,
        title_boxes_left,
        title_boxes_right,
        text_boxes_left,
        text_boxes_right,
        header,
        footer,
    ):
        """
        Tag each title with a text region.
        region_boxes=[[[Title],[Text]]]

        Box coordinates=[top left x,top left y,bottom right x, bottom right y]

        Row Layout: If row_span/total is greater than a threshold or if the right side (>width/2)
        is empty.

        Mixed (row+col) Layout: Check if the span of any text bounding boxes is greater
        than width/2. It will be treated as row layout.

        Column Layout: If neither of the above conditions are true. Left and right sides
        have seperate logic.
        """

        region_boxes = []
        temp_text = []
        # To keep original boxes intact
        temp_title_left = title_boxes_left
        temp_title_right = title_boxes_right

        # (Mixed Layout) Get all the text boxes that have span greater than width/2
        temp_text = [
            text_boxes_left[i]
            for i in range(len(text_boxes_left))
            if (text_boxes_left[i][2] > self.pdf.width / 2)
            and (text_boxes_left[i][2] - text_boxes_left[i][0] > self.pdf.width / 2)
        ]

        # Row/Mixed Layout:
        if (
            row_span / (row_span + col_span) > 0.8
            or (len(title_boxes_right) == 0 and len(text_boxes_right) == 0)
            or len(temp_text) > 0
        ):
            # Check if there is any title in the page
            if len(title_boxes_left) > 0:
                # Check if there is any text in the page (For pages with Figures or Tables)
                if len(text_boxes_left) > 0:
                    # Check if there is text at the top of document without any title above it
                    if text_boxes_left[0][1] < title_boxes_left[0][1]:
                        # Capture the text region. Title will be empty in this case
                        temp_text = [
                            (10, header),
                            (self.pdf.width - 10, title_boxes_left[0][1]),
                        ]
                        region_boxes.append([[], temp_text])

                # Capture text region for Figures and Tables
                else:
                    temp_text = [
                        (10, header),
                        (self.pdf.width - 10, title_boxes_left[0][1]),
                    ]
                    region_boxes.append([[], temp_text])

                # Iterate over the title boxes and capture the text region between two titles
                for i in range(len(title_boxes_left) - 1):
                    # Increase the horizontal span as padding
                    temp_title_left[i][0], temp_title_left[i][2] = (
                        10,
                        self.pdf.width - 10,
                    )
                    temp_text = [
                        (10, title_boxes_left[i][3]),
                        (self.pdf.width - 10, title_boxes_left[i + 1][1]),
                    ]
                    region_boxes.append([temp_title_left[i], temp_text])

                # For the last title in the page
                temp_title_left[-1][0], temp_title_left[-1][2] = 10, self.pdf.width - 10
                temp_text = [
                    (10, title_boxes_left[-1][3]),
                    (self.pdf.width - 10, footer),
                ]
                region_boxes.append([temp_title_left[-1], temp_text])

            # Only text in the page --> capture entire page as one text region
            else:
                temp_text = [(10, header), ((self.pdf.width) - 10, footer)]
                region_boxes.append([[], temp_text])

        # Columnar Layout
        else:
            # Get the min x coordinate of the right side
            if len(text_boxes_right) > 0:
                min_x = min(
                    [
                        text_boxes_right[i][0]
                        for i in range(len(text_boxes_right))
                        if text_boxes_right[i][0] >= self.pdf.width / 2
                    ]
                )
            else:
                min_x = (self.pdf.width / 2) + 10

            # Set the max x coordinate of the left side
            max_x = min_x - 20

            # Left side
            # Check if the page has any titles
            if len(title_boxes_left) > 0:
                # Check if there is text at the top of document without any title above it
                if text_boxes_left[0][1] < title_boxes_left[0][1]:
                    temp_text = [(10, header), (max_x, title_boxes_left[0][1])]
                    region_boxes.append([[], temp_text])

                # Iterate over the titles
                for i in range(len(title_boxes_left) - 1):
                    if (
                        title_boxes_left[i][2] < self.pdf.width / 2
                        and title_boxes_left[i][2] < max_x
                    ):
                        temp_title_left[i][0], temp_title_left[i][2] = 10, max_x
                    if title_boxes_left[i][2] < self.pdf.width / 2:
                        temp_title_left[i][0] = 10
                    temp_text = [
                        (10, title_boxes_left[i][3]),
                        (max_x, title_boxes_left[i + 1][1]),
                    ]
                    region_boxes.append([temp_title_left[i], temp_text])

                # Get the text region in the last title
                temp_title_left[-1][0], temp_title_left[-1][2] = 10, max_x
                temp_text = [(10, title_boxes_left[-1][3]), (max_x, footer - 3)]
                region_boxes.append([temp_title_left[-1], temp_text])

            # No titles --> Get the entire side as one region
            else:
                temp_text = [(10, header), (max_x, footer)]
                region_boxes.append([[], temp_text])

            # Right side

            # Check if the side has any titles
            if len(title_boxes_right) > 0:
                # Check if there is any text without title above it
                if text_boxes_right[0][1] < title_boxes_right[0][1]:
                    # If there is a centered title present --> change the top left y coordinate to avoid overlap
                    if len(title_boxes_left) > 0:
                        if title_boxes_left[0][2] > (self.pdf.width / 2) + 40:
                            temp_text = [
                                (min_x, title_boxes_left[0][3]),
                                ((self.pdf.width) - 20, title_boxes_right[0][1]),
                            ]
                        # No centered titles
                        else:
                            temp_text = [
                                (min_x, header),
                                ((self.pdf.width) - 20, title_boxes_right[0][1]),
                            ]

                    # No titles on left side
                    else:
                        temp_text = [
                            (min_x, header),
                            ((self.pdf.width) - 20, title_boxes_right[0][1]),
                        ]
                    region_boxes.append([[], temp_text])

                # Iterate over the titles
                for i in range(len(title_boxes_right) - 1):
                    if title_boxes_right[i][0] < min_x:
                        temp_title_right[i][0], temp_title_right[i][2] = (
                            self.pdf.width / 2
                        ) + 10, self.pdf.width - 20
                    else:
                        temp_title_right[i][0], temp_title_right[i][2] = (
                            min_x,
                            self.pdf.width - 20,
                        )
                    temp_text = [
                        (min_x, title_boxes_right[i][3]),
                        ((self.pdf.width) - 20, title_boxes_right[i + 1][1]),
                    ]
                    region_boxes.append([temp_title_right[i], temp_text])

                # Get the text region for the last title in the side
                temp_title_right[-1][0], temp_title_right[-1][2] = (
                    min_x,
                    self.pdf.width - 20,
                )
                temp_text = [
                    (min_x, title_boxes_right[-1][3]),
                    ((self.pdf.width) - 20, footer),
                ]
                region_boxes.append([temp_title_right[-1], temp_text])

            # No titles --> Get the entire side as one region
            else:
                temp_text = [(min_x, header), ((self.pdf.width) - 20, footer)]
                region_boxes.append([[], temp_text])

        return region_boxes

    def calculate_iou(self, bb1, bb2):
        # Get the IOU
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            iou = 0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    def overlap_region(self, region_boxes, threshold=0):
        """
        Merge the overlapping title boxes in region_boxes.
        1. Check the iou, if greater than a threshold --> merge.
        2. Check the vetrical distance, if less than a threshold --> merge
        """

        # Merge overlapping title bounding boxes
        i = 0
        while i < len(region_boxes) - 1:
            if len(region_boxes[i][0]) > 0 and len(region_boxes[i + 1][0]) > 0:
                # Calculate the iou
                iou = self.calculate_iou(region_boxes[i][0], region_boxes[i + 1][0])

                if iou > threshold:
                    # If the bounding boxes are overlapping --> merge
                    region_boxes[i + 1][0] = self.merge_boxes(
                        region_boxes[i][0], region_boxes[i + 1][0]
                    )
                    region_boxes.pop(i)
                else:
                    i += 1
            else:
                i += 1

        # Merge with minimum vertical span between them (no overlap)
        i = 0
        while i < len(region_boxes) - 1:
            if len(region_boxes[i][0]) > 0 and len(region_boxes[i + 1][0]) > 0:
                if abs(region_boxes[i][0][3] - region_boxes[i + 1][0][1]) < 10:
                    region_boxes[i + 1][0] = self.merge_boxes(
                        region_boxes[i][0], region_boxes[i + 1][0]
                    )
                    region_boxes.pop(i)
                else:
                    i += 1
            else:
                i += 1
        return region_boxes

    def ocr(self, region_boxes, image):
        title_ocr = []
        text_ocr = []
        text_coords = []
        title_coords = []

        for region in region_boxes:
            # Title Region
            if len(region[0]) > 0:
                # Crop image
                title_img = image[
                    math.floor(region[0][1] - 2.5) : math.ceil(region[0][3] + 2.5),
                    math.floor(region[0][0] - 2.5) : math.ceil(region[0][2] + 2.5),
                ]
                if len(title_img) > 0:
                    title_temp = pytesseract.image_to_string(
                        title_img, config=get_pytessaract_custom_config()
                    )
                    y1 = math.floor(region[0][1] - 2.5)
                    y2 = math.ceil(region[0][3] + 2.5)
                    x1 = math.floor(region[0][0] - 2.5)
                    x2 = math.ceil(region[0][2] + 2.5)
                    ttl_coords = (x1, y1, x2, y2)
                else:
                    title_temp = []
                    ttl_coords = ()
            # Empty title region
            else:
                title_temp = []
                ttl_coords = ()

            # Text Region
            if len(region[1]) > 0:
                # Crop image
                text_img = image[
                    math.floor(region[1][0][1]) : math.ceil(region[1][1][1]),
                    math.floor(region[1][0][0] - 2.5) : math.ceil(
                        region[1][1][0] + 2.5
                    ),
                ]
                if len(text_img) > 0:
                    txt_temp = pytesseract.image_to_string(
                        text_img, config=get_pytessaract_custom_config()
                    )
                    y1 = math.floor(region[1][0][1])
                    y2 = math.ceil(region[1][1][1])
                    x1 = math.floor(region[1][0][0] - 2.5)
                    x2 = math.ceil(region[1][1][0] + 2.5)
                    txt_coords = (x1, y1, x2, y2)
                else:
                    txt_temp = []
                    txt_coords = ()
            # Empty text region
            else:
                txt_temp = []
                txt_coords = ()

            title_ocr.append(title_temp)
            text_ocr.append(txt_temp)
            title_coords.append(ttl_coords)
            text_coords.append(txt_coords)

        return title_ocr, text_ocr, title_coords, text_coords

    def visualise_post_process(self, img_file, region_boxes):
        image = Image.open(os.path.join(self.pdf.folder_path, img_file))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for region in region_boxes:
            if len(region[0]) > 0:
                draw.rectangle(region[0], outline="red")
                draw.text(
                    (region[0][0] + 10, region[0][1] - 10), text="Title", fill="red"
                )
            draw.rectangle(region[1], outline="green")
        os.makedirs(
            os.path.join(self.pdf.file_path, "vis-post", self.pdf.pdf_name),
            exist_ok=True,
        )
        image.save(
            os.path.join(self.pdf.file_path, "vis-post", self.pdf.pdf_name, img_file)
        )

    def visualise_model(self, img_file, output):
        boxes = [output[i].pred_boxes.tensor.cpu().numpy() for i in range(len(output))]
        labels = [output[i].pred_classes.cpu().numpy() for i in range(len(output))]
        classes = ["text", "title", "list", "table", "figure"]
        image = Image.open(os.path.join(self.pdf.folder_path, img_file))
        draw = ImageDraw.Draw(image)
        fontsize = 10

        for idx, box in enumerate(boxes):
            predicted_label = classes[labels[idx][0]]
            draw.rectangle(box, outline="red")
            draw.text(
                (box[0][0] + 10, box[0][1] - 10), text=predicted_label, fill="green"
            )

        os.makedirs(
            os.path.join(self.pdf.file_path, "vis", self.pdf.pdf_name), exist_ok=True
        )
        image.save(os.path.join(self.pdf.file_path, "vis", self.pdf.pdf_name, img_file))

    def is_camel_case(self, s):
        return s != s.lower() and s != s.upper()

    def df_operations(self, df):
        if df.loc[0, "Title"] == []:
            df.loc[0, "Title"] = self.pdf.pdf_name
        i = 1
        while i < len(df):
            if df.loc[i, "Title"] == []:
                df.loc[i, "Title"] = df.loc[i - 1, "Title"]
            i += 1

        upper_count, lower_count, camel_count = 0, 0, 0
        for title in df["Title"]:
            if self.is_camel_case(title):
                camel_count += 1
            if title.isupper():
                upper_count += 1
            if title.islower():
                lower_count += 1

        # Remove text tagged as title
        """        # Capital Title
        if upper_count / (lower_count + upper_count + camel_count) >= 0.7:
            i = 1
            while i < len(df):
                if not df.iloc[i]["Title"].isupper() and len(df.iloc[i]["Title"]):
                    df.iloc[i - 1]["Content"] += (
                        df.iloc[i]["Title"] + df.iloc[i]["Content"]
                    )
                    df = df.drop(df.index[i])
                else:
                    i += 1"""

        # Camel case title
        """if camel_count / (upper_count + lower_count + camel_count) > 0.8:
            i = 1
            while i < len(df):
                if len(df.iloc[i]["Title"].strip()) > 70:
                    df.iloc[i - 1]["Content"] += (
                        df.iloc[i]["Title"] + df.iloc[i]["Content"]
                    )
                    df = df.drop(df.index[i])
                else:
                    i += 1"""
        # Remove stop words with title tags
        i = 1
        while i < len(df):
            word_tokens = word_tokenize(df.iloc[i]["Title"])
            filtered_sentence = [w.lower() for w in word_tokens if w.isalpha()]
            res = [w for w in filtered_sentence if w not in stop_words]
            if len(res) == 0:
                df.iloc[i - 1]["Content"] += df.iloc[i]["Title"] + df.iloc[i]["Content"]
                df = df.drop(df.index[i])
            else:
                i += 1

        # Save the results as json file
        new_obj = []
        df = df.reset_index(drop=True)
        for i in range(df.shape[0]):
            cur_page_num = int(df.loc[i, "PageNo"])
            img_name = sorted(os.listdir(self.pdf.folder_path))[cur_page_num - 1]
            img = cv2.imread(os.path.join(self.pdf.folder_path, img_name), 0)
            if i == 0:
                prev_page_num = int(df.loc[i, "PageNo"])
                content = [f"Title: {df.loc[i, 'Title'].strip()}"]
                metadata = {
                    "Content": [f"Title: {df.loc[i, 'Title'].strip()}"],
                    "Coords": [df.loc[i, "TitleCoords"]],
                }
                if isinstance(df.loc[i, "Content"], list):
                    content.extend(df.loc[i, "Content"])
                    metadata["Content"].extend(df.loc[i, "Content"])
                    metadata["Coords"].extend(df.loc[i, "ContentCoords"])
                else:
                    content.extend(
                        list(filter(text_filteration, df.loc[i, "Content"].split("\n")))
                    )
                    metadata["Content"].append(df.loc[i, "Content"])
                    metadata["Coords"].append(df.loc[i, "ContentCoords"])

                obj = {"Content": content, "PageNo": cur_page_num, "Metadata": metadata}
                new_obj.append(obj)
                continue
            if cur_page_num == prev_page_num:
                new_obj[-1]["Content"].append(f"Title: {df.loc[i, 'Title'].strip()}")
                new_obj[-1]["Metadata"]["Content"].append(
                    f"Title: {df.loc[i, 'Title'].strip()}"
                )
                new_obj[-1]["Metadata"]["Coords"].append(df.loc[i, "TitleCoords"])
                if isinstance(df.loc[i, "Content"], list):
                    new_obj[-1]["Content"].extend(df.loc[i, "Content"])
                    new_obj[-1]["Metadata"]["Content"].extend(df.loc[i, "Content"])
                    new_obj[-1]["Metadata"]["Coords"].extend(df.loc[i, "ContentCoords"])
                else:
                    new_obj[-1]["Content"].extend(
                        list(filter(text_filteration, df.loc[i, "Content"].split("\n")))
                    )
                    new_obj[-1]["Metadata"]["Content"].append(df.loc[i, "Content"])
                    new_obj[-1]["Metadata"]["Coords"].append(df.loc[i, "ContentCoords"])
            else:
                content = [f"Title: {df.loc[i, 'Title'].strip()}"]
                metadata = {
                    "Content": [f"Title: {df.loc[i, 'Title'].strip()}"],
                    "Coords": [df.loc[i, "TitleCoords"]],
                }
                if isinstance(df.loc[i, "Content"], list):
                    content.extend(df.loc[i, "Content"])
                    metadata["Content"].extend(df.loc[i, "Content"])
                    metadata["Coords"].extend(df.loc[i, "ContentCoords"])
                else:
                    content.extend(
                        list(filter(text_filteration, df.loc[i, "Content"].split("\n")))
                    )
                    metadata["Content"].append(df.loc[i, "Content"])
                    metadata["Coords"].append(df.loc[i, "ContentCoords"])
                obj = {"Content": content, "PageNo": cur_page_num, "Metadata": metadata}
                new_obj.append(obj)
            prev_page_num = cur_page_num

        return new_obj
    
    def run_v1(self, ):
        """
        Post Processing
        """

        # Convert the pdf to image and get the sorted order of images
        sorted_files = self.pdf.pdf_to_image()
        print(sorted_files)
        scanned_pdf = False
        diff_dims = False
        
        first_image = cv2.imread(os.path.join(self.pdf.folder_path, sorted_files[0]))
        first_image_shape = first_image.shape
        for imgfile in sorted_files:
            image = cv2.imread(
                os.path.join(self.pdf.file_path, self.pdf.pdf_name, imgfile)
            )
            if not first_image.shape == image.shape:
                diff_dims = True
        
        self.header = [10] * len(sorted_files)
        self.footer = [self.pdf.height - 1] * len(sorted_files)
        # Check if the pdf is scanned or readable
        if self.pdf.is_readable_pdf(text_ratio_thres=0.0001) and diff_dims == False:
            temp_header_1, temp_footer_1 = self.cal_header_footer(sorted_files)
            print("pdf is readable")
            for idx in range(len(sorted_files)):
                self.header[idx], self.footer[idx] = temp_header_1, temp_footer_1

        elif self.pdf.is_readable_pdf(text_ratio_thres=0.0001) and diff_dims == True:
            print("pdf is mix of both")
            for idx, imgfile in enumerate(sorted_files):
                image = cv2.imread(
                    os.path.join(self.pdf.file_path, self.pdf.pdf_name, imgfile)
                )
                self.header[idx], self.footer[idx] = 10, image.shape[0] - 10

        else:
            print("pdf is scanned")
            scanned_pdf = True
            self.header = [10] * len(sorted_files)
            self.footer = [self.pdf.height - 1] * len(sorted_files)

        result = []
        # Iterate over the images
        for idx, imgfile in enumerate(sorted_files):
            image = cv2.imread(
                os.path.join(self.pdf.file_path, self.pdf.pdf_name, imgfile)
            )
            output = self.model_results(image)
            df = pd.DataFrame(
                columns=["Title", "Content", "TitleCoords", "ContentCoords", "PageNo"]
            )
            self.pdf.width = image.shape[1]
            self.pdf.height = image.shape[0]
            # Seperate text and title boxes and convert tensor bounding box to numpy array
            boxes = [
                output[i].pred_boxes.tensor.cpu().numpy() for i in range(len(output))
            ]
            text_boxes = [
                output[i].pred_boxes.tensor.cpu().numpy()
                for i in range(len(output))
                if output[i].pred_classes == 0
            ]
            title_boxes = [
                output[i].pred_boxes.tensor.cpu().numpy()
                for i in range(len(output))
                if output[i].pred_classes == 1
            ]
            list_boxes = [
                output[i].pred_boxes.tensor.cpu().numpy()
                for i in range(len(output))
                if output[i].pred_classes == 2
            ]
            table_boxes = [
                output[i].pred_boxes.tensor.cpu().numpy()
                for i in range(len(output))
                if output[i].pred_classes == 3
            ]
            figure_boxes = [
                output[i].pred_boxes.tensor.cpu().numpy()
                for i in range(len(output))
                if output[i].pred_classes == 4
            ]

            # Check columnar
            row_count, col_count = self.check_layout(boxes)
            # Remove overlap
            list_boxes, list_boxes = self.overlap_remove(
                title_boxes, list_boxes, 0.09
            ), self.overlap_remove(text_boxes, list_boxes, 0.09)
            title_boxes, text_boxes, list_boxes = (
                self.overlap_remove(table_boxes, title_boxes),
                self.overlap_remove(table_boxes, text_boxes),
                self.overlap_remove(table_boxes, list_boxes),
            )
            title_boxes, text_boxes, list_boxes = (
                self.overlap_remove(figure_boxes, title_boxes),
                self.overlap_remove(figure_boxes, text_boxes),
                self.overlap_remove(figure_boxes, list_boxes),
            )

            # Sort and divide the boxes
            (
                title_boxes_left,
                title_boxes_right,
                text_boxes_left,
                text_boxes_right,
            ) = self.sort_divide(
                title_boxes, text_boxes, list_boxes, figure_boxes, table_boxes
            )
            # merge titles
            title_boxes_left, title_boxes_right = self.merge_title(
                title_boxes_left
            ), self.merge_title(title_boxes_right)

            # Get box regions
            region_boxes = self.region_recog(
                row_count,
                col_count,
                title_boxes_left,
                title_boxes_right,
                text_boxes_left,
                text_boxes_right,
                self.header[idx],
                self.footer[idx],
            )

            # title overlap
            region_boxes = self.overlap_region(region_boxes, 0.05)

            # OCR
            title_ocr, text_ocr, title_coords, text_coords = self.ocr(
                region_boxes, image
            )

            # Append to df
            df = pd.DataFrame(
                data=zip(
                    title_ocr,
                    #pdf_file_path,
                    text_ocr,
                    title_coords,
                    text_coords,
                    [idx + 1] * len(title_ocr),
                ),
                columns=df.columns,
            )
            #modified for 
            result.append(self.df_operations(df)[0])
        result = pd.DataFrame(result)
        source=self.pdf.pdf_source()

        # content_list = []
        # meta_list = []
        # 
        # for metadata, page_no in zip(result['Metadata'], result['PageNo']):
        #     # Extract the 'Content' key from the dictionary
        #     content = metadata['Content']
            
        #     # Join the list of text segments into a single string
        #     content_text = ' '.join(content)
            
        #     coords=str(metadata['Coords'])
        #     # Create the 'meta' dictionary including the file path
        #     meta_dict = {
        #         'source':source,
        #         'page': page_no,
        #         'file_name':f'{self.pdf.pdf_name}.pdf',
        #         'coordinates':coords ,
        #     }
            
        #     # Append the results to the content_list and meta_list
        #     content_list.append(content_text)
        #     meta_list.append(meta_dict)


        content_list = []
        coords_list = []
        meta_list=[]

        for metadata, page_no in zip(result['Metadata'], result['PageNo']):
            # Extract the 'Content' key from the dictionary
            content_text = metadata['Content']
        
            coords=metadata["Coords"]
            if len(content_text)%2 ==0:
                for i in range(0, len(content_text), 2):
                    joined_element = content_text[i] + content_text[i + 1]
                    
                    joined_coords=coords[i],coords[i+1]
                    final_coords=str(joined_coords)
                    meta_dict = {
                            'source':source,
                            'page': page_no,
                            'file_name':f'{self.pdf.pdf_name}.pdf',
                            'coordinates':final_coords ,
                        }
                        
                        # Append the results to the content_list and meta_list
                    content_list.append(joined_element)
                    meta_list.append(meta_dict)
            else:
                for i in range(len(content_text)):
                    joined_element = content_text[i]
                    
                    # Check if there is a corresponding coordinate
                    if i < len(coords):
                        joined_coords = coords[i]
                    else:
                        joined_coords = () 
                    # Create the 'meta' dictionary
                    final_coords=str(joined_coords)

                    meta_dict = {
                        'source': source,
                        'page': page_no,
                        'file_name': f'{self.pdf.pdf_name}.pdf',
                        'coordinates': final_coords,
                    }

                # Append the results to the content_list and meta_list
                    content_list.append(joined_element)
                    meta_list.append(meta_dict)


        # Create a new DataFrame with the extracted content and 'meta' dictionaries
        new_df = pd.DataFrame({'page_content': content_list, 'metadata': meta_list})
        new_df['page_content'] = new_df['page_content'].str.replace('Title: ', ' ')
        document_list = []
        for _, row in new_df.iterrows():
            page_content = row['page_content']
            metadata = row['metadata']
            document_instance = Document(page_content=page_content, metadata=metadata)
            document_list.append(document_instance)

        return document_list
    

"""# Instantiate the PDF and ImageExtractor classes


pdf_file_path = "./unreadable contracts/The Bridge Club.pdf"


# Instantiate the PDF and ImageExtractor classes
pdf_instance = PDF(pdf_file_path)
image_extractor = ImageExtractor(pdf_instance=pdf_instance)

# Access the pdf_file_path attribute using the method
result = image_extractor.run_v1()

print(result)"""