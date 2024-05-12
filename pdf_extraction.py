from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
import pandas as pd


def extract_text_from_image(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config,lang="eng")
    return text

def extraction_check(img_file):
    img = cv2.imread(img_file)
    #  img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=2, fy=2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.bitwise_not(img)
    #coords = np.column_stack(np.where(img > 0))
 
    #kernel = np.ones((1,1), np.uint8)
   
    #for psm in range(6,13+1):
    for psm in range(6,7):
        config = '--oem 3 --psm %d' % psm
        txt = pytesseract.image_to_string(img, config = config, lang='eng')
        print('psm ', psm, ':',txt)


def extract_text_from_pages(pages):
    # Create a list to store extracted text from all pages
    extracted_text = []

    for index,page in enumerate(pages):
        img_file = f"output\img_{index}.jpg"
        page.save(img_file,"JPEG")
        extraction_check(img_file)
        # Step 2: Preprocess the image (deskew)
       
        # Step 3: Extract text using OCR
        #text = extract_text_from_image(preprocessed_image)
    #return extracted_text

def main():
    # Replace 'input_file.pdf' with the path to your PDF file
    pdf_file = 'sample-files/DE-ID DOC026.pdf'
    pages = convert_from_path(pdf_file)
    extract_text_from_pages(pages)
    #text_per_page = extract_text_from_pages(pages)
    #print(text_per_page)



if __name__ == "__main__":
    main()