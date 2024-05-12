from PIL import Image
import cv2
import pytesseract
import os
import numpy as np
import pandas as pd
import re
from pdf2image import convert_from_bytes

'''
Main part of OCR:
pages_df: save eextracted text for each pdf file, index by page
OCR_dic : dict for saving df of each pdf, filename is the key
'''


OCR_list=[]

# Some help functions 
def get_conf(page_gray):
    '''return a average confidence value of OCR result '''
    df = pytesseract.image_to_data(page_gray,output_type='data.frame')
    df.drop(df[df.conf==-1].index.values,inplace=True)
    df.reset_index()
    return df.conf.mean()

def deskew(image):
    '''deskew the image'''
    gray = cv2.bitwise_not(image)
    temp_arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(temp_arr > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

def process_files(file_list):
    for index,file in enumerate(file_list):
        # convert pdf into image
        pdf_file = convert_from_bytes(open(file, 'rb').read())
        # create a df to save each pdf's text
        pages_df = pd.DataFrame(columns=['conf','text'])
        for (i,page) in enumerate(pdf_file) :
            try:
                # transfer image of pdf_file into array
                page_arr = np.asarray(page)
                # transfer into grayscale
                page_arr_gray = cv2.cvtColor(page_arr,cv2.COLOR_BGR2GRAY)
                #page_arr_gray = cv2.fastNlMeansDenoising(page_arr_gray,None,3,7,21)
                page_deskew = deskew(page_arr_gray)
                # cal confidence value
                page_conf = get_conf(page_deskew)
                # extract string 
                d = pytesseract.image_to_data(page_deskew,output_type=pytesseract.Output.DICT)
                #print(f"{i} image to data  {d}")
                d_df = pd.DataFrame.from_dict(d)
                
                # get block number
                block_num = int(d_df.loc[d_df['level']==2,['block_num']].max())
                # drop header and footer by index
                #head_index = d_df[d_df['block_num']==1].index.values
                foot_index = d_df[d_df['block_num']==block_num].index.values
                #d_df.drop(head_index,inplace=True)
                d_df.drop(foot_index,inplace=True)
                #print(d_df.head(20))
                # combine text in dataframe
                text = combine_texts(d_df.loc[(d_df['level']==5) & (d_df['conf'] > 50) ,'text'].values)
                #print(f" text extracted {text}")
                #pages_df = pages_df.append({'conf': page_conf,'text': text}, ignore_index=True)
                pages_df = pd.concat([pages_df,pd.DataFrame([{'conf': page_conf,'text': text}])], ignore_index=True)
                #print("now pages_df is ",pages_df)
            except Exception as e:
                # if can't extract then give some notes into df
                if hasattr(e,'message'):
                   # pages_df = pages_df.append({'conf': -1,'text': e.message}, ignore_index=True)
                   pages_df = pd.concat([pages_df,pd.DataFrame([{'conf': -1,'text': e.message}])],ignore_index=True)
                else:
                    #pages_df = pages_df.append({'conf': -1,'text': e}, ignore_index=True)
                    pages_df = pd.concat([pages_df,pd.DataFrame([{'conf': -1,'text': e}])],ignore_index=True)
                continue
        # save df into a dict with filename as key 
        print("Now pages df contains ",pages_df.head(10))  
        pages_df = pages_df[pages_df['conf']>50]     
        OCR_list.append(pages_df)
        print('{} is done'.format(file))
    return OCR_list

def write_to_file(df,outfile):
   np.savetxt(
        f'{outfile}',
        df['text'].values,
        fmt='%s'
    )
   
def main():
    # Replace 'input_file.pdf' with the path to your PDF file
    pdf_file = 'sample-files/DE-ID DOC019.pdf'
    out_file = 'sample-files/DE-ID DOC019.txt'
    file_list = []
    file_list.append(pdf_file)
    Ocr_list = process_files(file_list)
    # pd.set_option('display.max_rows',100)
    # pd.set_option('display.max_columns',100)
    #pd.set_option('display.max_colwidth', None)
    #print(Ocr_dic[file_list[0]]['text'])
    final_df = Ocr_list[0]
    #print("final_df",final_df)
    print("---------------------------------------------------")
    print(final_df.head(10))
    write_to_file(final_df,out_file)
    print("---------------------------------------------------")



if __name__ == "__main__":
    main()