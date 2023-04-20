# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:49:20 2023

@author: yaobv
"""

import PyPDF2

file_path = 'data/grabby_aliens.pdf'
   
if __name__ == '__main__':
    # Open the PDF file in read-binary mode
    with open(file_path, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)

        # Get the number of pages in the PDF document
        num_pages = pdf_reader.numPages
        
        # Create an empty string variable to store the text
        text = []
        
        # Loop through each page and extract the text
        for i in range(num_pages):
            page = pdf_reader.getPage(i)
            page_text = page.extractText()
            page_text = page_text.replace('\n', ' ')
            print(len(page_text), type(page_text))
            text.append(page_text + '\n\n')
             
        text = '\n\n\n'.join(text)
    # Save the extracted text to a text file
    with open('data/grabby_aliens.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(text)