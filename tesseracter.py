import pytesseract

# image = 'tesseract_test.jfif'
image = '2020-01-18 (3).png'
# tesseract_test.jfif as a_image

# print(dir(pytesseract))

print(pytesseract.image_to_string(image))
# print(pytesseract.image_to_pdf_or_hocr(image))
print('Vei')