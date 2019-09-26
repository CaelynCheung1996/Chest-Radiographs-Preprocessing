# Chest-Radiograph-Pre-processing
### Pre-processing pipeline for ICU chest radiographs for further machine learning and deep learning related applications
The pre-processing pipeline includes:
- Otsu thresholding binarization
- Select adaptive bounding box to include the main part of the chest radiographs
- Resize image to 2048 * 2048
- Apply Sauvola filter to recognize the annotation label on the image and blur the text with median filter
- Contrast-limited adaptive histogram equalization
- Anisotropic denoising


