Road Segmentation using U-Net

A deep learning project that performs semantic segmentation of roads from satellite/aerial imagery using a U-Net convolutional neural network, built and trained in Google Colab with TensorFlow/Keras.

📁 Project Structure

.
└── road_segmentation_.ipynb   # End-to-end notebook: data loading, model, training, evaluation

 Overview

Given an aerial/satellite image, the model predicts a binary mask identifying road pixels vs. non-road pixels. The architecture is a classic U-Net (encoder–decoder with skip connections), implemented from scratch in Keras.

 Pipeline (Notebook Walkthrough)


Mount Google Drive — dataset is read directly from Google Drive in Colab.
GPU Check — verifies TensorFlow can see an available GPU.
Data Loading & Preprocessing

Reads paired image/mask .tiff / .tif files from:

train/, train_labels/
val/, val_labels/
test/, test_labels/



Resizes all images and masks to 256×256
Normalizes images to [0, 1]
Binarizes masks with a > 0.5 threshold
Supports loading a fraction of the dataset (use_fraction) for faster experimentation



Model: U-Net

Encoder: 2 downsampling blocks (Conv2D ×2 + MaxPooling), filters 64 → 128
Bottleneck: Conv2D ×2 with 256 filters
Decoder: 2 upsampling blocks (UpSampling2D + skip-connection concat + Conv2D ×2), filters 128 → 64
Output: 1×1 Conv2D with sigmoid activation (binary mask)
Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy



Training

model.fit() for 10 epochs, batch size 8
Validated on the validation split



Evaluation & Visualization

Plots training/validation accuracy curves
Runs predictions on validation images
Displays side-by-side comparisons of input image, ground-truth mask, and predicted mask



Model Saving

Trained model saved to Google Drive as unet_model.h5





Dataset Format

The notebook expects a directory structure (on Google Drive) like:

MyDrive/
├── train/             # training images (.tiff)
├── train_labels/      # training masks (.tif)
├── val/                # validation images
├── val_labels/         # validation masks
├── test/               # test images
└── test_labels/        # test masks

Each image must have a matching mask with the same base filename (e.g. image_01.tiff ↔ image_01.tif).

 Getting Started

Requirements

bashpip install tensorflow opencv-python numpy matplotlib

Run


Open road_segmentation_.ipynb in Google Colab (or Jupyter, removing the Drive-mount cell if running locally).
Update train_image_dir, train_mask_dir, etc. to point to your dataset location.
Run all cells in order.


Load the Trained Model

pythonfrom tensorflow.keras.models import load_model

model = load_model('unet_model.h5')
predicted_mask = model.predict(input_image)

Output


Training/validation accuracy plot
Visual comparisons: Input Image → Ground Truth Mask → Predicted Mask


TODO / Possible Improvements


 Add quantitative segmentation metrics (IoU, Dice coefficient, F1-score) beyond accuracy
 Add data augmentation for better generalization
 Parameterize paths via config/CLI instead of hardcoded Drive paths
 Add requirements.txt
 Use a segmentation-specific loss (e.g. Dice loss) instead of plain accuracy/BCE
 Evaluate on the held-out test set (currently loaded but not used for final evaluation
