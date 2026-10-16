PhytoSense leaf disease CNN - how it was trained (13 Oct 2026)

Dataset : PlantVillage (github.com/spMohanty/PlantVillage-Dataset, raw/color), 38 classes
          = 14 crops x healthy + common diseases. Balanced subset: up to 450 photos per class,
          16,354 photos in total. 85 % training (13,920), 15 % test (2,434), split per class.
Model   : 5 conv blocks (32-64-128-128-192 filters, 3x3, BatchNorm, ReLU, 2x2 max-pool),
          global average pooling, dropout 0.3, dense 38. Input 96 x 96 RGB.
Training: TensorFlow/Keras, Adam with cosine decay (2e-3), 24 epochs, batch 64, augmentation
          (flips, rotation, zoom, brightness, contrast). CPU, about 53 minutes.
Result  : 97.0 % top-1 and 99.8 % top-3 accuracy on the 2,434 test photos
          (see ../../models/leaf_cnn_report.json for per-class accuracy).
Export  : export.py folds BatchNorm into the conv weights and saves models/leaf_cnn.npz,
          which leaf_model.py runs with NumPy only (about 20 ms per photo).
Limits  : only these 14 crops; trained on lab-style photos of single leaves, so real field
          photos are less accurate. In PhytoSense the AI vision model decides whether the
          plant is one of these crops before using the CNN result.

To retrain: put the dataset at ../pv/raw/color, then  python prep.py ; python train.py ; python export.py
(needs: pip install tensorflow pillow)
