# export PYTHONPATH=/chemin_vers_github_ultralytics_branche_hierarchical
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/Users/marc/Documents/Eclipse-Work-Space/liclipse-2025/datasets/SemanticPascalPart/tiny/SemanticPascalPart.yaml", epochs=1000, 
# results = model.train(data="coco8.yaml", epochs=3, 
    device="mps",
    use_scores=True, # (bool) use class scores instead of class labels
    use_km=True, # (bool) use knowledge model if available
    use_km_scores=True, # (bool) use knowledge model scores if available
    use_variant_selection=True, # (bool) select main class based on knowledge model scores if available
    use_km_losses=True, # (bool) use knowledge model losses if available
    km=100,
    use_refinement=True, # (bool) use refinement relation if available
    km_specialization_weight=1.0, # (float) weight of the specialization loss
    km_specialization_exclusion_weight=1.0, # (float) weight of the specialization exclusion loss
    km_generalization_weight=1.0, # (float) weight of the generalization loss
    use_composition=False, # (bool) use composition relation if available
    km_composition_weight=1.0, # (float) weight of the composition loss
    km_composition_exclusion_weight=1.0, # (float) weight of the composition exclusion loss
    km_decomposition_weight=1.0, # (float) weight of the decomposition loss
    km_decomposition_exclusion_weight=1.0, # (float) weight of the decomposition exclusion loss
    project="results/",
    name="train",
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=0.0,
    erasing=0.0,
    auto_augment=None,
    )

# Evaluate the model's performance on the validation set
results = model.val(
    project="results/", 
    name="val",
    )

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")