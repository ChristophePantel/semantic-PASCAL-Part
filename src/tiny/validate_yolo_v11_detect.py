# export PYTHONPATH=/chemin_vers_github_ultralytics_branche_hierarchical
from ultralytics import YOLO

path = "/data/christophe/hierarchical/semantic-PASCAL-Part/src/tiny/"

# Load a pretrained YOLO model (recommended for training)
model = YOLO(path + "results/augmented/train_with_scores_metrics_0/train/weights/best.pt")

metrics_threshold = 0

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.val(
    data=path + "SemanticPascalPart.yaml", 
    save_txt=True,
    save_conf=True,
    save_json=True,
    visualize=True,
    use_scores=True, # (bool) use class scores instead of class labels
    use_km=True, # (bool) use knowledge model if available
    use_km_scores=True, # (bool) use knowledge model scores if available
    use_km_metrics=True, # (bool) use knowledge model metrics -- class compatibility matrix -- if available
    km_metrics_threshold=metrics_threshold, # (int) number of knowledge model metrics produced -- thresholf for the class compatibility metrics
    use_variant_selection=True, # (bool) select main class based on knowledge model scores if available
    use_km_losses=False, # (bool) use knowledge model losses if available
    use_refinement=False, # (bool) use refinement relation if available
    km_specialization_weight=1.0, # (float) weight of the specialization loss
    km_specialization_exclusion_weight=1.0, # (float) weight of the specialization exclusion loss
    km_generalization_weight=1.0, # (float) weight of the generalization loss
    use_composition=False, # (bool) use composition relation if available
    km_composition_weight=1.0, # (float) weight of the composition loss
    km_composition_exclusion_weight=1.0, # (float) weight of the composition exclusion loss
    km_decomposition_weight=1.0, # (float) weight of the decomposition loss
    km_decomposition_exclusion_weight=1.0, # (float) weight of the decomposition exclusion loss
    project= path + "results/augmented/validate_with_scores_metrics_"+ str(metrics_threshold),
    name="val",
    )

pass