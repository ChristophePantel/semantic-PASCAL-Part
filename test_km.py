
import km 
import torch 
import torch.nn as nn

def encode_variants(class_number, class_variants):
    result = torch.zeros((len(class_variants),class_number))
    for variant_index in class_variants:
        for class_index in class_variants[variant_index]:
            result[variant_index,class_index] = 1.0
    return result

def generate_distance_matrix(class_number, class_variants, distance):
    # calcule la distance de chaque variant à chaque variant en utilisant la fuzzy logic ou la bce
    encoded_variants = encode_variants(class_number, class_variants)
    simulated_variants = 0.2 * encoded_variants + 0.4
    result = distance(simulated_variants, encoded_variants)
    return result 

def scores_bce(batch_scores, prediction_scores):
    """Compute binary cross entropy between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    bce_calculator = nn.BCELoss(reduction="none")
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    batch_scores_sum = torch.sum(batch_scores,1)
    aligned_batch_scores_sum = torch.unsqueeze(batch_scores_sum, 0).expand(prediction_range,-1)
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    bce_per_class = bce_calculator(aligned_prediction_scores,aligned_batch_scores) 
    bce = torch.sum(bce_per_class,2) / batch_scores_sum
    # detected = torch.where(bce < 1)
    return bce

def scores_fuzzy_equiv(batch_scores, prediction_scores, alpha=0.5, power=3):
    """Compute fuzzy equivalence between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    negated_aligned_batch_scores = 1.0 - aligned_batch_scores
    negated_aligned_prediction_scores = 1.0 - aligned_prediction_scores
    positive_component = aligned_batch_scores * aligned_prediction_scores
    negative_component = negated_aligned_batch_scores * negated_aligned_prediction_scores
    # Version Mean of positive contribution (bad according to CP/IRIT)
    # positive_component_sum = positive_component.sum(-1)
    # aligned_batch_scores_sum = aligned_batch_scores.sum(-1)
    # batch_prediction_equiv_mean = positive_component_sum  / aligned_batch_scores_sum 
    # Version Balanced Mean of positive and negative contribution
    batch_prediction_equiv = (alpha *  positive_component + (1 - alpha) * negative_component) / alpha
    # Version enhanced power mean
    batch_prediction_equiv_mean = batch_prediction_equiv.pow(power).mean(-1).pow(1/power)
    # Version linear mean
    # batch_prediction_equiv_mean = batch_prediction_equiv.mean(-1)
    return batch_prediction_equiv_mean

classes = km.get_class_names()
abstracts = km.get_abstract_classes()
part_hierarchy = km.get_contained_classes()
class_hierarchy = km.get_refined_classes()

others = []
extended_class_hierarchy = class_hierarchy
for abstract in abstracts:
    other = 'Other_' + abstract
    others.append(other)
    extended_class_hierarchy[other] = frozenset({abstract})
extended_classes = classes.union(frozenset(others))
class_number = len(extended_classes)
#class_name_adapter = get_SemanticPascalPart_to_Yolo_class_name_adapter()
code_to_class, class_to_code = km.associate_number_to_class_and_class_to_number(sorted(extended_classes))
#adapted_class_to_code = { name : class_to_code[class_name_adapter[name]] 
#                                        for name in class_name_adapter if class_name_adapter[name] in class_to_code }
coded_part_hierarchy = km.associate_codes_to_hierarchies(class_to_code, part_hierarchy)
coded_class_hierarchy = km.associate_codes_to_hierarchies(class_to_code, extended_class_hierarchy)
coded_class_hierarchy_inverted = km.invert_relation(coded_class_hierarchy)

class_codes = frozenset(code_to_class.keys())
abstract_codes = km.class_names_to_codes( abstracts, class_to_code)
full_composition = km.resolve(class_codes,coded_part_hierarchy,coded_class_hierarchy)
inverted_full_composition = km.invert_relation(full_composition)
# Build Composition variants
class_variants, variant_to_class = km.variants(class_codes,abstract_codes,inverted_full_composition)
# Add Inheritance relation
generalized_class_variants = km.generalize(class_codes,class_variants,coded_class_hierarchy)
# variant_names = ...
encoded_class_variants = km.encode_variants(class_number, generalized_class_variants)

result = generate_distance_matrix(class_number, class_variants, lambda a, b : scores_fuzzy_equiv( a, b, 0.9, 3 ))
print(result)

result = generate_distance_matrix(class_number, class_variants, scores_bce)
print(result)