# Annotation Protocol

## YOLO Bounding Box Annotation
1. Use LabelImg or Roboflow for bounding box annotation
2. Classes: figure (0), table (1), equation (2), chart (3), micrograph (4)
3. Include all visible figures/tables on each PDF page
4. Export in YOLO format (class x_center y_center width height, normalized)

## Microstructure Classification
1. Extract figure crops using YOLO bounding boxes
2. Binary classification: determine whether the image contains a microstructure (label 0: microstructure) or not (label 1: other)
3. Each image labeled by at least 2 annotators
4. Disagreements resolved by domain expert

## Semantic Extraction Annotation
1. Read full paper text
2. Fill in the JSON schema: composition, processes, microstructures, properties
3. Mark causal relations with trigger words and confidence
4. Cross-validate with at least 1 other annotator
