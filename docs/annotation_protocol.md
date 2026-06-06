# Annotation Protocol

## YOLO Bounding Box Annotation
1. Use LabelImg or Roboflow for bounding box annotation
2. Classes: image (0), caption (1), subgraph (2), subgraph_label (3)
3. Include all visible figures and associated elements on each PDF page
4. Export in YOLO format (class x_center y_center width height, normalized)

## Microstructure Classification
1. Extract figure crops using YOLO bounding boxes
2. Binary classification: determine whether the image contains a microstructure (label 0: microstructure) or not (label 1: other)
3. Annotations are pre-annotated using DeepSeek-assisted labeling, followed by manual verification

## Semantic Extraction Annotation
1. Read full paper text
2. Fill in the JSON schema: composition, processes, microstructures, properties
3. Mark causal relations with trigger words and confidence
4. DeepSeek-assisted pre-annotation followed by manual verification
