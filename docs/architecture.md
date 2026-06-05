# Batch Literature Processing - Function Architecture Design

## 1. System Architecture Overview

### 1.1 Current System Architecture

```
PDF -> YOLO Detection -> Semantic Extraction -> Knowledge Graph / Causal Hypothesis Graph
```

### 1.2 Extended Architecture

```
Batch PDF Upload -> Processing Queue -> Parallel Processing -> Data Storage -> Data Integration -> Knowledge Graph / Causal Hypothesis Graph
```

## 2. Core Component Design

### 2.1 Batch Processing Components

#### 2.1.1 Batch Upload Interface
- Supports multi-file upload
- Supports folder upload
- Upload file validation (PDF format check)

#### 2.1.2 Processing Queue
- Asynchronous processing based on task queue
- Task priority management
- Task status tracking (waiting, processing, completed, failed)
- Task retry mechanism

#### 2.1.3 Parallel Processor
- Multi-process / multi-thread parallel processing
- Resource utilization optimization
- Processing progress monitoring

### 2.2 Data Management Components

#### 2.2.1 Data Storage
- Paper metadata storage
- Detection result storage
- Semantic extraction result storage
- Image and caption information storage

#### 2.2.2 Data Integration Engine
- Cross-paper data association
- Entity alignment (identifying and merging the same entity across different papers)
- Relation integration
- Data conflict resolution

#### 2.2.3 Data Cleaning and Standardization
- Unit standardization (e.g., temperature degC/degF conversion)
- Naming standardization (e.g., element symbol unification)
- Numerical range validation
- Outlier detection and handling

### 2.3 Enhanced Image Extraction Components

#### 2.3.1 Image Detection Enhancement
- YOLOv11 detection optimization
- Supports multi-class image detection
- Improved detection accuracy

#### 2.3.2 Caption Extraction
- PaddleOCR integration
- Caption text recognition
- Caption-image association

#### 2.3.3 Image Content Analysis
- Image feature extraction
- Image similarity analysis
- Cross-paper image comparison

## 3. Data Model Design

### 3.1 Paper Metadata Model

```python
class PaperMetadata:
    paper_id: str
    filename: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    file_path: str
    status: str  # waiting, processing, completed, failed
    processing_start_time: Optional[str]
    processing_end_time: Optional[str]
    error_message: Optional[str]
```

### 3.2 Integrated Data Model

```python
class IntegratedEntity:
    entity_id: str
    original_entities: List[str]  # Original entity IDs from different papers
    entity_type: str
    standard_name: str
    aliases: List[str]  # Aliases in different papers
    properties: Dict[str, Any]
    evidence: List[str]  # List of paper IDs supporting this entity

class IntegratedRelation:
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    polarity: Optional[str]
    strength: str
    confidence: float
    evidence: List[str]  # List of paper IDs supporting this relation
    paper_sources: List[str]  # Source paper ID list
```

## 4. Processing Flow Design

### 4.1 Batch Processing Flow

1. User uploads multiple PDF files
2. System generates unique task ID and paper ID
3. Task is added to the processing queue
4. Parallel processor fetches tasks from the queue
5. For each PDF, execute the following:
   a. YOLO detection
   b. Image and caption extraction
   c. Semantic extraction
6. Processing results are stored in the database
7. Data integration engine processes all results
8. Generate integrated knowledge graph and causal hypothesis graph
9. User views results

### 4.2 Data Integration Flow

1. Entity identification and alignment
   - Entity matching based on name similarity
   - Entity matching based on attribute similarity
   - Entity merging and deduplication

2. Relation integration
   - Relation type standardization
   - Relation strength calculation (based on multi-paper evidence)
   - Relation conflict resolution

3. Data standardization
   - Unit conversion
   - Naming convention unification
   - Numerical format standardization

## 5. System Interface Design

### 5.1 API Interfaces

- `/api/batch/upload` - Batch upload PDF files
- `/api/batch/tasks` - Get task list
- `/api/batch/task/{task_id}` - Get task details
- `/api/batch/start` - Start batch processing
- `/api/batch/cancel/{task_id}` - Cancel task
- `/api/integrated/data` - Get integrated data
- `/api/integrated/kg` - Get integrated knowledge graph
- `/api/integrated/chg` - Get integrated causal hypothesis graph

### 5.2 UI Interface

- Batch upload area
- Task status monitoring panel
- Processing progress visualization
- Integrated data display
- Knowledge graph and causal hypothesis graph visualization

## 6. Technology Selection

### 6.1 Batch Processing
- **Task Queue**: Celery
- **Storage**: SQLite/PostgreSQL
- **Parallel Processing**: Multi-process pool

### 6.2 Data Integration
- **Entity Alignment**: Rule-based and similarity algorithms
- **Data Cleaning**: Pandas
- **Standardization**: Custom rule engine

### 6.3 Image Enhancement
- **YOLOv11**: Already integrated
- **PaddleOCR**: New integration
- **Image Analysis**: OpenCV

## 7. Performance Optimization

- Parallel processing optimization
- Memory usage optimization
- Database query optimization
- Caching mechanism

## 8. Extensibility Considerations

- Support for new detection model integration
- Support for new OCR engine integration
- Support for new data type processing
- Support for distributed deployment

## 9. Security Considerations

- File upload security
- Data access control
- Task execution isolation
- Error handling and logging
