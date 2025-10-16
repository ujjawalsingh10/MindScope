from dataclasses import dataclass

"""
artifact_entity: output results (metadata and file paths)
defines what output each stage will produce and pass to the next stage
"""

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str