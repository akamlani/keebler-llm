import transformers 
from   typing import List 



def get_task_mapping():
    return dict(
        tokenizer         = transformers.AutoTokenizer.from_pretrained,           #"bert-base-cased"
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained,
        masked_lm         = transformers.AutoModelForMaskedLM.from_pretrained,    #"bert-base-uncased" 
        sequence_clf      = transformers.AutoModelForSequenceClassification
    )

def get_tasks_available() -> List[str]:
    return transformers.pipelines.PIPELINE_REGISTRY.get_supported_tasks()
