from FVHandler import FVHandler, FVHandlerBatch

_service = FVHandlerBatch()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    if data is None:
        return None
    
    processed_data = _service.preprocess(data)
    inferred_data = _service.inference(processed_data)
    fvec_list = _service.postprocess(inferred_data)
    
    return fvec_list